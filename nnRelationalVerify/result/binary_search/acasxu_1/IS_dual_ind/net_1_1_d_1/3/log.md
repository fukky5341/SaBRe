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
execution time: IAR + LP analysis = 1.84 + 0.97 = 2.81 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.7126415, upper bound: 0.7126415


# Binary Search by BASE starts (time budget: 1197.19 seconds, max iter: 100)

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
Binary search time: 49.39 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0017311790288658813


# Individual Split (IS_dual_ind) starts
Time budget: 1147.81 seconds

## Binary search (step 0) starts
Candidate diff: 0.0917747


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7096954, upper bound: 0.6862236
time: 0.30 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7092829, upper bound: 0.7092829
time: 0.30 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.75 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.75
Output dim: 0, lower bound: -0.7096954, upper bound: 0.6862236
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.75
Output dim: 0, lower bound: -0.7092829, upper bound: 0.7092829

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0017378, 0.9577341, -0.0318909, 0.7472411, -0.7489790, 0.9896250
1: -0.1117451, 1.2035816, -0.1545317, 0.9228499, -1.0345950, 1.3581133
2: -0.0625789, 1.0695953, -0.0736544, 0.9239269, -0.9865058, 1.1432498
3: -0.2743888, 1.0868173, -0.3136785, 0.9005898, -1.1749785, 1.4004958
4: -0.2543875, 0.9355542, -0.2507498, 0.9391413, -1.1935288, 1.1863041

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6862236, upper bound: 0.6862236
time: 0.29 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6862236, upper bound: 0.6862236
time: 0.31 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0124662, 0.7046853, -0.0323386, 0.7495631, -0.7620293, 0.7370239
1: -0.1260490, 0.8739170, -0.1552227, 0.9255341, -1.0515832, 1.0291396
2: -0.0592167, 0.8673251, -0.0741289, 0.9267865, -0.9860032, 0.9414539
3: -0.2830637, 0.8561093, -0.3143692, 0.9027434, -1.1858070, 1.1704785
4: -0.2274244, 0.8807485, -0.2515757, 0.9410419, -1.1684663, 1.1323242

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6862236, upper bound: 0.7092829
time: 0.32 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6862236, upper bound: 0.7092829
time: 0.29 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.43 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.43
Output dim: 0, lower bound: -0.6862236, upper bound: 0.6862236
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.43
Output dim: 0, lower bound: -0.6862236, upper bound: 0.6862236
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.43
Output dim: 0, lower bound: -0.6862236, upper bound: 0.7092829
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.43
Output dim: 0, lower bound: -0.6862236, upper bound: 0.7092829

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0017378, 0.9577341, -0.0017378, 0.9577341, -0.9594719, 0.9594719
1: -0.1117451, 1.2035816, -0.1117451, 1.2035816, -1.3153267, 1.3153267
2: -0.0625789, 1.0695953, -0.0625789, 1.0695953, -1.1321743, 1.1321743
3: -0.2743888, 1.0868173, -0.2743888, 1.0868173, -1.3612061, 1.3612061
4: -0.2543875, 0.9355542, -0.2543875, 0.9355542, -1.1899416, 1.1899416

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6756641, upper bound: 0.6516575
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6609718, upper bound: 0.6438791
time: 0.29 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0017378, 0.9577341, -0.0124662, 0.7046853, -0.7064232, 0.9702003
1: -0.1117451, 1.2035816, -0.1260490, 0.8739170, -0.9856621, 1.3296306
2: -0.0625789, 1.0695953, -0.0592167, 0.8673251, -0.9299040, 1.1288121
3: -0.2743888, 1.0868173, -0.2830637, 0.8561093, -1.1304981, 1.3698809
4: -0.2543875, 0.9355542, -0.2274244, 0.8807485, -1.1351360, 1.1629786

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6756641, upper bound: 0.6516575
time: 0.30 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6609718, upper bound: 0.6438791
time: 0.30 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0124662, 0.7046853, -0.0017378, 0.9577341, -0.9702003, 0.7064232
1: -0.1260490, 0.8739170, -0.1117451, 1.2035816, -1.3296306, 0.9856621
2: -0.0592167, 0.8673251, -0.0625789, 1.0695953, -1.1288121, 0.9299040
3: -0.2830637, 0.8561093, -0.2743888, 1.0868173, -1.3698809, 1.1304981
4: -0.2274244, 0.8807485, -0.2543875, 0.9355542, -1.1629786, 1.1351360

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6851157, upper bound: 0.7078883
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6855405, upper bound: 0.7067790
time: 0.35 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0124662, 0.7046853, -0.0124662, 0.7046853, -0.7171515, 0.7171515
1: -0.1260490, 0.8739170, -0.1260490, 0.8739170, -0.9999660, 0.9999660
2: -0.0592167, 0.8673251, -0.0592167, 0.8673251, -0.9265418, 0.9265418
3: -0.2830637, 0.8561093, -0.2830637, 0.8561093, -1.1391729, 1.1391729
4: -0.2274244, 0.8807485, -0.2274244, 0.8807485, -1.1081729, 1.1081729

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6851157, upper bound: 0.7078883
time: 0.32 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6855405, upper bound: 0.7067790
time: 0.35 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.23 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.23
Output dim: 0, lower bound: -0.6756641, upper bound: 0.6516575
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 3.23
Output dim: 0, lower bound: -0.6609718, upper bound: 0.6438791
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.23
Output dim: 0, lower bound: -0.6756641, upper bound: 0.6516575
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 3.23
Output dim: 0, lower bound: -0.6609718, upper bound: 0.6438791
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.23
Output dim: 0, lower bound: -0.6851157, upper bound: 0.7078883
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.23
Output dim: 0, lower bound: -0.6855405, upper bound: 0.7067790
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.23
Output dim: 0, lower bound: -0.6851157, upper bound: 0.7078883
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.23
Output dim: 0, lower bound: -0.6855405, upper bound: 0.7067790

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0000615, 0.9523284, -0.0017378, 0.9577341, -0.9576726, 0.9540663
1: -0.1095166, 1.1974459, -0.1117451, 1.2035816, -1.3130982, 1.3091910
2: -0.0606787, 1.0631847, -0.0625789, 1.0695953, -1.1302741, 1.1257637
3: -0.2725427, 1.0815248, -0.2743888, 1.0868173, -1.3593600, 1.3559136
4: -0.2520238, 0.9301934, -0.2543875, 0.9355542, -1.1875780, 1.1845809

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6639844, upper bound: 0.6639844
time: 0.31 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6639844, upper bound: 0.6639844
time: 0.28 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0000615, 0.9523284, -0.0124662, 0.7046853, -0.7046238, 0.9647946
1: -0.1095166, 1.1974459, -0.1260490, 0.8739170, -0.9834336, 1.3234949
2: -0.0606787, 1.0631847, -0.0592167, 0.8673251, -0.9280038, 1.1224015
3: -0.2725427, 1.0815248, -0.2830637, 0.8561093, -1.1286520, 1.3645885
4: -0.2520238, 0.9301934, -0.2274244, 0.8807485, -1.1327723, 1.1576178

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6609718, upper bound: 0.6438791
time: 0.32 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6609718, upper bound: 0.6438791
time: 0.33 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0054424, 0.6499398, -0.0017378, 0.9577341, -0.9522917, 0.6516776
1: -0.1022949, 0.8109143, -0.1117451, 1.2035816, -1.3058765, 0.9226594
2: -0.0393627, 0.7986160, -0.0625789, 1.0695953, -1.1089580, 0.8611949
3: -0.2590857, 0.7998861, -0.2743888, 1.0868173, -1.3459029, 1.0742749
4: -0.1939640, 0.8231770, -0.2543875, 0.9355542, -1.1295183, 1.0775645

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6760281, upper bound: 0.7083998
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6779333, upper bound: 0.7022436
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6779333, upper bound: 0.7072936
time: 0.33 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0004525, 0.7221838, -0.0011210, 0.9538052, -0.9533527, 0.7233049
1: -0.1048932, 0.8939738, -0.1108930, 1.1991937, -1.3040869, 1.0048668
2: -0.0500441, 0.8598962, -0.0618782, 1.0645330, -1.1145771, 0.9217744
3: -0.2612641, 0.8577256, -0.2734051, 1.0833064, -1.3445705, 1.1311307
4: -0.2176540, 0.8560207, -0.2532487, 0.9323727, -1.1500268, 1.1092694

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6515660, upper bound: 0.6751746
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6438791, upper bound: 0.6607286
time: 0.34 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0054424, 0.6499398, -0.0124662, 0.7046853, -0.6992429, 0.6624060
1: -0.1022949, 0.8109143, -0.1260490, 0.8739170, -0.9762119, 0.9369633
2: -0.0393627, 0.7986160, -0.0592167, 0.8673251, -0.9066877, 0.8578327
3: -0.2590857, 0.7998861, -0.2830637, 0.8561093, -1.1151949, 1.0829498
4: -0.1939640, 0.8231770, -0.2274244, 0.8807485, -1.0747125, 1.0506014

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6926397, upper bound: 0.7078883
time: 0.32 seconds

## Relational analysis of IS_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6833646, upper bound: 0.7063542
time: 0.32 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6833646, upper bound: 0.7067790
time: 0.31 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0004525, 0.7221838, -0.0107715, 0.6992106, -0.6987581, 0.7329553
1: -0.1048932, 0.8939738, -0.1236067, 0.8677564, -0.9726496, 1.0175805
2: -0.0500441, 0.8598962, -0.0578458, 0.8597074, -0.9097515, 0.9177420
3: -0.2612641, 0.8577256, -0.2807093, 0.8507382, -1.1120023, 1.1384349
4: -0.2176540, 0.8560207, -0.2250874, 0.8738575, -1.0915115, 1.0811081

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6833646, upper bound: 0.7063542
time: 0.32 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6833646, upper bound: 0.7067790
time: 0.33 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.24 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 0, lower bound: -0.6639844, upper bound: 0.6639844
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 0, lower bound: -0.6639844, upper bound: 0.6639844
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 3.24
Output dim: 0, lower bound: -0.6609718, upper bound: 0.6438791
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 3.24
Output dim: 0, lower bound: -0.6609718, upper bound: 0.6438791
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 0, lower bound: -0.6779333, upper bound: 0.7022436
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 0, lower bound: -0.6779333, upper bound: 0.7072936
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 0, lower bound: -0.6515660, upper bound: 0.6751746
IS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 3.24
Output dim: 0, lower bound: -0.6438791, upper bound: 0.6607286
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 0, lower bound: -0.6833646, upper bound: 0.7063542
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 0, lower bound: -0.6833646, upper bound: 0.7067790
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 0, lower bound: -0.6833646, upper bound: 0.7063542
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 0, lower bound: -0.6833646, upper bound: 0.7067790

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0000615, 0.9523284, 0.0000615, 0.9523284, -0.9522669, 0.9522669
1: -0.1095166, 1.1974459, -0.1095166, 1.1974459, -1.3069625, 1.3069625
2: -0.0606787, 1.0631847, -0.0606787, 1.0631847, -1.1238635, 1.1238635
3: -0.2725427, 1.0815248, -0.2725427, 1.0815248, -1.3540676, 1.3540676
4: -0.2520238, 0.9301934, -0.2520238, 0.9301934, -1.1822172, 1.1822172

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6731502, upper bound: 0.6642457
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6781872, upper bound: 0.6714281
time: 0.31 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0000615, 0.9523284, -0.0512598, 1.8787861, -1.8787246, 1.0035882
1: -0.1095166, 1.1974459, -0.1754494, 2.2964354, -2.4059520, 1.3728952
2: -0.0606787, 1.0631847, -0.1450286, 2.0645037, -2.1251824, 1.2082133
3: -0.2725427, 1.0815248, -0.3347113, 1.8700678, -2.1426105, 1.4162362
4: -0.2520238, 0.9301934, -0.4980223, 1.5064290, -1.7584528, 1.4282157

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6731502, upper bound: 0.6642457
time: 0.29 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6781872, upper bound: 0.6714281
time: 0.31 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0054424, 0.6499398, 0.0197756, 0.9062033, -0.9007609, 0.6301641
1: -0.1022949, 0.8109143, -0.0792882, 1.1450822, -1.2473772, 0.8902025
2: -0.0393627, 0.7986160, -0.0386198, 1.0002036, -1.0395663, 0.8372357
3: -0.2590857, 0.7998861, -0.2423158, 1.0302336, -1.2893193, 1.0422019
4: -0.1939640, 0.8231770, -0.2225227, 0.8589398, -1.0529038, 1.0456997

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5

Time for candidate selection: 1.72 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6776237, upper bound: 0.6663589
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6540380, upper bound: 0.6756027
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0054424, 0.6499398, -0.0046289, 1.0400095, -1.0345671, 0.6545687
1: -0.1022949, 0.8109143, -0.1109180, 1.3149409, -1.4172359, 0.9218323
2: -0.0393627, 0.7986160, -0.0750914, 1.1419265, -1.1812892, 0.8737074
3: -0.2590857, 0.7998861, -0.2722509, 1.1791039, -1.4381895, 1.0721370
4: -0.1939640, 0.8231770, -0.2956257, 0.9621377, -1.1561017, 1.1188027

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5

Time for candidate selection: 1.70 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6776237, upper bound: 0.6773665
time: 0.33 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6540380, upper bound: 0.6866103
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0004525, 0.7221838, 0.0006769, 0.9484155, -0.9479630, 0.7215070
1: -0.1048932, 0.8939738, -0.1086650, 1.1930711, -1.2979643, 1.0026388
2: -0.0500441, 0.8598962, -0.0599797, 1.0581393, -1.1081834, 0.9198759
3: -0.2612641, 0.8577256, -0.2715602, 1.0780238, -1.3392879, 1.1292858
4: -0.2176540, 0.8560207, -0.2508883, 0.9270457, -1.1446997, 1.1069090

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6438791, upper bound: 0.6607286
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6438791, upper bound: 0.6607286
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0054424, 0.6499398, 0.0054424, 0.6499398, -0.6444974, 0.6444974
1: -0.1022949, 0.8109143, -0.1022949, 0.8109143, -0.9132092, 0.9132092
2: -0.0393627, 0.7986160, -0.0393627, 0.7986160, -0.8379787, 0.8379787
3: -0.2590857, 0.7998861, -0.2590857, 0.7998861, -1.0589718, 1.0589718
4: -0.1939640, 0.8231770, -0.1939640, 0.8231770, -1.0171410, 1.0171410

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5

Time for candidate selection: 1.76 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6885325, upper bound: 0.6729682
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6703457, upper bound: 0.6728899
time: 0.35 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0054424, 0.6499398, 0.0004525, 0.7221838, -0.7167414, 0.6494873
1: -0.1022949, 0.8109143, -0.1048932, 0.8939738, -0.9962687, 0.9158075
2: -0.0393627, 0.7986160, -0.0500441, 0.8598962, -0.8992589, 0.8486601
3: -0.2590857, 0.7998861, -0.2612641, 0.8577256, -1.1168113, 1.0611502
4: -0.1939640, 0.8231770, -0.2176540, 0.8560207, -1.0499847, 1.0408310

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5

Time for candidate selection: 1.74 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6885325, upper bound: 0.6729682
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6703457, upper bound: 0.6728899
time: 0.34 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0004525, 0.7221838, 0.0054424, 0.6499398, -0.6494873, 0.7167414
1: -0.1048932, 0.8939738, -0.1022949, 0.8109143, -0.9158075, 0.9962687
2: -0.0500441, 0.8598962, -0.0393627, 0.7986160, -0.8486601, 0.8992589
3: -0.2612641, 0.8577256, -0.2590857, 0.7998861, -1.0611502, 1.1168113
4: -0.2176540, 0.8560207, -0.1939640, 0.8231770, -1.0408310, 1.0499847

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 47

Time for candidate selection: 1.74 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6825935, upper bound: 0.6732698
time: 0.34 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6615940, upper bound: 0.6811418
time: 0.33 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0004525, 0.7221838, 0.0004525, 0.7221838, -0.7217313, 0.7217313
1: -0.1048932, 0.8939738, -0.1048932, 0.8939738, -0.9988670, 0.9988670
2: -0.0500441, 0.8598962, -0.0500441, 0.8598962, -0.9099402, 0.9099402
3: -0.2612641, 0.8577256, -0.2612641, 0.8577256, -1.1189897, 1.1189897
4: -0.2176540, 0.8560207, -0.2176540, 0.8560207, -1.0736747, 1.0736747

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 47

Time for candidate selection: 1.74 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6825935, upper bound: 0.6737599
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6615940, upper bound: 0.6836975
time: 0.33 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.04 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -0.6731502, upper bound: 0.6642457
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -0.6781872, upper bound: 0.6714281
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -0.6731502, upper bound: 0.6642457
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -0.6781872, upper bound: 0.6714281
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -0.6776237, upper bound: 0.6663589
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -0.6540380, upper bound: 0.6756027
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -0.6776237, upper bound: 0.6773665
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -0.6540380, upper bound: 0.6866103
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.04
Output dim: 0, lower bound: -0.6438791, upper bound: 0.6607286
IS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.04
Output dim: 0, lower bound: -0.6438791, upper bound: 0.6607286
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -0.6885325, upper bound: 0.6729682
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -0.6703457, upper bound: 0.6728899
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -0.6885325, upper bound: 0.6729682
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -0.6703457, upper bound: 0.6728899
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -0.6825935, upper bound: 0.6732698
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -0.6615940, upper bound: 0.6811418
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -0.6825935, upper bound: 0.6737599
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -0.6615940, upper bound: 0.6836975

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0213993, 0.9013143, 0.0000615, 0.9523284, -0.9309292, 0.9012527
1: -0.0770538, 1.1394897, -0.1095166, 1.1974459, -1.2744997, 1.2490063
2: -0.0367112, 0.9943478, -0.0606787, 1.0631847, -1.0998960, 1.0550265
3: -0.2404699, 1.0253115, -0.2725427, 1.0815248, -1.3219948, 1.2978542
4: -0.2202464, 0.8539461, -0.2520238, 0.9301934, -1.1504399, 1.1059699

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6736547, upper bound: 0.6736547
time: 0.31 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6736547, upper bound: 0.6786917
time: 0.32 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0028670, 1.0353181, 0.0006769, 0.9484155, -0.9512825, 1.0346413
1: -0.1088188, 1.3095722, -0.1086650, 1.1930711, -1.3018899, 1.4182372
2: -0.0732558, 1.1363764, -0.0599797, 1.0581393, -1.1313951, 1.1963561
3: -0.2704966, 1.1744311, -0.2715602, 1.0780238, -1.3485204, 1.4459913
4: -0.2937192, 0.9576517, -0.2508883, 0.9270457, -1.2207649, 1.2085401

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6786917, upper bound: 0.6808372
time: 0.31 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6786917, upper bound: 0.6858741
time: 0.31 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0213993, 0.9013143, -0.0512598, 1.8787861, -1.8573868, 0.9525740
1: -0.0770538, 1.1394897, -0.1754494, 2.2964354, -2.3734891, 1.3149390
2: -0.0367112, 0.9943478, -0.1450286, 2.0645037, -2.1012149, 1.1393764
3: -0.2404699, 1.0253115, -0.3347113, 1.8700678, -2.1105378, 1.3600228
4: -0.2202464, 0.8539461, -0.4980223, 1.5064290, -1.7266754, 1.3519684

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47

Time for candidate selection: 1.67 seconds

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6219948, upper bound: 0.6639361
time: 0.30 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6449059, upper bound: 0.6571256
time: 0.31 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0028670, 1.0353181, -0.0497918, 1.8739424, -1.8768094, 1.0851099
1: -0.1088188, 1.3095722, -0.1735365, 2.2909322, -2.3997509, 1.4831088
2: -0.0732558, 1.1363764, -0.1432242, 2.0585322, -2.1317880, 1.2796006
3: -0.2704966, 1.1744311, -0.3333235, 1.8649230, -2.1354196, 1.5077546
4: -0.2937192, 0.9576517, -0.4949807, 1.5007753, -1.7944945, 1.4526324

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6400006, upper bound: 0.6714281
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6441794, upper bound: 0.6534000
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, 0.0197756, 0.9062033, -0.8950152, 0.5450859
1: -0.0944340, 0.7130072, -0.0792882, 1.1450822, -1.2395163, 0.7922955
2: -0.0312903, 0.7124612, -0.0386198, 1.0002036, -1.0314939, 0.7510810
3: -0.2500148, 0.7301772, -0.2423158, 1.0302336, -1.2802484, 0.9724930
4: -0.1787794, 0.7720095, -0.2225227, 0.8589398, -1.0377191, 0.9945322

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6731086, upper bound: 0.6661886
time: 0.33 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6284010, upper bound: 0.6618038
time: 0.32 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47

Time for candidate selection: 2.37 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6663589
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6663589
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, 0.0197756, 0.9062033, -0.9367702, 0.6508826
1: -0.1440713, 0.8347254, -0.0792882, 1.1450822, -1.2891536, 0.9140136
2: -0.0742025, 0.8280591, -0.0386198, 1.0002036, -1.0744061, 0.8666789
3: -0.2901301, 0.8312570, -0.2423158, 1.0302336, -1.3203638, 1.0735729
4: -0.2451730, 0.8507983, -0.2225227, 0.8589398, -1.1041127, 1.0733211

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6530712, upper bound: 0.6756027
time: 0.33 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6042474, upper bound: 0.6708263
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47

Time for candidate selection: 2.41 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6756027
time: 0.31 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6756027
time: 0.32 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, -0.0046289, 1.0400095, -1.0288215, 0.5694904
1: -0.0944340, 0.7130072, -0.1109180, 1.3149409, -1.4093750, 0.8239253
2: -0.0312903, 0.7124612, -0.0750914, 1.1419265, -1.1732168, 0.7875526
3: -0.2500148, 0.7301772, -0.2722509, 1.1791039, -1.4291186, 1.0024281
4: -0.1787794, 0.7720095, -0.2956257, 0.9621377, -1.1409171, 1.0676352

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6753027, upper bound: 0.6771959
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 5

Time for candidate selection: 2.05 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6444212, upper bound: 0.6773662
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6444212, upper bound: 0.6773662
time: 0.32 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, -0.0046289, 1.0400095, -1.0705764, 0.6752871
1: -0.1440713, 0.8347254, -0.1109180, 1.3149409, -1.4590123, 0.9456434
2: -0.0742025, 0.8280591, -0.0750914, 1.1419265, -1.2161291, 0.9031505
3: -0.2901301, 0.8312570, -0.2722509, 1.1791039, -1.4692340, 1.1035080
4: -0.2451730, 0.8507983, -0.2956257, 0.9621377, -1.2073107, 1.1464241

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6547696, upper bound: 0.6866100
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 5

Time for candidate selection: 2.06 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6444212, upper bound: 0.6866100
time: 0.33 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6444212, upper bound: 0.6866100
time: 0.37 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0158405, 0.4982040, 0.0054424, 0.6499398, -0.6340992, 0.4927616
1: -0.0890739, 0.6250523, -0.1022949, 0.8109143, -0.8999882, 0.7273472
2: -0.0220215, 0.6451663, -0.0393627, 0.7986160, -0.8206375, 0.6845290
3: -0.2441075, 0.6654664, -0.2590857, 0.7998861, -1.0439936, 0.9245521
4: -0.1598573, 0.7338556, -0.1939640, 0.8231770, -0.9830343, 0.9278196

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6900481, upper bound: 0.6845138
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6901820, upper bound: 0.6886574
time: 0.37 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0892951, 0.7884323, 0.0054424, 0.6499398, -0.7392349, 0.7829899
1: -0.2055373, 0.9655911, -0.1022949, 0.8109143, -1.0164516, 1.0678861
2: -0.1576910, 0.9697834, -0.0393627, 0.7986160, -0.9563070, 1.0091461
3: -0.3306990, 0.9586977, -0.2590857, 0.7998861, -1.1305851, 1.2177833
4: -0.3441186, 0.9710234, -0.1939640, 0.8231770, -1.1672956, 1.1649873

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6750211, upper bound: 0.6849745
time: 0.36 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6752757, upper bound: 0.6892572
time: 0.35 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0158405, 0.4982040, 0.0004525, 0.7221838, -0.7063433, 0.4977515
1: -0.0890739, 0.6250523, -0.1048932, 0.8939738, -0.9830477, 0.7299455
2: -0.0220215, 0.6451663, -0.0500441, 0.8598962, -0.8819177, 0.6952104
3: -0.2441075, 0.6654664, -0.2612641, 0.8577256, -1.1018331, 0.9267305
4: -0.1598573, 0.7338556, -0.2176540, 0.8560207, -1.0158780, 0.9515096

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6870449, upper bound: 0.6728652
time: 0.36 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47

Time for candidate selection: 2.16 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6587127, upper bound: 0.6726318
time: 0.31 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6674482, upper bound: 0.6578935
time: 0.34 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0892951, 0.7884323, 0.0004525, 0.7221838, -0.8114790, 0.7879798
1: -0.2055373, 0.9655911, -0.1048932, 0.8939738, -1.0995111, 1.0704844
2: -0.1576910, 0.9697834, -0.0500441, 0.8598962, -1.0175872, 1.0198275
3: -0.3306990, 0.9586977, -0.2612641, 0.8577256, -1.1884246, 1.2199618
4: -0.3441186, 0.9710234, -0.2176540, 0.8560207, -1.2001393, 1.1886773

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6696861, upper bound: 0.6727927
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47

Time for candidate selection: 2.22 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6429276, upper bound: 0.6727466
time: 0.33 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6525701, upper bound: 0.6583646
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0061345, 0.6507710, 0.0054424, 0.6499398, -0.6438053, 0.6453286
1: -0.0976210, 0.8106803, -0.1022949, 0.8109143, -0.9085352, 0.9129752
2: -0.0417109, 0.7880348, -0.0393627, 0.7986160, -0.8403268, 0.8273975
3: -0.2534082, 0.7965333, -0.2590857, 0.7998861, -1.0532943, 1.0556190
4: -0.2023642, 0.8128391, -0.1939640, 0.8231770, -1.0255412, 1.0068030

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 5

Time for candidate selection: 1.93 seconds

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6555374, upper bound: 0.6668210
time: 0.36 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6512620, upper bound: 0.6559544
time: 0.35 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0329778, 0.7304660, 0.0054424, 0.6499398, -0.6829176, 0.7250236
1: -0.1447692, 0.9029138, -0.1022949, 0.8109143, -0.9556835, 1.0052087
2: -0.0813575, 0.8816222, -0.0393627, 0.7986160, -0.8799735, 0.9209849
3: -0.2928538, 0.8801655, -0.2590857, 0.7998861, -1.0927399, 1.1392511
4: -0.2609611, 0.8848863, -0.1939640, 0.8231770, -1.0841380, 1.0788503

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6440740, upper bound: 0.6418644
time: 0.34 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6475655, upper bound: 0.6816905
time: 0.34 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 5

Time for candidate selection: 2.52 seconds

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6383973, upper bound: 0.6757573
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6337648, upper bound: 0.6653052
time: 0.35 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0061345, 0.6507710, 0.0004525, 0.7221838, -0.7160493, 0.6503185
1: -0.0976210, 0.8106803, -0.1048932, 0.8939738, -0.9915948, 0.9155735
2: -0.0417109, 0.7880348, -0.0500441, 0.8598962, -0.9016070, 0.8380789
3: -0.2534082, 0.7965333, -0.2612641, 0.8577256, -1.1111338, 1.0577974
4: -0.2023642, 0.8128391, -0.2176540, 0.8560207, -1.0583849, 1.0304930

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47

Time for candidate selection: 1.91 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6530561, upper bound: 0.6737598
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6530561, upper bound: 0.6737598
time: 0.35 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0329778, 0.7304660, 0.0004525, 0.7221838, -0.7551616, 0.7300135
1: -0.1447692, 0.9029138, -0.1048932, 0.8939738, -1.0387430, 1.0078070
2: -0.0813575, 0.8816222, -0.0500441, 0.8598962, -0.9412537, 0.9316663
3: -0.2928538, 0.8801655, -0.2612641, 0.8577256, -1.1505795, 1.1414295
4: -0.2609611, 0.8848863, -0.2176540, 0.8560207, -1.1169817, 1.1025403

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6509603, upper bound: 0.6519631
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6361090, upper bound: 0.6398205
time: 0.36 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.35 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 0, lower bound: -0.6736547, upper bound: 0.6736547
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 0, lower bound: -0.6736547, upper bound: 0.6786917
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 0, lower bound: -0.6786917, upper bound: 0.6808372
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 0, lower bound: -0.6786917, upper bound: 0.6858741
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 0, lower bound: -0.6219948, upper bound: 0.6639361
IS_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.35
Output dim: 0, lower bound: -0.6449059, upper bound: 0.6571256
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 0, lower bound: -0.6400006, upper bound: 0.6714281
IS_A1_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.35
Output dim: 0, lower bound: -0.6441794, upper bound: 0.6534000
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6663589
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6663589
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6756027
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6756027
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 0, lower bound: -0.6444212, upper bound: 0.6773662
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 0, lower bound: -0.6444212, upper bound: 0.6773662
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 0, lower bound: -0.6444212, upper bound: 0.6866100
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 0, lower bound: -0.6444212, upper bound: 0.6866100
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 0, lower bound: -0.6900481, upper bound: 0.6845138
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 0, lower bound: -0.6901820, upper bound: 0.6886574
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 0, lower bound: -0.6750211, upper bound: 0.6849745
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 0, lower bound: -0.6752757, upper bound: 0.6892572
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 0, lower bound: -0.6587127, upper bound: 0.6726318
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 0, lower bound: -0.6674482, upper bound: 0.6578935
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 0, lower bound: -0.6429276, upper bound: 0.6727466
IS_A2_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.35
Output dim: 0, lower bound: -0.6525701, upper bound: 0.6583646
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 0, lower bound: -0.6555374, upper bound: 0.6668210
IS_A2_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.35
Output dim: 0, lower bound: -0.6512620, upper bound: 0.6559544
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 0, lower bound: -0.6383973, upper bound: 0.6757573
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 0, lower bound: -0.6337648, upper bound: 0.6653052
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 0, lower bound: -0.6530561, upper bound: 0.6737598
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 0, lower bound: -0.6530561, upper bound: 0.6737598
IS_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.35
Output dim: 0, lower bound: -0.6509603, upper bound: 0.6519631
IS_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.35
Output dim: 0, lower bound: -0.6361090, upper bound: 0.6398205

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0213993, 0.9013143, 0.0213993, 0.9013143, -0.8799150, 0.8799150
1: -0.0770538, 1.1394897, -0.0770538, 1.1394897, -1.2165434, 1.2165434
2: -0.0367112, 0.9943478, -0.0367112, 0.9943478, -1.0310590, 1.0310590
3: -0.2404699, 1.0253115, -0.2404699, 1.0253115, -1.2657814, 1.2657814
4: -0.2202464, 0.8539461, -0.2202464, 0.8539461, -1.0741925, 1.0741925

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 47

Time for candidate selection: 1.33 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6733452, upper bound: 0.6341388
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6461180, upper bound: 0.6461180
time: 0.31 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0213993, 0.9013143, -0.0028670, 1.0353181, -1.0139189, 0.9041812
1: -0.0770538, 1.1394897, -0.1088188, 1.3095722, -1.3866260, 1.2483084
2: -0.0367112, 0.9943478, -0.0732558, 1.1363764, -1.1730876, 1.0676036
3: -0.2404699, 1.0253115, -0.2704966, 1.1744311, -1.4149010, 1.2958081
4: -0.2202464, 0.8539461, -0.2937192, 0.9576517, -1.1778982, 1.1476653

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 47

Time for candidate selection: 1.45 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6733452, upper bound: 0.6450348
time: 0.31 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6461180, upper bound: 0.6570139
time: 0.31 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0028670, 1.0353181, 0.0213993, 0.9013143, -0.9041812, 1.0139189
1: -0.1088188, 1.3095722, -0.0770538, 1.1394897, -1.2483084, 1.3866260
2: -0.0732558, 1.1363764, -0.0367112, 0.9943478, -1.0676036, 1.1730876
3: -0.2704966, 1.1744311, -0.2404699, 1.0253115, -1.2958081, 1.4149010
4: -0.2937192, 0.9576517, -0.2202464, 0.8539461, -1.1476653, 1.1778982

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5

Time for candidate selection: 1.39 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6733452, upper bound: 0.6365467
time: 0.30 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6570136, upper bound: 0.6530225
time: 0.33 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0028670, 1.0353181, -0.0028670, 1.0353181, -1.0381851, 1.0381851
1: -0.1088188, 1.3095722, -0.1088188, 1.3095722, -1.4183910, 1.4183910
2: -0.0732558, 1.1363764, -0.0732558, 1.1363764, -1.2096322, 1.2096322
3: -0.2704966, 1.1744311, -0.2704966, 1.1744311, -1.4449277, 1.4449277
4: -0.2937192, 0.9576517, -0.2937192, 0.9576517, -1.2513709, 1.2513709

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5

Time for candidate selection: 1.42 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6733452, upper bound: 0.6468780
time: 0.31 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6570139, upper bound: 0.6639186
time: 0.31 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0213993, 0.9013143, -0.0368996, 1.6852140, -1.6638148, 0.9382138
1: -0.0770538, 1.1394897, -0.1573129, 2.0717196, -2.1487734, 1.2968025
2: -0.0367112, 0.9943478, -0.1241670, 1.8526795, -1.8893907, 1.1185148
3: -0.2404699, 1.0253115, -0.3175256, 1.7035871, -1.9440570, 1.3428371
4: -0.2202464, 0.8539461, -0.4543926, 1.3758390, -1.5960854, 1.3083386

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 47

Time for candidate selection: 1.40 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6219948, upper bound: 0.6451464
time: 0.31 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6219948, upper bound: 0.6571256
time: 0.30 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0028670, 1.0353181, -0.0487354, 1.8702989, -1.8731658, 1.0840535
1: -0.1088188, 1.3095722, -0.1722147, 2.2867708, -2.3955896, 1.4817870
2: -0.0732558, 1.1363764, -0.1420550, 2.0543475, -2.1276033, 1.2784314
3: -0.2704966, 1.1744311, -0.3320103, 1.8612552, -2.1317518, 1.5064414
4: -0.2937192, 0.9576517, -0.4934373, 1.4973459, -1.7910651, 1.4510890

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6400006, upper bound: 0.6526529
time: 0.33 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6400006, upper bound: 0.6534000
time: 0.31 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, 0.0248351, 0.8239982, -0.8128102, 0.5400264
1: -0.0944340, 0.7130072, -0.0720310, 1.0479953, -1.1424294, 0.7850382
2: -0.0312903, 0.7124612, -0.0307086, 0.9141164, -0.9454067, 0.7431698
3: -0.2500148, 0.7301772, -0.2340453, 0.9604603, -1.2104751, 0.9642225
4: -0.1787794, 0.7720095, -0.2052969, 0.8165774, -0.9953567, 0.9773064

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5

Time for candidate selection: 1.75 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6656445, upper bound: 0.6663589
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6663589
time: 0.33 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, -0.0167825, 0.9250188, -0.9138308, 0.5816441
1: -0.0944340, 0.7130072, -0.1252913, 1.1663461, -1.2607801, 0.8382986
2: -0.0312903, 0.7124612, -0.0773900, 1.0296290, -1.0609193, 0.7898512
3: -0.2500148, 0.7301772, -0.2777247, 1.0633786, -1.3133934, 1.0079019
4: -0.1787794, 0.7720095, -0.2678475, 0.8997647, -1.0785441, 1.0398570

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5

Time for candidate selection: 1.77 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6776237, upper bound: 0.6663589
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6540380, upper bound: 0.6663589
time: 0.34 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, 0.0248351, 0.8239982, -0.8545651, 0.6458231
1: -0.1440713, 0.8347254, -0.0720310, 1.0479953, -1.1920667, 0.9067564
2: -0.0742025, 0.8280591, -0.0307086, 0.9141164, -0.9883189, 0.8587677
3: -0.2901301, 0.8312570, -0.2340453, 0.9604603, -1.2505904, 1.0653024
4: -0.2451730, 0.8507983, -0.2052969, 0.8165774, -1.0617504, 1.0560951

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5

Time for candidate selection: 1.80 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6663589
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6756027
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, -0.0167825, 0.9250188, -0.9555857, 0.6874408
1: -0.1440713, 0.8347254, -0.1252913, 1.1663461, -1.3104174, 0.9600167
2: -0.0742025, 0.8280591, -0.0773900, 1.0296290, -1.1038315, 0.9054491
3: -0.2901301, 0.8312570, -0.2777247, 1.0633786, -1.3535087, 1.1089818
4: -0.2451730, 0.8507983, -0.2678475, 0.8997647, -1.1449378, 1.1186459

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5

Time for candidate selection: 1.78 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6540380, upper bound: 0.6663589
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6540380, upper bound: 0.6756027
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, 0.0006924, 0.9639854, -0.9527974, 0.5641692
1: -0.0944340, 0.7130072, -0.1038828, 1.2270458, -1.3214798, 0.8168900
2: -0.0312903, 0.7124612, -0.0676236, 1.0606856, -1.0919759, 0.7800848
3: -0.2500148, 0.7301772, -0.2647138, 1.1143060, -1.3643208, 0.9948910
4: -0.1787794, 0.7720095, -0.2805611, 0.9134201, -1.0921994, 1.0525706

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6533288, upper bound: 0.6773035
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5

Time for candidate selection: 2.22 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6680524, upper bound: 0.6773662
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6444212, upper bound: 0.6773662
time: 0.34 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, -0.0360885, 1.0472777, -1.0360897, 0.6009500
1: -0.0944340, 0.7130072, -0.1492593, 1.3218184, -1.4162524, 0.8622665
2: -0.0312903, 0.7124612, -0.1066136, 1.1590223, -1.1903126, 0.8190749
3: -0.2500148, 0.7301772, -0.3045232, 1.1983278, -1.4483426, 1.0347004
4: -0.1787794, 0.7720095, -0.3275616, 0.9970202, -1.1757996, 1.0995711

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6533288, upper bound: 0.6773035
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5

Time for candidate selection: 2.20 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6845282, upper bound: 0.6773662
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6608970, upper bound: 0.6773662
time: 0.32 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, 0.0006924, 0.9639854, -0.9945524, 0.6699659
1: -0.1440713, 0.8347254, -0.1038828, 1.2270458, -1.3711171, 0.9386082
2: -0.0742025, 0.8280591, -0.0676236, 1.0606856, -1.1348882, 0.8956828
3: -0.2901301, 0.8312570, -0.2647138, 1.1143060, -1.4044361, 1.0959709
4: -0.2451730, 0.8507983, -0.2805611, 0.9134201, -1.1585931, 1.1313593

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6298693, upper bound: 0.6866100
time: 0.33 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5

Time for candidate selection: 2.13 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6444212, upper bound: 0.6773662
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6444212, upper bound: 0.6866100
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, -0.0360885, 1.0472777, -1.0778446, 0.7067467
1: -0.1440713, 0.8347254, -0.1492593, 1.3218184, -1.4658897, 0.9839847
2: -0.0742025, 0.8280591, -0.1066136, 1.1590223, -1.2332249, 0.9346728
3: -0.2901301, 0.8312570, -0.3045232, 1.1983278, -1.4884579, 1.1357803
4: -0.2451730, 0.8507983, -0.3275616, 0.9970202, -1.2421932, 1.1783600

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6298693, upper bound: 0.6866100
time: 0.33 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5

Time for candidate selection: 2.15 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6608970, upper bound: 0.6773662
time: 0.33 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6608970, upper bound: 0.6866100
time: 0.32 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0158405, 0.4982040, 0.0073309, 0.6935004, -0.6776599, 0.4908731
1: -0.0890739, 0.6250523, -0.0999169, 0.8545845, -0.9436584, 0.7249692
2: -0.0220215, 0.6451663, -0.0374527, 0.8373916, -0.8594131, 0.6826190
3: -0.2441075, 0.6654664, -0.2552719, 0.8256997, -1.0698073, 0.9207383
4: -0.1598573, 0.7338556, -0.1935482, 0.8381348, -0.9979921, 0.9274038

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6873946, upper bound: 0.6845138
time: 0.36 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6873948, upper bound: 0.6845138
time: 0.33 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0158405, 0.4982040, 0.0294173, 0.5563495, -0.5405090, 0.4687867
1: -0.0890739, 0.6250523, -0.0742822, 0.6929038, -0.7819777, 0.6993344
2: -0.0220215, 0.6451663, 0.0004599, 0.7029566, -0.7249781, 0.6447064
3: -0.2441075, 0.6654664, -0.2323704, 0.6962168, -0.9403243, 0.8978368
4: -0.1598573, 0.7338556, -0.1339579, 0.7538315, -0.9136888, 0.8678135

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6873946, upper bound: 0.6882732
time: 0.33 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6873948, upper bound: 0.6886574
time: 0.40 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0892951, 0.7884323, 0.0073309, 0.6935004, -0.7827955, 0.7811014
1: -0.2055373, 0.9655911, -0.0999169, 0.8545845, -1.0601218, 1.0655081
2: -0.1576910, 0.9697834, -0.0374527, 0.8373916, -0.9950826, 1.0072361
3: -0.3306990, 0.9586977, -0.2552719, 0.8256997, -1.1563988, 1.2139696
4: -0.3441186, 0.9710234, -0.1935482, 0.8381348, -1.1822534, 1.1645715

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6594252, upper bound: 0.6808003
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6594252, upper bound: 0.6849745
time: 0.35 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0892951, 0.7884323, 0.0294173, 0.5563495, -0.6456447, 0.7590150
1: -0.2055373, 0.9655911, -0.0742822, 0.6929038, -0.8984411, 1.0398734
2: -0.1576910, 0.9697834, 0.0004599, 0.7029566, -0.8606476, 0.9693235
3: -0.3306990, 0.9586977, -0.2323704, 0.6962168, -1.0269158, 1.1910681
4: -0.3441186, 0.9710234, -0.1339579, 0.7538315, -1.0979501, 1.1049812

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6594252, upper bound: 0.6844697
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6594252, upper bound: 0.6892572
time: 0.35 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0158405, 0.4982040, 0.0061345, 0.6507710, -0.6349305, 0.4920695
1: -0.0890739, 0.6250523, -0.0976210, 0.8106803, -0.8997542, 0.7226732
2: -0.0220215, 0.6451663, -0.0417109, 0.7880348, -0.8100563, 0.6868772
3: -0.2441075, 0.6654664, -0.2534082, 0.7965333, -1.0406408, 0.9188746
4: -0.1598573, 0.7338556, -0.2023642, 0.8128391, -0.9726964, 0.9362198

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5

Time for candidate selection: 2.08 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6587127, upper bound: 0.6726318
time: 0.32 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6429276, upper bound: 0.6725800
time: 0.33 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0158405, 0.4982040, -0.0329778, 0.7304660, -0.7146255, 0.5311818
1: -0.0890739, 0.6250523, -0.1447692, 0.9029138, -0.9919877, 0.7698215
2: -0.0220215, 0.6451663, -0.0813575, 0.8816222, -0.9036437, 0.7265238
3: -0.2441075, 0.6654664, -0.2928538, 0.8801655, -1.1242729, 0.9583203
4: -0.1598573, 0.7338556, -0.2609611, 0.8848863, -1.0447435, 0.9948167

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6647639, upper bound: 0.6379180
time: 0.36 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5

Time for candidate selection: 2.26 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6674482, upper bound: 0.6578935
time: 0.33 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6525701, upper bound: 0.6578416
time: 0.38 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0892951, 0.7884323, 0.0061345, 0.6507710, -0.7400662, 0.7822978
1: -0.2055373, 0.9655911, -0.0976210, 0.8106803, -1.0162176, 1.0632122
2: -0.1576910, 0.9697834, -0.0417109, 0.7880348, -0.9457258, 1.0114943
3: -0.3306990, 0.9586977, -0.2534082, 0.7965333, -1.1272323, 1.2121059
4: -0.3441186, 0.9710234, -0.2023642, 0.8128391, -1.1569576, 1.1733875

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5

Time for candidate selection: 1.91 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6429276, upper bound: 0.6494225
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6429276, upper bound: 0.6583646
time: 0.35 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0061345, 0.6507710, 0.0158405, 0.4982040, -0.4920695, 0.6349305
1: -0.0976210, 0.8106803, -0.0890739, 0.6250523, -0.7226732, 0.8997542
2: -0.0417109, 0.7880348, -0.0220215, 0.6451663, -0.6868772, 0.8100563
3: -0.2534082, 0.7965333, -0.2441075, 0.6654664, -0.9188746, 1.0406408
4: -0.2023642, 0.8128391, -0.1598573, 0.7338556, -0.9362198, 0.9726964

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6493852, upper bound: 0.6569906
time: 0.36 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 47

Time for candidate selection: 2.24 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6512621, upper bound: 0.6559546
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6512621, upper bound: 0.6559546
time: 0.40 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0329778, 0.7304660, 0.0158405, 0.4982040, -0.5311818, 0.7146255
1: -0.1447692, 0.9029138, -0.0890739, 0.6250523, -0.7698215, 0.9919877
2: -0.0813575, 0.8816222, -0.0220215, 0.6451663, -0.7265238, 0.9036437
3: -0.2928538, 0.8801655, -0.2441075, 0.6654664, -0.9583203, 1.1242729
4: -0.2609611, 0.8848863, -0.1598573, 0.7338556, -0.9948167, 1.0447435

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6327927, upper bound: 0.6695331
time: 0.36 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 47

Time for candidate selection: 2.25 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6383973, upper bound: 0.6668210
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6383973, upper bound: 0.6757573
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0329778, 0.7304660, -0.0892951, 0.7884323, -0.8214101, 0.8197612
1: -0.1447692, 0.9029138, -0.2055373, 0.9655911, -1.1103604, 1.1084511
2: -0.0813575, 0.8816222, -0.1576910, 0.9697834, -1.0511409, 1.0393132
3: -0.2928538, 0.8801655, -0.3306990, 0.9586977, -1.2515515, 1.2108644
4: -0.2609611, 0.8848863, -0.3441186, 0.9710234, -1.2319844, 1.2290049

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6328156, upper bound: 0.6653052
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 47

Time for candidate selection: 2.36 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6337648, upper bound: 0.6559546
time: 0.36 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6337648, upper bound: 0.6653052
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0061345, 0.6507710, 0.0061345, 0.6507710, -0.6446365, 0.6446365
1: -0.0976210, 0.8106803, -0.0976210, 0.8106803, -0.9083012, 0.9083012
2: -0.0417109, 0.7880348, -0.0417109, 0.7880348, -0.8297457, 0.8297457
3: -0.2534082, 0.7965333, -0.2534082, 0.7965333, -1.0499415, 1.0499415
4: -0.2023642, 0.8128391, -0.2023642, 0.8128391, -1.0152032, 1.0152032

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 47

Time for candidate selection: 2.03 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6667848, upper bound: 0.6496201
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6436402, upper bound: 0.6436381
time: 0.38 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0061345, 0.6507710, -0.0329778, 0.7304660, -0.7243315, 0.6837488
1: -0.0976210, 0.8106803, -0.1447692, 0.9029138, -1.0005348, 0.9554495
2: -0.0417109, 0.7880348, -0.0813575, 0.8816222, -0.9233330, 0.8693923
3: -0.2534082, 0.7965333, -0.2928538, 0.8801655, -1.1335737, 1.0893872
4: -0.2023642, 0.8128391, -0.2609611, 0.8848863, -1.0872505, 1.0738001

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 47

Time for candidate selection: 2.01 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6667848, upper bound: 0.6496201
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6436402, upper bound: 0.6436381
time: 0.41 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.63 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.63
Output dim: 0, lower bound: -0.6733452, upper bound: 0.6341388
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 0, lower bound: -0.6461180, upper bound: 0.6461180
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.63
Output dim: 0, lower bound: -0.6733452, upper bound: 0.6450348
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 0, lower bound: -0.6461180, upper bound: 0.6570139
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.63
Output dim: 0, lower bound: -0.6733452, upper bound: 0.6365467
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 0, lower bound: -0.6570136, upper bound: 0.6530225
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.63
Output dim: 0, lower bound: -0.6733452, upper bound: 0.6468780
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.63
Output dim: 0, lower bound: -0.6570139, upper bound: 0.6639186
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 0, lower bound: -0.6219948, upper bound: 0.6451464
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 0, lower bound: -0.6219948, upper bound: 0.6571256
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 0, lower bound: -0.6400006, upper bound: 0.6526529
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 0, lower bound: -0.6400006, upper bound: 0.6534000
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.63
Output dim: 0, lower bound: -0.6656445, upper bound: 0.6663589
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.63
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6663589
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.63
Output dim: 0, lower bound: -0.6776237, upper bound: 0.6663589
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.63
Output dim: 0, lower bound: -0.6540380, upper bound: 0.6663589
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.63
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6663589
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.63
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6756027
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.63
Output dim: 0, lower bound: -0.6540380, upper bound: 0.6663589
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.63
Output dim: 0, lower bound: -0.6540380, upper bound: 0.6756027
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.63
Output dim: 0, lower bound: -0.6680524, upper bound: 0.6773662
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.63
Output dim: 0, lower bound: -0.6444212, upper bound: 0.6773662
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.63
Output dim: 0, lower bound: -0.6845282, upper bound: 0.6773662
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.63
Output dim: 0, lower bound: -0.6608970, upper bound: 0.6773662
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.63
Output dim: 0, lower bound: -0.6444212, upper bound: 0.6773662
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.63
Output dim: 0, lower bound: -0.6444212, upper bound: 0.6866100
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.63
Output dim: 0, lower bound: -0.6608970, upper bound: 0.6773662
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.63
Output dim: 0, lower bound: -0.6608970, upper bound: 0.6866100
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.63
Output dim: 0, lower bound: -0.6873946, upper bound: 0.6845138
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.63
Output dim: 0, lower bound: -0.6873948, upper bound: 0.6845138
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.63
Output dim: 0, lower bound: -0.6873946, upper bound: 0.6882732
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.63
Output dim: 0, lower bound: -0.6873948, upper bound: 0.6886574
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.63
Output dim: 0, lower bound: -0.6594252, upper bound: 0.6808003
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.63
Output dim: 0, lower bound: -0.6594252, upper bound: 0.6849745
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.63
Output dim: 0, lower bound: -0.6594252, upper bound: 0.6844697
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.63
Output dim: 0, lower bound: -0.6594252, upper bound: 0.6892572
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.63
Output dim: 0, lower bound: -0.6587127, upper bound: 0.6726318
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.63
Output dim: 0, lower bound: -0.6429276, upper bound: 0.6725800
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.63
Output dim: 0, lower bound: -0.6674482, upper bound: 0.6578935
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 0, lower bound: -0.6525701, upper bound: 0.6578416
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 0, lower bound: -0.6429276, upper bound: 0.6494225
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 0, lower bound: -0.6429276, upper bound: 0.6583646
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 0, lower bound: -0.6512621, upper bound: 0.6559546
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 0, lower bound: -0.6512621, upper bound: 0.6559546
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.63
Output dim: 0, lower bound: -0.6383973, upper bound: 0.6668210
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.63
Output dim: 0, lower bound: -0.6383973, upper bound: 0.6757573
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 0, lower bound: -0.6337648, upper bound: 0.6559546
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.63
Output dim: 0, lower bound: -0.6337648, upper bound: 0.6653052
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.63
Output dim: 0, lower bound: -0.6667848, upper bound: 0.6496201
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 0, lower bound: -0.6436402, upper bound: 0.6436381
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.63
Output dim: 0, lower bound: -0.6667848, upper bound: 0.6496201
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.63
Output dim: 0, lower bound: -0.6436402, upper bound: 0.6436381

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0265114, 0.8204293, 0.0213993, 0.9013143, -0.8748028, 0.7990301
1: -0.0697420, 1.0438113, -0.0770538, 1.1394897, -1.2092316, 1.1208651
2: -0.0287337, 0.9096682, -0.0367112, 0.9943478, -1.0230815, 0.9463794
3: -0.2322266, 0.9569638, -0.2404699, 1.0253115, -1.2575381, 1.1974337
4: -0.2029902, 0.8125998, -0.2202464, 0.8539461, -1.0569363, 1.0328462

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47

Time for candidate selection: 1.45 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6341388, upper bound: 0.6341388
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6341388, upper bound: 0.6341388
time: 0.29 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0265114, 0.8204293, -0.0028670, 1.0353181, -1.0088067, 0.8232963
1: -0.0697420, 1.0438113, -0.1088188, 1.3095722, -1.3793142, 1.1526301
2: -0.0287337, 0.9096682, -0.0732558, 1.1363764, -1.1651101, 0.9829240
3: -0.2322266, 0.9569638, -0.2704966, 1.1744311, -1.4066577, 1.2274604
4: -0.2029902, 0.8125998, -0.2937192, 0.9576517, -1.1606419, 1.1063190

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 5

Time for candidate selection: 1.40 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6341388, upper bound: 0.6450345
time: 0.33 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6341388, upper bound: 0.6450345
time: 0.30 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0022838, 0.9603593, 0.0213993, 0.9013143, -0.8990304, 0.9389601
1: -0.1017990, 1.2228236, -0.0770538, 1.1394897, -1.2412887, 1.2998774
2: -0.0658340, 1.0562885, -0.0367112, 0.9943478, -1.0601819, 1.0929997
3: -0.2629817, 1.1104673, -0.2404699, 1.0253115, -1.2882931, 1.3509372
4: -0.2787794, 0.9093835, -0.2202464, 0.8539461, -1.1327255, 1.1296300

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47

Time for candidate selection: 1.50 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6341388, upper bound: 0.6365467
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6341388, upper bound: 0.6365467
time: 0.35 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0022838, 0.9603593, -0.0028670, 1.0353181, -1.0330343, 0.9632263
1: -0.1017990, 1.2228236, -0.1088188, 1.3095722, -1.4113712, 1.3316424
2: -0.0658340, 1.0562885, -0.0732558, 1.1363764, -1.2022104, 1.1295443
3: -0.2629817, 1.1104673, -0.2704966, 1.1744311, -1.4374127, 1.3809639
4: -0.2787794, 0.9093835, -0.2937192, 0.9576517, -1.2364311, 1.2031027

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 5

Time for candidate selection: 1.46 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6474426, upper bound: 0.6468780
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6474426, upper bound: 0.6468780
time: 0.32 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0339961, 1.0422370, -0.0028670, 1.0353181, -1.0693142, 1.0451040
1: -0.1467624, 1.3160231, -0.1088188, 1.3095722, -1.4563346, 1.4248419
2: -0.1044300, 1.1530347, -0.0732558, 1.1363764, -1.2408063, 1.2262905
3: -0.3024733, 1.1932034, -0.2704966, 1.1744311, -1.4769044, 1.4637001
4: -0.3252316, 0.9918821, -0.2937192, 0.9576517, -1.2828833, 1.2856013

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6504679, upper bound: 0.6634342
time: 0.38 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6512377, upper bound: 0.6512695
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, 0.0248351, 0.8239982, -0.8128102, 0.5400264
1: -0.0944340, 0.7130072, -0.0720310, 1.0479953, -1.1424294, 0.7850382
2: -0.0312903, 0.7124612, -0.0307086, 0.9141164, -0.9454067, 0.7431698
3: -0.2500148, 0.7301772, -0.2340453, 0.9604603, -1.2104751, 0.9642225
4: -0.1787794, 0.7720095, -0.2052969, 0.8165774, -0.9953567, 0.9773064

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6406149, upper bound: 0.6935838
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5913267, upper bound: 0.6838013
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47

Time for candidate selection: 2.61 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6937541
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6663589
time: 0.34 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, 0.0248351, 0.8239982, -0.8545651, 0.6458231
1: -0.1440713, 0.8347254, -0.0720310, 1.0479953, -1.1920667, 0.9067564
2: -0.0742025, 0.8280591, -0.0307086, 0.9141164, -0.9883189, 0.8587677
3: -0.2901301, 0.8312570, -0.2340453, 0.9604603, -1.2505904, 1.0653024
4: -0.2451730, 0.8507983, -0.2052969, 0.8165774, -1.0617504, 1.0560951

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6406149, upper bound: 0.6935838
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5913267, upper bound: 0.6838013
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47

Time for candidate selection: 2.53 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6937541
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6663589
time: 0.34 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, -0.0167825, 0.9250188, -0.9138308, 0.5816441
1: -0.0944340, 0.7130072, -0.1252913, 1.1663461, -1.2607801, 0.8382986
2: -0.0312903, 0.7124612, -0.0773900, 1.0296290, -1.0609193, 0.7898512
3: -0.2500148, 0.7301772, -0.2777247, 1.0633786, -1.3133934, 1.0079019
4: -0.1787794, 0.7720095, -0.2678475, 0.8997647, -1.0785441, 1.0398570

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6406149, upper bound: 0.6661886
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5913267, upper bound: 0.6618038
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47

Time for candidate selection: 2.51 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6663589
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6663589
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, -0.0167825, 0.9250188, -0.9555857, 0.6874408
1: -0.1440713, 0.8347254, -0.1252913, 1.1663461, -1.3104174, 0.9600167
2: -0.0742025, 0.8280591, -0.0773900, 1.0296290, -1.1038315, 0.9054491
3: -0.2901301, 0.8312570, -0.2777247, 1.0633786, -1.3535087, 1.1089818
4: -0.2451730, 0.8507983, -0.2678475, 0.8997647, -1.1449378, 1.1186459

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6406149, upper bound: 0.6661886
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5913267, upper bound: 0.6618038
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47

Time for candidate selection: 2.51 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6663589
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6663589
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, 0.0248351, 0.8239982, -0.8128102, 0.5400264
1: -0.0944340, 0.7130072, -0.0720310, 1.0479953, -1.1424294, 0.7850382
2: -0.0312903, 0.7124612, -0.0307086, 0.9141164, -0.9454067, 0.7431698
3: -0.2500148, 0.7301772, -0.2340453, 0.9604603, -1.2104751, 0.9642225
4: -0.1787794, 0.7720095, -0.2052969, 0.8165774, -0.9953567, 0.9773064

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6406149, upper bound: 0.6935838
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5913267, upper bound: 0.6838013
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47

Time for candidate selection: 2.53 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6937541
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6663589
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, 0.0248351, 0.8239982, -0.8545651, 0.6458231
1: -0.1440713, 0.8347254, -0.0720310, 1.0479953, -1.1920667, 0.9067564
2: -0.0742025, 0.8280591, -0.0307086, 0.9141164, -0.9883189, 0.8587677
3: -0.2901301, 0.8312570, -0.2340453, 0.9604603, -1.2505904, 1.0653024
4: -0.2451730, 0.8507983, -0.2052969, 0.8165774, -1.0617504, 1.0560951

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6406149, upper bound: 0.7029979
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5913267, upper bound: 0.6927868
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47

Time for candidate selection: 2.69 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6420588, upper bound: 0.7029979
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6756027
time: 0.32 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, -0.0167825, 0.9250188, -0.9138308, 0.5816441
1: -0.0944340, 0.7130072, -0.1252913, 1.1663461, -1.2607801, 0.8382986
2: -0.0312903, 0.7124612, -0.0773900, 1.0296290, -1.0609193, 0.7898512
3: -0.2500148, 0.7301772, -0.2777247, 1.0633786, -1.3133934, 1.0079019
4: -0.1787794, 0.7720095, -0.2678475, 0.8997647, -1.0785441, 1.0398570

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6406149, upper bound: 0.6661886
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5913267, upper bound: 0.6618038
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47

Time for candidate selection: 2.55 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6663589
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6663589
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, -0.0167825, 0.9250188, -0.9555857, 0.6874408
1: -0.1440713, 0.8347254, -0.1252913, 1.1663461, -1.3104174, 0.9600167
2: -0.0742025, 0.8280591, -0.0773900, 1.0296290, -1.1038315, 0.9054491
3: -0.2901301, 0.8312570, -0.2777247, 1.0633786, -1.3535087, 1.1089818
4: -0.2451730, 0.8507983, -0.2678475, 0.8997647, -1.1449378, 1.1186459

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6406149, upper bound: 0.6756027
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5913267, upper bound: 0.6708264
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47

Time for candidate selection: 2.64 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6756027
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6540380, upper bound: 0.6756027
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, 0.0006924, 0.9639854, -0.9527974, 0.5641692
1: -0.0944340, 0.7130072, -0.1038828, 1.2270458, -1.3214798, 0.8168900
2: -0.0312903, 0.7124612, -0.0676236, 1.0606856, -1.0919759, 0.7800848
3: -0.2500148, 0.7301772, -0.2647138, 1.1143060, -1.3643208, 0.9948910
4: -0.1787794, 0.7720095, -0.2805611, 0.9134201, -1.0921994, 1.0525706

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47

Time for candidate selection: 1.89 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6444212, upper bound: 0.6988041
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6444212, upper bound: 0.6773662
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, 0.0006924, 0.9639854, -0.9945524, 0.6699659
1: -0.1440713, 0.8347254, -0.1038828, 1.2270458, -1.3711171, 0.9386082
2: -0.0742025, 0.8280591, -0.0676236, 1.0606856, -1.1348882, 0.8956828
3: -0.2901301, 0.8312570, -0.2647138, 1.1143060, -1.4044361, 1.0959709
4: -0.2451730, 0.8507983, -0.2805611, 0.9134201, -1.1585931, 1.1313593

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47

Time for candidate selection: 1.88 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6444212, upper bound: 0.6988041
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6444212, upper bound: 0.6773662
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, -0.0360885, 1.0472777, -1.0360897, 0.6009500
1: -0.0944340, 0.7130072, -0.1492593, 1.3218184, -1.4162524, 0.8622665
2: -0.0312903, 0.7124612, -0.1066136, 1.1590223, -1.1903126, 0.8190749
3: -0.2500148, 0.7301772, -0.3045232, 1.1983278, -1.4483426, 1.0347004
4: -0.1787794, 0.7720095, -0.3275616, 0.9970202, -1.1757996, 1.0995711

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6401203, upper bound: 0.6771959
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 5

Time for candidate selection: 2.33 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6444212, upper bound: 0.6773662
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6444212, upper bound: 0.6773662
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, -0.0360885, 1.0472777, -1.0778446, 0.7067467
1: -0.1440713, 0.8347254, -0.1492593, 1.3218184, -1.4658897, 0.9839847
2: -0.0742025, 0.8280591, -0.1066136, 1.1590223, -1.2332249, 0.9346728
3: -0.2901301, 0.8312570, -0.3045232, 1.1983278, -1.4884579, 1.1357803
4: -0.2451730, 0.8507983, -0.3275616, 0.9970202, -1.2421932, 1.1783600

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6401203, upper bound: 0.6771959
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 5

Time for candidate selection: 2.25 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6444212, upper bound: 0.6773662
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6444212, upper bound: 0.6773662
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, 0.0006924, 0.9639854, -0.9527974, 0.5641692
1: -0.0944340, 0.7130072, -0.1038828, 1.2270458, -1.3214798, 0.8168900
2: -0.0312903, 0.7124612, -0.0676236, 1.0606856, -1.0919759, 0.7800848
3: -0.2500148, 0.7301772, -0.2647138, 1.1143060, -1.3643208, 0.9948910
4: -0.1787794, 0.7720095, -0.2805611, 0.9134201, -1.0921994, 1.0525706

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47

Time for candidate selection: 1.95 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6444212, upper bound: 0.6988041
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6444212, upper bound: 0.6773662
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, 0.0006924, 0.9639854, -0.9945524, 0.6699659
1: -0.1440713, 0.8347254, -0.1038828, 1.2270458, -1.3711171, 0.9386082
2: -0.0742025, 0.8280591, -0.0676236, 1.0606856, -1.1348882, 0.8956828
3: -0.2901301, 0.8312570, -0.2647138, 1.1143060, -1.4044361, 1.0959709
4: -0.2451730, 0.8507983, -0.2805611, 0.9134201, -1.1585931, 1.1313593

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47

Time for candidate selection: 1.96 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6444212, upper bound: 0.7080479
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6444212, upper bound: 0.6866100
time: 0.34 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, -0.0360885, 1.0472777, -1.0360897, 0.6009500
1: -0.0944340, 0.7130072, -0.1492593, 1.3218184, -1.4162524, 0.8622665
2: -0.0312903, 0.7124612, -0.1066136, 1.1590223, -1.1903126, 0.8190749
3: -0.2500148, 0.7301772, -0.3045232, 1.1983278, -1.4483426, 1.0347004
4: -0.1787794, 0.7720095, -0.3275616, 0.9970202, -1.1757996, 1.0995711

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6401203, upper bound: 0.6771959
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 5

Time for candidate selection: 2.36 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6444212, upper bound: 0.6773662
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6444212, upper bound: 0.6773662
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, -0.0360885, 1.0472777, -1.0778446, 0.7067467
1: -0.1440713, 0.8347254, -0.1492593, 1.3218184, -1.4658897, 0.9839847
2: -0.0742025, 0.8280591, -0.1066136, 1.1590223, -1.2332249, 0.9346728
3: -0.2901301, 0.8312570, -0.3045232, 1.1983278, -1.4884579, 1.1357803
4: -0.2451730, 0.8507983, -0.3275616, 0.9970202, -1.2421932, 1.1783600

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6401203, upper bound: 0.6771959
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 5

Time for candidate selection: 2.26 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6444212, upper bound: 0.6866100
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6608968, upper bound: 0.6866100
time: 0.37 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0185366, 0.5137528, 0.0073309, 0.6935004, -0.6749638, 0.5064219
1: -0.0853710, 0.6410187, -0.0999169, 0.8545845, -0.9399555, 0.7409357
2: -0.0191109, 0.6585484, -0.0374527, 0.8373916, -0.8565025, 0.6960011
3: -0.2388380, 0.6765358, -0.2552719, 0.8256997, -1.0645378, 0.9318078
4: -0.1577795, 0.7368335, -0.1935482, 0.8381348, -0.9959143, 0.9303817

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 47

Time for candidate selection: 1.47 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6599698, upper bound: 0.6840726
time: 0.40 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6689120, upper bound: 0.6702318
time: 0.39 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0387027, 0.4375544, 0.0073309, 0.6935004, -0.6547977, 0.4302235
1: -0.0613070, 0.5365020, -0.0999169, 0.8545845, -0.9158914, 0.6364189
2: 0.0174038, 0.5755119, -0.0374527, 0.8373916, -0.8199878, 0.6129646
3: -0.2170205, 0.5777811, -0.2552719, 0.8256997, -1.0427203, 0.8330530
4: -0.1004417, 0.6784918, -0.1935482, 0.8381348, -0.9385765, 0.8720400

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 47

Time for candidate selection: 1.48 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6599698, upper bound: 0.6841774
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6689120, upper bound: 0.6703365
time: 0.34 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0185366, 0.5137528, 0.0294173, 0.5563495, -0.5378129, 0.4843355
1: -0.0853710, 0.6410187, -0.0742822, 0.6929038, -0.7782748, 0.7153009
2: -0.0191109, 0.6585484, 0.0004599, 0.7029566, -0.7220675, 0.6580884
3: -0.2388380, 0.6765358, -0.2323704, 0.6962168, -0.9350548, 0.9089062
4: -0.1577795, 0.7368335, -0.1339579, 0.7538315, -0.9116110, 0.8707913

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5

Time for candidate selection: 1.55 seconds

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6873844, upper bound: 0.6882732
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6873844, upper bound: 0.6882732
time: 0.39 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0387027, 0.4375544, 0.0294173, 0.5563495, -0.5176468, 0.4081371
1: -0.0613070, 0.5365020, -0.0742822, 0.6929038, -0.7542108, 0.6107842
2: 0.0174038, 0.5755119, 0.0004599, 0.7029566, -0.6855527, 0.5750520
3: -0.2170205, 0.5777811, -0.2323704, 0.6962168, -0.9132373, 0.8101515
4: -0.1004417, 0.6784918, -0.1339579, 0.7538315, -0.8542732, 0.8124496

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5

Time for candidate selection: 1.55 seconds

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6873844, upper bound: 0.6882643
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6873844, upper bound: 0.6882642
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0860746, 0.7981433, 0.0073309, 0.6935004, -0.7795750, 0.7908124
1: -0.2016160, 0.9745648, -0.0999169, 0.8545845, -1.0562005, 1.0744817
2: -0.1532123, 0.9791473, -0.0374527, 0.8373916, -0.9906039, 1.0165999
3: -0.3256128, 0.9652334, -0.2552719, 0.8256997, -1.1513126, 1.2205054
4: -0.3422897, 0.9691569, -0.1935482, 0.8381348, -1.1804245, 1.1627052

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 47

Time for candidate selection: 1.51 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6376543, upper bound: 0.6808003
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6453843, upper bound: 0.6669595
time: 0.37 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0603211, 0.6870018, 0.0073309, 0.6935004, -0.7538215, 0.6796709
1: -0.1749001, 0.8297409, -0.0999169, 0.8545845, -1.0294845, 0.9296579
2: -0.1159739, 0.8579424, -0.0374527, 0.8373916, -0.9533656, 0.8953951
3: -0.3003569, 0.8422859, -0.2552719, 0.8256997, -1.1260567, 1.0975578
4: -0.2768354, 0.8908350, -0.1935482, 0.8381348, -1.1149702, 1.0843832

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 47

Time for candidate selection: 1.47 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6376543, upper bound: 0.6808003
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6453843, upper bound: 0.6709013
time: 0.39 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0860746, 0.7981433, 0.0294173, 0.5563495, -0.6424241, 0.7687260
1: -0.2016160, 0.9745648, -0.0742822, 0.6929038, -0.8945199, 1.0488470
2: -0.1532123, 0.9791473, 0.0004599, 0.7029566, -0.8561689, 0.9786873
3: -0.3256128, 0.9652334, -0.2323704, 0.6962168, -1.0218296, 1.1976038
4: -0.3422897, 0.9691569, -0.1339579, 0.7538315, -1.0961212, 1.1031148

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5

Time for candidate selection: 1.46 seconds

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6594252, upper bound: 0.6844697
time: 0.39 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6594252, upper bound: 0.6844697
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0603211, 0.6870018, 0.0294173, 0.5563495, -0.6166706, 0.6575845
1: -0.1749001, 0.8297409, -0.0742822, 0.6929038, -0.8678039, 0.9040231
2: -0.1159739, 0.8579424, 0.0004599, 0.7029566, -0.8189305, 0.8574825
3: -0.3003569, 0.8422859, -0.2323704, 0.6962168, -0.9965737, 1.0746562
4: -0.2768354, 0.8908350, -0.1339579, 0.7538315, -1.0306669, 1.0247929

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5

Time for candidate selection: 1.48 seconds

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6594252, upper bound: 0.6808003
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6594252, upper bound: 0.6808003
time: 0.37 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0158405, 0.4982040, 0.0061345, 0.6507710, -0.6349305, 0.4920695
1: -0.0890739, 0.6250523, -0.0976210, 0.8106803, -0.8997542, 0.7226732
2: -0.0220215, 0.6451663, -0.0417109, 0.7880348, -0.8100563, 0.6868772
3: -0.2441075, 0.6654664, -0.2534082, 0.7965333, -1.0406408, 0.9188746
4: -0.1598573, 0.7338556, -0.2023642, 0.8128391, -0.9726964, 0.9362198

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6549726, upper bound: 0.6725288
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47

Time for candidate selection: 2.31 seconds

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6429276, upper bound: 0.6725800
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6429276, upper bound: 0.6725800
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0892661, 0.7843834, 0.0061345, 0.6507710, -0.7400371, 0.7782488
1: -0.2051971, 0.9613039, -0.0976210, 0.8106803, -1.0158774, 1.0589249
2: -0.1573753, 0.9656872, -0.0417109, 0.7880348, -0.9454101, 1.0073980
3: -0.3306475, 0.9540346, -0.2534082, 0.7965333, -1.1271808, 1.2074428
4: -0.3440671, 0.9667487, -0.2023642, 0.8128391, -1.1569061, 1.1691129

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6428539, upper bound: 0.6724770
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47

Time for candidate selection: 2.38 seconds

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6429276, upper bound: 0.6725800
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6429276, upper bound: 0.6725800
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0158405, 0.4982040, -0.0329778, 0.7304660, -0.7146255, 0.5311818
1: -0.0890739, 0.6250523, -0.1447692, 0.9029138, -0.9919877, 0.7698215
2: -0.0220215, 0.6451663, -0.0813575, 0.8816222, -0.9036437, 0.7265238
3: -0.2441075, 0.6654664, -0.2928538, 0.8801655, -1.1242729, 0.9583203
4: -0.1598573, 0.7338556, -0.2609611, 0.8848863, -1.0447435, 0.9948167

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6658006, upper bound: 0.6578935
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47

Time for candidate selection: 2.34 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6587127, upper bound: 0.6578935
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6674482, upper bound: 0.6578935
time: 0.35 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0061345, 0.6507710, 0.0158405, 0.4982040, -0.4920695, 0.6349305
1: -0.0976210, 0.8106803, -0.0890739, 0.6250523, -0.7226732, 0.8997542
2: -0.0417109, 0.7880348, -0.0220215, 0.6451663, -0.6868772, 0.8100563
3: -0.2534082, 0.7965333, -0.2441075, 0.6654664, -0.9188746, 1.0406408
4: -0.2023642, 0.8128391, -0.1598573, 0.7338556, -0.9362198, 0.9726964

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5

Time for candidate selection: 2.03 seconds

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6555374, upper bound: 0.6668210
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6512620, upper bound: 0.6559544
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0329778, 0.7304660, 0.0158405, 0.4982040, -0.5311818, 0.7146255
1: -0.1447692, 0.9029138, -0.0890739, 0.6250523, -0.7698215, 0.9919877
2: -0.0813575, 0.8816222, -0.0220215, 0.6451663, -0.7265238, 0.9036437
3: -0.2928538, 0.8801655, -0.2441075, 0.6654664, -0.9583203, 1.1242729
4: -0.2609611, 0.8848863, -0.1598573, 0.7338556, -0.9948167, 1.0447435

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6186174, upper bound: 0.6734475
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5

Time for candidate selection: 2.39 seconds

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6383973, upper bound: 0.6757573
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6337648, upper bound: 0.6653052
time: 0.35 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0329778, 0.7304660, -0.0892951, 0.7884323, -0.8214101, 0.8197612
1: -0.1447692, 0.9029138, -0.2055373, 0.9655911, -1.1103604, 1.1084511
2: -0.0813575, 0.8816222, -0.1576910, 0.9697834, -1.0511409, 1.0393132
3: -0.2928538, 0.8801655, -0.3306990, 0.9586977, -1.2515515, 1.2108644
4: -0.2609611, 0.8848863, -0.3441186, 0.9710234, -1.2319844, 1.2290049

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6154483, upper bound: 0.6613079
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 5

Time for candidate selection: 2.36 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6261426, upper bound: 0.6653052
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6261426, upper bound: 0.6653052
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0168498, 0.4963174, 0.0061345, 0.6507710, -0.6339213, 0.4901829
1: -0.0846639, 0.6186082, -0.0976210, 0.8106803, -0.8953441, 0.7162292
2: -0.0223837, 0.6408591, -0.0417109, 0.7880348, -0.8104185, 0.6825700
3: -0.2404602, 0.6593775, -0.2534082, 0.7965333, -1.0369935, 0.9127856
4: -0.1628952, 0.7334037, -0.2023642, 0.8128391, -0.9757343, 0.9357679

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47

Time for candidate selection: 2.14 seconds

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6261426, upper bound: 0.6583764
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6261426, upper bound: 0.6583764
time: 0.37 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0168498, 0.4963174, -0.0329778, 0.7304660, -0.7136163, 0.5292952
1: -0.0846639, 0.6186082, -0.1447692, 0.9029138, -0.9875777, 0.7633774
2: -0.0223837, 0.6408591, -0.0813575, 0.8816222, -0.9040059, 0.7222166
3: -0.2404602, 0.6593775, -0.2928538, 0.8801655, -1.1206256, 0.9522313
4: -0.1628952, 0.7334037, -0.2609611, 0.8848863, -1.0477815, 0.9943648

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47

Time for candidate selection: 2.14 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6461472, upper bound: 0.6496201
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6461472, upper bound: 0.6496201
time: 0.37 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 4.81 seconds
IS_A1_B1_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6341388, upper bound: 0.6341388
IS_A1_B1_A1_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6341388, upper bound: 0.6341388
IS_A1_B1_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6341388, upper bound: 0.6450345
IS_A1_B1_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6341388, upper bound: 0.6450345
IS_A1_B1_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6341388, upper bound: 0.6365467
IS_A1_B1_A1_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6341388, upper bound: 0.6365467
IS_A1_B1_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6474426, upper bound: 0.6468780
IS_A1_B1_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6474426, upper bound: 0.6468780
IS_A1_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6504679, upper bound: 0.6634342
IS_A1_B1_A1_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6512377, upper bound: 0.6512695
IS_A2_B1_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6937541
IS_A2_B1_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6663589
IS_A2_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6937541
IS_A2_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6663589
IS_A2_B1_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6663589
IS_A2_B1_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6663589
IS_A2_B1_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6663589
IS_A2_B1_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6663589
IS_A2_B1_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6937541
IS_A2_B1_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6663589
IS_A2_B1_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6420588, upper bound: 0.7029979
IS_A2_B1_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6756027
IS_A2_B1_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6663589
IS_A2_B1_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6663589
IS_A2_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6756027
IS_A2_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6540380, upper bound: 0.6756027
IS_A2_B1_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6444212, upper bound: 0.6988041
IS_A2_B1_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6444212, upper bound: 0.6773662
IS_A2_B1_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6444212, upper bound: 0.6988041
IS_A2_B1_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6444212, upper bound: 0.6773662
IS_A2_B1_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6444212, upper bound: 0.6773662
IS_A2_B1_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6444212, upper bound: 0.6773662
IS_A2_B1_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6444212, upper bound: 0.6773662
IS_A2_B1_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6444212, upper bound: 0.6773662
IS_A2_B1_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6444212, upper bound: 0.6988041
IS_A2_B1_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6444212, upper bound: 0.6773662
IS_A2_B1_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6444212, upper bound: 0.7080479
IS_A2_B1_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6444212, upper bound: 0.6866100
IS_A2_B1_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6444212, upper bound: 0.6773662
IS_A2_B1_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6444212, upper bound: 0.6773662
IS_A2_B1_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6444212, upper bound: 0.6866100
IS_A2_B1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6608968, upper bound: 0.6866100
IS_A2_B2_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6599698, upper bound: 0.6840726
IS_A2_B2_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6689120, upper bound: 0.6702318
IS_A2_B2_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6599698, upper bound: 0.6841774
IS_A2_B2_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6689120, upper bound: 0.6703365
IS_A2_B2_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6873844, upper bound: 0.6882732
IS_A2_B2_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6873844, upper bound: 0.6882732
IS_A2_B2_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6873844, upper bound: 0.6882643
IS_A2_B2_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6873844, upper bound: 0.6882642
IS_A2_B2_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6376543, upper bound: 0.6808003
IS_A2_B2_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6453843, upper bound: 0.6669595
IS_A2_B2_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6376543, upper bound: 0.6808003
IS_A2_B2_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6453843, upper bound: 0.6709013
IS_A2_B2_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6594252, upper bound: 0.6844697
IS_A2_B2_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6594252, upper bound: 0.6844697
IS_A2_B2_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6594252, upper bound: 0.6808003
IS_A2_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6594252, upper bound: 0.6808003
IS_A2_B2_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6429276, upper bound: 0.6725800
IS_A2_B2_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6429276, upper bound: 0.6725800
IS_A2_B2_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6429276, upper bound: 0.6725800
IS_A2_B2_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6429276, upper bound: 0.6725800
IS_A2_B2_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6587127, upper bound: 0.6578935
IS_A2_B2_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6674482, upper bound: 0.6578935
IS_A2_B2_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6555374, upper bound: 0.6668210
IS_A2_B2_A2_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6512620, upper bound: 0.6559544
IS_A2_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6383973, upper bound: 0.6757573
IS_A2_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6337648, upper bound: 0.6653052
IS_A2_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6261426, upper bound: 0.6653052
IS_A2_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6261426, upper bound: 0.6653052
IS_A2_B2_A2_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6261426, upper bound: 0.6583764
IS_A2_B2_A2_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6261426, upper bound: 0.6583764
IS_A2_B2_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6461472, upper bound: 0.6496201
IS_A2_B2_A2_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 4.81
Output dim: 0, lower bound: -0.6461472, upper bound: 0.6496201

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0339961, 1.0422370, -0.0014365, 1.0292757, -1.0632718, 1.0436735
1: -0.1467624, 1.3160231, -0.1070645, 1.3027349, -1.4494972, 1.4230876
2: -0.1044300, 1.1530347, -0.0715175, 1.1294107, -1.2338407, 1.2245522
3: -0.3024733, 1.1932034, -0.2690842, 1.1685212, -1.4709945, 1.4622877
4: -0.3252316, 0.9918821, -0.2915230, 0.9521359, -1.2773675, 1.2834051

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6504679, upper bound: 0.6459153
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6504679, upper bound: 0.6512694
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, 0.0248351, 0.8239982, -0.8128102, 0.5400264
1: -0.0944340, 0.7130072, -0.0720310, 1.0479953, -1.1424294, 0.7850382
2: -0.0312903, 0.7124612, -0.0307086, 0.9141164, -0.9454067, 0.7431698
3: -0.2500148, 0.7301772, -0.2340453, 0.9604603, -1.2104751, 0.9642225
4: -0.1787794, 0.7720095, -0.2052969, 0.8165774, -0.9953567, 0.9773064

Time for backsubstitution: 1.84 seconds
Binary search (step 0): status=Status.UNKNOWN, low=0.0017312, high=0.0917747, mid=0.0917747, abs_max=0.7819017171859741
rel_dist={0: [-0.7126414684186158, 0.7126414684186155]}

## Binary search (step 1) starts
Candidate diff: 0.0467529


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6999231, upper bound: 0.6862236
time: 0.30 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6996897, upper bound: 0.7000153
time: 0.32 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.79 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.79
Output dim: 0, lower bound: -0.6999231, upper bound: 0.6862236
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.79
Output dim: 0, lower bound: -0.6996897, upper bound: 0.7000153

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0017378, 0.9577341, -0.0274801, 0.7245864, -0.7263242, 0.9852142
1: -0.1117451, 1.2035816, -0.1478488, 0.8963833, -1.0081284, 1.3514304
2: -0.0625789, 1.0695953, -0.0690396, 0.8955200, -0.9580989, 1.1386349
3: -0.2743888, 1.0868173, -0.3069696, 0.8793733, -1.1537621, 1.3937869
4: -0.2543875, 0.9355542, -0.2427759, 0.9202661, -1.1746535, 1.1783302

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6862236, upper bound: 0.6862236
time: 0.30 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6862236, upper bound: 0.6862236
time: 0.32 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0124662, 0.7046853, -0.0323386, 0.7495631, -0.7620293, 0.7370239
1: -0.1260490, 0.8739170, -0.1552227, 0.9255341, -1.0515832, 1.0291396
2: -0.0592167, 0.8673251, -0.0741289, 0.9267865, -0.9860032, 0.9414539
3: -0.2830637, 0.8561093, -0.3143692, 0.9027434, -1.1858070, 1.1704785
4: -0.2274244, 0.8807485, -0.2515757, 0.9410419, -1.1684663, 1.1323242

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6862236, upper bound: 0.6996897
time: 0.31 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6862236, upper bound: 0.7000153
time: 0.31 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.39 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.39
Output dim: 0, lower bound: -0.6862236, upper bound: 0.6862236
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.39
Output dim: 0, lower bound: -0.6862236, upper bound: 0.6862236
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.39
Output dim: 0, lower bound: -0.6862236, upper bound: 0.6996897
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.39
Output dim: 0, lower bound: -0.6862236, upper bound: 0.7000153

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0017378, 0.9577341, -0.0017378, 0.9577341, -0.9594719, 0.9594719
1: -0.1117451, 1.2035816, -0.1117451, 1.2035816, -1.3153267, 1.3153267
2: -0.0625789, 1.0695953, -0.0625789, 1.0695953, -1.1321743, 1.1321743
3: -0.2743888, 1.0868173, -0.2743888, 1.0868173, -1.3612061, 1.3612061
4: -0.2543875, 0.9355542, -0.2543875, 0.9355542, -1.1899416, 1.1899416

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6756641, upper bound: 0.6516575
time: 0.30 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6609718, upper bound: 0.6438791
time: 0.29 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0017378, 0.9577341, -0.0124662, 0.7046853, -0.7064232, 0.9702003
1: -0.1117451, 1.2035816, -0.1260490, 0.8739170, -0.9856621, 1.3296306
2: -0.0625789, 1.0695953, -0.0592167, 0.8673251, -0.9299040, 1.1288121
3: -0.2743888, 1.0868173, -0.2830637, 0.8561093, -1.1304981, 1.3698809
4: -0.2543875, 0.9355542, -0.2274244, 0.8807485, -1.1351360, 1.1629786

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6756641, upper bound: 0.6516575
time: 0.29 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6609718, upper bound: 0.6438791
time: 0.36 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0124662, 0.7046853, -0.0017378, 0.9577341, -0.9702003, 0.7064232
1: -0.1260490, 0.8739170, -0.1117451, 1.2035816, -1.3296306, 0.9856621
2: -0.0592167, 0.8673251, -0.0625789, 1.0695953, -1.1288121, 0.9299040
3: -0.2830637, 0.8561093, -0.2743888, 1.0868173, -1.3698809, 1.1304981
4: -0.2274244, 0.8807485, -0.2543875, 0.9355542, -1.1629786, 1.1351360

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6851157, upper bound: 0.6981797
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6855405, upper bound: 0.6977845
time: 0.32 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0124662, 0.7046853, -0.0124662, 0.7046853, -0.7171515, 0.7171515
1: -0.1260490, 0.8739170, -0.1260490, 0.8739170, -0.9999660, 0.9999660
2: -0.0592167, 0.8673251, -0.0592167, 0.8673251, -0.9265418, 0.9265418
3: -0.2830637, 0.8561093, -0.2830637, 0.8561093, -1.1391729, 1.1391729
4: -0.2274244, 0.8807485, -0.2274244, 0.8807485, -1.1081729, 1.1081729

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6851157, upper bound: 0.6983001
time: 0.32 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6855405, upper bound: 0.6977845
time: 0.33 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.11 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.11
Output dim: 0, lower bound: -0.6756641, upper bound: 0.6516575
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 3.11
Output dim: 0, lower bound: -0.6609718, upper bound: 0.6438791
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.11
Output dim: 0, lower bound: -0.6756641, upper bound: 0.6516575
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 3.11
Output dim: 0, lower bound: -0.6609718, upper bound: 0.6438791
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.11
Output dim: 0, lower bound: -0.6851157, upper bound: 0.6981797
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.11
Output dim: 0, lower bound: -0.6855405, upper bound: 0.6977845
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.11
Output dim: 0, lower bound: -0.6851157, upper bound: 0.6983001
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.11
Output dim: 0, lower bound: -0.6855405, upper bound: 0.6977845

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0000615, 0.9523284, -0.0017378, 0.9577341, -0.9576726, 0.9540663
1: -0.1095166, 1.1974459, -0.1117451, 1.2035816, -1.3130982, 1.3091910
2: -0.0606787, 1.0631847, -0.0625789, 1.0695953, -1.1302741, 1.1257637
3: -0.2725427, 1.0815248, -0.2743888, 1.0868173, -1.3593600, 1.3559136
4: -0.2520238, 0.9301934, -0.2543875, 0.9355542, -1.1875780, 1.1845809

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6639844, upper bound: 0.6639844
time: 0.30 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6639844, upper bound: 0.6639844
time: 0.30 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0000615, 0.9523284, -0.0124662, 0.7046853, -0.7046238, 0.9647946
1: -0.1095166, 1.1974459, -0.1260490, 0.8739170, -0.9834336, 1.3234949
2: -0.0606787, 1.0631847, -0.0592167, 0.8673251, -0.9280038, 1.1224015
3: -0.2725427, 1.0815248, -0.2830637, 0.8561093, -1.1286520, 1.3645885
4: -0.2520238, 0.9301934, -0.2274244, 0.8807485, -1.1327723, 1.1576178

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6609718, upper bound: 0.6438791
time: 0.29 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6609718, upper bound: 0.6438791
time: 0.28 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0054424, 0.6499398, -0.0017378, 0.9577341, -0.9522917, 0.6516776
1: -0.1022949, 0.8109143, -0.1117451, 1.2035816, -1.3058765, 0.9226594
2: -0.0393627, 0.7986160, -0.0625789, 1.0695953, -1.1089580, 0.8611949
3: -0.2590857, 0.7998861, -0.2743888, 1.0868173, -1.3459029, 1.0742749
4: -0.1939640, 0.8231770, -0.2543875, 0.9355542, -1.1295183, 1.0775645

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6618203, upper bound: 0.6803843
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6779333, upper bound: 0.6932491
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6779333, upper bound: 0.6982991
time: 0.30 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0004525, 0.7221838, 0.0004685, 0.9435406, -0.9430881, 0.7217153
1: -0.1048932, 0.8939738, -0.1086941, 1.1876776, -1.2925708, 1.0026679
2: -0.0500441, 0.8598962, -0.0600665, 1.0513515, -1.1013956, 0.9199626
3: -0.2612641, 0.8577256, -0.2708709, 1.0741191, -1.3353832, 1.1285965
4: -0.2176540, 0.8560207, -0.2502931, 0.9241737, -1.1418277, 1.1063139

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6515660, upper bound: 0.6751746
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6438791, upper bound: 0.6607286
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0054424, 0.6499398, -0.0124662, 0.7046853, -0.6992429, 0.6624060
1: -0.1022949, 0.8109143, -0.1260490, 0.8739170, -0.9762119, 0.9369633
2: -0.0393627, 0.7986160, -0.0592167, 0.8673251, -0.9066877, 0.8578327
3: -0.2590857, 0.7998861, -0.2830637, 0.8561093, -1.1151949, 1.0829498
4: -0.1939640, 0.8231770, -0.2274244, 0.8807485, -1.0747125, 1.0506014

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6804400, upper bound: 0.6945447
time: 0.33 seconds

## Relational analysis of IS_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6779333, upper bound: 0.6973597
time: 0.36 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6779333, upper bound: 0.6977845
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0004525, 0.7221838, -0.0070884, 0.6870102, -0.6865577, 0.7292722
1: -0.1048932, 0.8939738, -0.1182010, 0.8538061, -0.9586993, 1.0121748
2: -0.0500441, 0.8598962, -0.0546446, 0.8421040, -0.8921480, 0.9145408
3: -0.2612641, 0.8577256, -0.2754803, 0.8384964, -1.0997605, 1.1332059
4: -0.2176540, 0.8560207, -0.2195926, 0.8588813, -1.0765352, 1.0756133

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6515660, upper bound: 0.6746649
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6408665, upper bound: 0.6408665
time: 0.34 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.27 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 0, lower bound: -0.6639844, upper bound: 0.6639844
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 0, lower bound: -0.6639844, upper bound: 0.6639844
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 3.27
Output dim: 0, lower bound: -0.6609718, upper bound: 0.6438791
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 3.27
Output dim: 0, lower bound: -0.6609718, upper bound: 0.6438791
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 0, lower bound: -0.6779333, upper bound: 0.6932491
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 0, lower bound: -0.6779333, upper bound: 0.6982991
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 0, lower bound: -0.6515660, upper bound: 0.6751746
IS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 3.27
Output dim: 0, lower bound: -0.6438791, upper bound: 0.6607286
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 0, lower bound: -0.6779333, upper bound: 0.6973597
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 0, lower bound: -0.6779333, upper bound: 0.6977845
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 0, lower bound: -0.6515660, upper bound: 0.6746649
IS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 3.27
Output dim: 0, lower bound: -0.6408665, upper bound: 0.6408665

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0000615, 0.9523284, 0.0000615, 0.9523284, -0.9522669, 0.9522669
1: -0.1095166, 1.1974459, -0.1095166, 1.1974459, -1.3069625, 1.3069625
2: -0.0606787, 1.0631847, -0.0606787, 1.0631847, -1.1238635, 1.1238635
3: -0.2725427, 1.0815248, -0.2725427, 1.0815248, -1.3540676, 1.3540676
4: -0.2520238, 0.9301934, -0.2520238, 0.9301934, -1.1822172, 1.1822172

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6731502, upper bound: 0.6560802
time: 0.31 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6781872, upper bound: 0.6714281
time: 0.35 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0000615, 0.9523284, -0.0512598, 1.8787861, -1.8787246, 1.0035882
1: -0.1095166, 1.1974459, -0.1754494, 2.2964354, -2.4059520, 1.3728952
2: -0.0606787, 1.0631847, -0.1450286, 2.0645037, -2.1251824, 1.2082133
3: -0.2725427, 1.0815248, -0.3347113, 1.8700678, -2.1426105, 1.4162362
4: -0.2520238, 0.9301934, -0.4980223, 1.5064290, -1.7584528, 1.4282157

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6731502, upper bound: 0.6560802
time: 0.31 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6781872, upper bound: 0.6714281
time: 0.45 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0054424, 0.6499398, 0.0197756, 0.9062033, -0.9007609, 0.6301641
1: -0.1022949, 0.8109143, -0.0792882, 1.1450822, -1.2473772, 0.8902025
2: -0.0393627, 0.7986160, -0.0386198, 1.0002036, -1.0395663, 0.8372357
3: -0.2590857, 0.7998861, -0.2423158, 1.0302336, -1.2893193, 1.0422019
4: -0.1939640, 0.8231770, -0.2225227, 0.8589398, -1.0529038, 1.0456997

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5

Time for candidate selection: 1.67 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6776237, upper bound: 0.6577405
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6540380, upper bound: 0.6669338
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0054424, 0.6499398, -0.0046289, 1.0400095, -1.0345671, 0.6545687
1: -0.1022949, 0.8109143, -0.1109180, 1.3149409, -1.4172359, 0.9218323
2: -0.0393627, 0.7986160, -0.0750914, 1.1419265, -1.1812892, 0.8737074
3: -0.2590857, 0.7998861, -0.2722509, 1.1791039, -1.4381895, 1.0721370
4: -0.1939640, 0.8231770, -0.2956257, 0.9621377, -1.1561017, 1.1188027

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5

Time for candidate selection: 1.69 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6776237, upper bound: 0.6682850
time: 0.32 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6540380, upper bound: 0.6763568
time: 0.33 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0004525, 0.7221838, 0.0022480, 0.9381881, -0.9377356, 0.7199358
1: -0.1048932, 0.8939738, -0.1064734, 1.1815870, -1.2864802, 1.0004473
2: -0.0500441, 0.8598962, -0.0581751, 1.0449984, -1.0950425, 0.9180713
3: -0.2612641, 0.8577256, -0.2690332, 1.0688602, -1.3301243, 1.1267588
4: -0.2176540, 0.8560207, -0.2479466, 0.9189078, -1.1365618, 1.1039674

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6438791, upper bound: 0.6607286
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6438791, upper bound: 0.6607286
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0054424, 0.6499398, 0.0054424, 0.6499398, -0.6444974, 0.6444974
1: -0.1022949, 0.8109143, -0.1022949, 0.8109143, -0.9132092, 0.9132092
2: -0.0393627, 0.7986160, -0.0393627, 0.7986160, -0.8379787, 0.8379787
3: -0.2590857, 0.7998861, -0.2590857, 0.7998861, -1.0589718, 1.0589718
4: -0.1939640, 0.8231770, -0.1939640, 0.8231770, -1.0171410, 1.0171410

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5

Time for candidate selection: 1.77 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6842837, upper bound: 0.6729489
time: 0.34 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6636083, upper bound: 0.6699629
time: 0.38 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0054424, 0.6499398, 0.0004525, 0.7221838, -0.7167414, 0.6494873
1: -0.1022949, 0.8109143, -0.1048932, 0.8939738, -0.9962687, 0.9158075
2: -0.0393627, 0.7986160, -0.0500441, 0.8598962, -0.8992589, 0.8486601
3: -0.2590857, 0.7998861, -0.2612641, 0.8577256, -1.1168113, 1.0611502
4: -0.1939640, 0.8231770, -0.2176540, 0.8560207, -1.0499847, 1.0408310

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5

Time for candidate selection: 1.76 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6842837, upper bound: 0.6729489
time: 0.34 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6636083, upper bound: 0.6699629
time: 0.35 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0004525, 0.7221838, -0.0053399, 0.6811517, -0.6806992, 0.7275237
1: -0.1048932, 0.8939738, -0.1161585, 0.8470713, -0.9519645, 1.0101323
2: -0.0500441, 0.8598962, -0.0527055, 0.8357173, -0.8857614, 0.9126017
3: -0.2612641, 0.8577256, -0.2738860, 0.8327994, -1.0940635, 1.1316116
4: -0.2176540, 0.8560207, -0.2166147, 0.8540846, -1.0717386, 1.0726354

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6408665, upper bound: 0.6408665
time: 0.31 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6408665, upper bound: 0.6408665
time: 0.33 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.17 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.17
Output dim: 0, lower bound: -0.6731502, upper bound: 0.6560802
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.17
Output dim: 0, lower bound: -0.6781872, upper bound: 0.6714281
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.17
Output dim: 0, lower bound: -0.6731502, upper bound: 0.6560802
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.17
Output dim: 0, lower bound: -0.6781872, upper bound: 0.6714281
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.17
Output dim: 0, lower bound: -0.6776237, upper bound: 0.6577405
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.17
Output dim: 0, lower bound: -0.6540380, upper bound: 0.6669338
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.17
Output dim: 0, lower bound: -0.6776237, upper bound: 0.6682850
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.17
Output dim: 0, lower bound: -0.6540380, upper bound: 0.6763568
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.17
Output dim: 0, lower bound: -0.6438791, upper bound: 0.6607286
IS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.17
Output dim: 0, lower bound: -0.6438791, upper bound: 0.6607286
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.17
Output dim: 0, lower bound: -0.6842837, upper bound: 0.6729489
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.17
Output dim: 0, lower bound: -0.6636083, upper bound: 0.6699629
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.17
Output dim: 0, lower bound: -0.6842837, upper bound: 0.6729489
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.17
Output dim: 0, lower bound: -0.6636083, upper bound: 0.6699629
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.17
Output dim: 0, lower bound: -0.6408665, upper bound: 0.6408665
IS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.17
Output dim: 0, lower bound: -0.6408665, upper bound: 0.6408665

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0213993, 0.9013143, 0.0000615, 0.9523284, -0.9309292, 0.9012527
1: -0.0770538, 1.1394897, -0.1095166, 1.1974459, -1.2744997, 1.2490063
2: -0.0367112, 0.9943478, -0.0606787, 1.0631847, -1.0998960, 1.0550265
3: -0.2404699, 1.0253115, -0.2725427, 1.0815248, -1.3219948, 1.2978542
4: -0.2202464, 0.8539461, -0.2520238, 0.9301934, -1.1504399, 1.1059699

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6736547, upper bound: 0.6736547
time: 0.30 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6736547, upper bound: 0.6777835
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0028670, 1.0353181, 0.0022480, 0.9381881, -0.9410551, 1.0330701
1: -0.1088188, 1.3095722, -0.1064734, 1.1815870, -1.2904058, 1.4160457
2: -0.0732558, 1.1363764, -0.0581751, 1.0449984, -1.1182542, 1.1945515
3: -0.2704966, 1.1744311, -0.2690332, 1.0688602, -1.3393568, 1.4434643
4: -0.2937192, 0.9576517, -0.2479466, 0.9189078, -1.2126269, 1.2055984

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6777835, upper bound: 0.6808372
time: 0.31 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6777835, upper bound: 0.6858741
time: 0.31 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0213993, 0.9013143, -0.0512598, 1.8787861, -1.8573868, 0.9525740
1: -0.0770538, 1.1394897, -0.1754494, 2.2964354, -2.3734891, 1.3149390
2: -0.0367112, 0.9943478, -0.1450286, 2.0645037, -2.1012149, 1.1393764
3: -0.2404699, 1.0253115, -0.3347113, 1.8700678, -2.1105378, 1.3600228
4: -0.2202464, 0.8539461, -0.4980223, 1.5064290, -1.7266754, 1.3519684

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47

Time for candidate selection: 1.69 seconds

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6219948, upper bound: 0.6546898
time: 0.31 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6449059, upper bound: 0.6491899
time: 0.33 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0028670, 1.0353181, -0.0460036, 1.8619289, -1.8647959, 1.0813217
1: -0.1088188, 1.3095722, -0.1686013, 2.2772894, -2.3861082, 1.4781735
2: -0.0732558, 1.1363764, -0.1385605, 2.0437284, -2.1169841, 1.2749369
3: -0.2704966, 1.1744311, -0.3297453, 1.8520892, -2.1225858, 1.5041764
4: -0.2937192, 0.9576517, -0.4871181, 1.4866383, -1.7803575, 1.4447699

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6400006, upper bound: 0.6714281
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6441794, upper bound: 0.6534000
time: 0.32 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, 0.0197756, 0.9062033, -0.8950152, 0.5450859
1: -0.0944340, 0.7130072, -0.0792882, 1.1450822, -1.2395163, 0.7922955
2: -0.0312903, 0.7124612, -0.0386198, 1.0002036, -1.0314939, 0.7510810
3: -0.2500148, 0.7301772, -0.2423158, 1.0302336, -1.2802484, 0.9724930
4: -0.1787794, 0.7720095, -0.2225227, 0.8589398, -1.0377191, 0.9945322

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6615840, upper bound: 0.6577405
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47

Time for candidate selection: 2.18 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6420073, upper bound: 0.6577405
time: 0.32 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6420073, upper bound: 0.6577405
time: 0.34 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, 0.0197756, 0.9062033, -0.9367702, 0.6508826
1: -0.1440713, 0.8347254, -0.0792882, 1.1450822, -1.2891536, 0.9140136
2: -0.0742025, 0.8280591, -0.0386198, 1.0002036, -1.0744061, 0.8666789
3: -0.2901301, 0.8312570, -0.2423158, 1.0302336, -1.3203638, 1.0735729
4: -0.2451730, 0.8507983, -0.2225227, 0.8589398, -1.1041127, 1.0733211

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6398916, upper bound: 0.6668439
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47

Time for candidate selection: 2.06 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6420073, upper bound: 0.6669338
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6420073, upper bound: 0.6669338
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, -0.0046289, 1.0400095, -1.0288215, 0.5694904
1: -0.0944340, 0.7130072, -0.1109180, 1.3149409, -1.4093750, 0.8239253
2: -0.0312903, 0.7124612, -0.0750914, 1.1419265, -1.1732168, 0.7875526
3: -0.2500148, 0.7301772, -0.2722509, 1.1791039, -1.4291186, 1.0024281
4: -0.1787794, 0.7720095, -0.2956257, 0.9621377, -1.1409171, 1.0676352

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6615082, upper bound: 0.6595865
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 5

Time for candidate selection: 2.12 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6682847
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6682847
time: 0.34 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, -0.0046289, 1.0400095, -1.0705764, 0.6752871
1: -0.1440713, 0.8347254, -0.1109180, 1.3149409, -1.4590123, 0.9456434
2: -0.0742025, 0.8280591, -0.0750914, 1.1419265, -1.2161291, 0.9031505
3: -0.2901301, 0.8312570, -0.2722509, 1.1791039, -1.4692340, 1.1035080
4: -0.2451730, 0.8507983, -0.2956257, 0.9621377, -1.2073107, 1.1464241

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6394649, upper bound: 0.6668335
time: 0.33 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 5

Time for candidate selection: 2.05 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6763565
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6763565
time: 0.37 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0158405, 0.4982040, 0.0054424, 0.6499398, -0.6340992, 0.4927616
1: -0.0890739, 0.6250523, -0.1022949, 0.8109143, -0.8999882, 0.7273472
2: -0.0220215, 0.6451663, -0.0393627, 0.7986160, -0.8206375, 0.6845290
3: -0.2441075, 0.6654664, -0.2590857, 0.7998861, -1.0439936, 0.9245521
4: -0.1598573, 0.7338556, -0.1939640, 0.8231770, -0.9830343, 0.9278196

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6874736, upper bound: 0.6792575
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6874736, upper bound: 0.6813382
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0892951, 0.7884323, 0.0054424, 0.6499398, -0.7392349, 0.7829899
1: -0.2055373, 0.9655911, -0.1022949, 0.8109143, -1.0164516, 1.0678861
2: -0.1576910, 0.9697834, -0.0393627, 0.7986160, -0.9563070, 1.0091461
3: -0.3306990, 0.9586977, -0.2590857, 0.7998861, -1.1305851, 1.2177833
4: -0.3441186, 0.9710234, -0.1939640, 0.8231770, -1.1672956, 1.1649873

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6697728, upper bound: 0.6793275
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6722345, upper bound: 0.6820627
time: 0.37 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0158405, 0.4982040, 0.0004525, 0.7221838, -0.7063433, 0.4977515
1: -0.0890739, 0.6250523, -0.1048932, 0.8939738, -0.9830477, 0.7299455
2: -0.0220215, 0.6451663, -0.0500441, 0.8598962, -0.8819177, 0.6952104
3: -0.2441075, 0.6654664, -0.2612641, 0.8577256, -1.1018331, 0.9267305
4: -0.1598573, 0.7338556, -0.2176540, 0.8560207, -1.0158780, 0.9515096

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6746011, upper bound: 0.6728652
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47

Time for candidate selection: 2.18 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6526064, upper bound: 0.6725601
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6619454, upper bound: 0.6567131
time: 0.35 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0892951, 0.7884323, 0.0004525, 0.7221838, -0.8114790, 0.7879798
1: -0.2055373, 0.9655911, -0.1048932, 0.8939738, -1.0995111, 1.0704844
2: -0.1576910, 0.9697834, -0.0500441, 0.8598962, -1.0175872, 1.0198275
3: -0.3306990, 0.9586977, -0.2612641, 0.8577256, -1.1884246, 1.2199618
4: -0.3441186, 0.9710234, -0.2176540, 0.8560207, -1.2001393, 1.1886773

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6626402, upper bound: 0.6698976
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47

Time for candidate selection: 2.19 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6349868, upper bound: 0.6699629
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6445613, upper bound: 0.6548684
time: 0.38 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.61 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.6736547, upper bound: 0.6736547
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.6736547, upper bound: 0.6777835
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.6777835, upper bound: 0.6808372
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.6777835, upper bound: 0.6858741
IS_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.6219948, upper bound: 0.6546898
IS_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.6449059, upper bound: 0.6491899
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.6400006, upper bound: 0.6714281
IS_A1_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.6441794, upper bound: 0.6534000
IS_A2_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.6420073, upper bound: 0.6577405
IS_A2_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.6420073, upper bound: 0.6577405
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.6420073, upper bound: 0.6669338
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.6420073, upper bound: 0.6669338
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6682847
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6682847
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6763565
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6763565
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.6874736, upper bound: 0.6792575
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.6874736, upper bound: 0.6813382
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.6697728, upper bound: 0.6793275
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.6722345, upper bound: 0.6820627
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.6526064, upper bound: 0.6725601
IS_A2_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.6619454, upper bound: 0.6567131
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.6349868, upper bound: 0.6699629
IS_A2_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.61
Output dim: 0, lower bound: -0.6445613, upper bound: 0.6548684

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0213993, 0.9013143, 0.0213993, 0.9013143, -0.8799150, 0.8799150
1: -0.0770538, 1.1394897, -0.0770538, 1.1394897, -1.2165434, 1.2165434
2: -0.0367112, 0.9943478, -0.0367112, 0.9943478, -1.0310590, 1.0310590
3: -0.2404699, 1.0253115, -0.2404699, 1.0253115, -1.2657814, 1.2657814
4: -0.2202464, 0.8539461, -0.2202464, 0.8539461, -1.0741925, 1.0741925

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 47

Time for candidate selection: 1.42 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6733452, upper bound: 0.6341388
time: 0.31 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6461180, upper bound: 0.6461180
time: 0.30 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0213993, 0.9013143, -0.0028670, 1.0353181, -1.0139189, 0.9041812
1: -0.0770538, 1.1394897, -0.1088188, 1.3095722, -1.3866260, 1.2483084
2: -0.0367112, 0.9943478, -0.0732558, 1.1363764, -1.1730876, 1.0676036
3: -0.2404699, 1.0253115, -0.2704966, 1.1744311, -1.4149010, 1.2958081
4: -0.2202464, 0.8539461, -0.2937192, 0.9576517, -1.1778982, 1.1476653

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 47

Time for candidate selection: 1.38 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6733452, upper bound: 0.6450348
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6461180, upper bound: 0.6558997
time: 0.32 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0028670, 1.0353181, 0.0213993, 0.9013143, -0.9041812, 1.0139189
1: -0.1088188, 1.3095722, -0.0770538, 1.1394897, -1.2483084, 1.3866260
2: -0.0732558, 1.1363764, -0.0367112, 0.9943478, -1.0676036, 1.1730876
3: -0.2704966, 1.1744311, -0.2404699, 1.0253115, -1.2958081, 1.4149010
4: -0.2937192, 0.9576517, -0.2202464, 0.8539461, -1.1476653, 1.1778982

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5

Time for candidate selection: 1.38 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6774739, upper bound: 0.6365467
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6558995, upper bound: 0.6530225
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0028670, 1.0353181, -0.0025742, 1.0353181, -1.0381851, 1.0378923
1: -0.1088188, 1.3095722, -0.1087742, 1.3095722, -1.4183910, 1.4183464
2: -0.0732558, 1.1363764, -0.0731823, 1.1363764, -1.2096322, 1.2095587
3: -0.2704966, 1.1744311, -0.2704394, 1.1744311, -1.4449277, 1.4448705
4: -0.2937192, 0.9576517, -0.2936380, 0.9576517, -1.2513709, 1.2512897

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5

Time for candidate selection: 1.34 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6733452, upper bound: 0.6341388
time: 0.31 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6461180, upper bound: 0.6461180
time: 0.33 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0028670, 1.0353181, -0.0449519, 1.8581767, -1.8610437, 1.0802701
1: -0.1088188, 1.3095722, -0.1672845, 2.2730021, -2.3818209, 1.4768567
2: -0.0732558, 1.1363764, -0.1373940, 2.0394382, -2.1126940, 1.2737703
3: -0.2704966, 1.1744311, -0.3284359, 1.8483224, -2.1188190, 1.5028670
4: -0.2937192, 0.9576517, -0.4855750, 1.4831429, -1.7768620, 1.4432267

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6400006, upper bound: 0.6526529
time: 0.33 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6400006, upper bound: 0.6534000
time: 0.34 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, 0.0248351, 0.8239982, -0.8545651, 0.6458231
1: -0.1440713, 0.8347254, -0.0720310, 1.0479953, -1.1920667, 0.9067564
2: -0.0742025, 0.8280591, -0.0307086, 0.9141164, -0.9883189, 0.8587677
3: -0.2901301, 0.8312570, -0.2340453, 0.9604603, -1.2505904, 1.0653024
4: -0.2451730, 0.8507983, -0.2052969, 0.8165774, -1.0617504, 1.0560951

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5

Time for candidate selection: 1.81 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6420073, upper bound: 0.6577405
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6420073, upper bound: 0.6669338
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, -0.0167825, 0.9250188, -0.9555857, 0.6874408
1: -0.1440713, 0.8347254, -0.1252913, 1.1663461, -1.3104174, 0.9600167
2: -0.0742025, 0.8280591, -0.0773900, 1.0296290, -1.1038315, 0.9054491
3: -0.2901301, 0.8312570, -0.2777247, 1.0633786, -1.3535087, 1.1089818
4: -0.2451730, 0.8507983, -0.2678475, 0.8997647, -1.1449378, 1.1186459

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5

Time for candidate selection: 1.77 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6540380, upper bound: 0.6577405
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6540380, upper bound: 0.6669338
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, 0.0006924, 0.9639854, -0.9527974, 0.5641692
1: -0.0944340, 0.7130072, -0.1038828, 1.2270458, -1.3214798, 0.8168900
2: -0.0312903, 0.7124612, -0.0676236, 1.0606856, -1.0919759, 0.7800848
3: -0.2500148, 0.7301772, -0.2647138, 1.1143060, -1.3643208, 0.9948910
4: -0.1787794, 0.7720095, -0.2805611, 0.9134201, -1.0921994, 1.0525706

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6533288, upper bound: 0.6682771
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5

Time for candidate selection: 2.17 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6656445, upper bound: 0.6682847
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6682847
time: 0.33 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, -0.0360885, 1.0472777, -1.0360897, 0.6009500
1: -0.0944340, 0.7130072, -0.1492593, 1.3218184, -1.4162524, 0.8622665
2: -0.0312903, 0.7124612, -0.1066136, 1.1590223, -1.1903126, 0.8190749
3: -0.2500148, 0.7301772, -0.3045232, 1.1983278, -1.4483426, 1.0347004
4: -0.1787794, 0.7720095, -0.3275616, 0.9970202, -1.1757996, 1.0995711

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6533288, upper bound: 0.6682771
time: 0.33 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5

Time for candidate selection: 2.19 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6845282, upper bound: 0.6682847
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6599831, upper bound: 0.6682847
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, 0.0006924, 0.9639854, -0.9945524, 0.6699659
1: -0.1440713, 0.8347254, -0.1038828, 1.2270458, -1.3711171, 0.9386082
2: -0.0742025, 0.8280591, -0.0676236, 1.0606856, -1.1348882, 0.8956828
3: -0.2901301, 0.8312570, -0.2647138, 1.1143060, -1.4044361, 1.0959709
4: -0.2451730, 0.8507983, -0.2805611, 0.9134201, -1.1585931, 1.1313593

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6289553, upper bound: 0.6763565
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5

Time for candidate selection: 2.24 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6682847
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6763565
time: 0.34 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, -0.0360885, 1.0472777, -1.0778446, 0.7067467
1: -0.1440713, 0.8347254, -0.1492593, 1.3218184, -1.4658897, 0.9839847
2: -0.0742025, 0.8280591, -0.1066136, 1.1590223, -1.2332249, 0.9346728
3: -0.2901301, 0.8312570, -0.3045232, 1.1983278, -1.4884579, 1.1357803
4: -0.2451730, 0.8507983, -0.3275616, 0.9970202, -1.2421932, 1.1783600

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6289553, upper bound: 0.6763565
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5

Time for candidate selection: 2.16 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6599831, upper bound: 0.6682847
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6599831, upper bound: 0.6763565
time: 0.40 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0163505, 0.4959996, 0.0073309, 0.6935004, -0.6771499, 0.4886687
1: -0.0884151, 0.6224731, -0.0999169, 0.8545845, -0.9429996, 0.7223901
2: -0.0213518, 0.6426897, -0.0374527, 0.8373916, -0.8587434, 0.6801424
3: -0.2434175, 0.6633722, -0.2552719, 0.8256997, -1.0691173, 0.9186441
4: -0.1588757, 0.7319396, -0.1935482, 0.8381348, -0.9970105, 0.9254878

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6873948, upper bound: 0.6788838
time: 0.36 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6873948, upper bound: 0.6792575
time: 0.33 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0176041, 0.4923912, 0.0294173, 0.5563495, -0.5387454, 0.4629739
1: -0.0871999, 0.6171274, -0.0742822, 0.6929038, -0.7801037, 0.6914096
2: -0.0192325, 0.6387119, 0.0004599, 0.7029566, -0.7221891, 0.6382520
3: -0.2421939, 0.6586218, -0.2323704, 0.6962168, -0.9384108, 0.8909922
4: -0.1557364, 0.7292348, -0.1339579, 0.7538315, -0.9095680, 0.8631927

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6873948, upper bound: 0.6801620
time: 0.34 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6873948, upper bound: 0.6801620
time: 0.33 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0886977, 0.7857922, 0.0073309, 0.6935004, -0.7821981, 0.7784613
1: -0.2048204, 0.9623832, -0.0999169, 0.8545845, -1.0594049, 1.0623001
2: -0.1568820, 0.9668468, -0.0374527, 0.8373916, -0.9942737, 1.0042995
3: -0.3299770, 0.9559983, -0.2552719, 0.8256997, -1.1556768, 1.2112702
4: -0.3428915, 0.9688458, -0.1935482, 0.8381348, -1.1810262, 1.1623940

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6594252, upper bound: 0.6754512
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6594252, upper bound: 0.6793275
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0873141, 0.7822223, 0.0294173, 0.5563495, -0.6436636, 0.7528051
1: -0.2035983, 0.9574927, -0.0742822, 0.6929038, -0.8965021, 1.0317749
2: -0.1550481, 0.9631499, 0.0004599, 0.7029566, -0.8580047, 0.9626900
3: -0.3287745, 0.9514117, -0.2323704, 0.6962168, -1.0249913, 1.1837821
4: -0.3398001, 0.9661162, -0.1339579, 0.7538315, -1.0936316, 1.1000741

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6594252, upper bound: 0.6765326
time: 0.36 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6594252, upper bound: 0.6820628
time: 0.37 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0158405, 0.4982040, 0.0061345, 0.6507710, -0.6349305, 0.4920695
1: -0.0890739, 0.6250523, -0.0976210, 0.8106803, -0.8997542, 0.7226732
2: -0.0220215, 0.6451663, -0.0417109, 0.7880348, -0.8100563, 0.6868772
3: -0.2441075, 0.6654664, -0.2534082, 0.7965333, -1.0406408, 0.9188746
4: -0.1598573, 0.7338556, -0.2023642, 0.8128391, -0.9726964, 0.9362198

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5

Time for candidate selection: 1.94 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6526064, upper bound: 0.6725601
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6349868, upper bound: 0.6699629
time: 0.38 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0892951, 0.7884323, 0.0061345, 0.6507710, -0.7400662, 0.7822978
1: -0.2055373, 0.9655911, -0.0976210, 0.8106803, -1.0162176, 1.0632122
2: -0.1576910, 0.9697834, -0.0417109, 0.7880348, -0.9457258, 1.0114943
3: -0.3306990, 0.9586977, -0.2534082, 0.7965333, -1.1272323, 1.2121059
4: -0.3441186, 0.9710234, -0.2023642, 0.8128391, -1.1569576, 1.1733875

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5

Time for candidate selection: 1.94 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6349868, upper bound: 0.6465601
time: 0.39 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6349868, upper bound: 0.6548684
time: 0.40 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.54 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -0.6733452, upper bound: 0.6341388
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.54
Output dim: 0, lower bound: -0.6461180, upper bound: 0.6461180
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -0.6733452, upper bound: 0.6450348
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.54
Output dim: 0, lower bound: -0.6461180, upper bound: 0.6558997
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -0.6774739, upper bound: 0.6365467
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.54
Output dim: 0, lower bound: -0.6558995, upper bound: 0.6530225
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -0.6733452, upper bound: 0.6341388
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.54
Output dim: 0, lower bound: -0.6461180, upper bound: 0.6461180
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.54
Output dim: 0, lower bound: -0.6400006, upper bound: 0.6526529
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.54
Output dim: 0, lower bound: -0.6400006, upper bound: 0.6534000
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.54
Output dim: 0, lower bound: -0.6420073, upper bound: 0.6577405
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -0.6420073, upper bound: 0.6669338
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.54
Output dim: 0, lower bound: -0.6540380, upper bound: 0.6577405
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -0.6540380, upper bound: 0.6669338
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -0.6656445, upper bound: 0.6682847
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6682847
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -0.6845282, upper bound: 0.6682847
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -0.6599831, upper bound: 0.6682847
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6682847
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6763565
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -0.6599831, upper bound: 0.6682847
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -0.6599831, upper bound: 0.6763565
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -0.6873948, upper bound: 0.6788838
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -0.6873948, upper bound: 0.6792575
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -0.6873948, upper bound: 0.6801620
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -0.6873948, upper bound: 0.6801620
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -0.6594252, upper bound: 0.6754512
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -0.6594252, upper bound: 0.6793275
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -0.6594252, upper bound: 0.6765326
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -0.6594252, upper bound: 0.6820628
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -0.6526064, upper bound: 0.6725601
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.54
Output dim: 0, lower bound: -0.6349868, upper bound: 0.6699629
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.54
Output dim: 0, lower bound: -0.6349868, upper bound: 0.6465601
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.54
Output dim: 0, lower bound: -0.6349868, upper bound: 0.6548684

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0265114, 0.8204293, 0.0213993, 0.9013143, -0.8748028, 0.7990301
1: -0.0697420, 1.0438113, -0.0770538, 1.1394897, -1.2092316, 1.1208651
2: -0.0287337, 0.9096682, -0.0367112, 0.9943478, -1.0230815, 0.9463794
3: -0.2322266, 0.9569638, -0.2404699, 1.0253115, -1.2575381, 1.1974337
4: -0.2029902, 0.8125998, -0.2202464, 0.8539461, -1.0569363, 1.0328462

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47

Time for candidate selection: 1.50 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6341388, upper bound: 0.6341388
time: 0.31 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6341388, upper bound: 0.6341388
time: 0.32 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0265114, 0.8204293, -0.0028670, 1.0353181, -1.0088067, 0.8232963
1: -0.0697420, 1.0438113, -0.1088188, 1.3095722, -1.3793142, 1.1526301
2: -0.0287337, 0.9096682, -0.0732558, 1.1363764, -1.1651101, 0.9829240
3: -0.2322266, 0.9569638, -0.2704966, 1.1744311, -1.4066577, 1.2274604
4: -0.2029902, 0.8125998, -0.2937192, 0.9576517, -1.1606419, 1.1063190

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 5

Time for candidate selection: 1.39 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6341388, upper bound: 0.6450345
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6341388, upper bound: 0.6450345
time: 0.33 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0022838, 0.9603593, 0.0213993, 0.9013143, -0.8990304, 0.9389601
1: -0.1017990, 1.2228236, -0.0770538, 1.1394897, -1.2412887, 1.2998774
2: -0.0658340, 1.0562885, -0.0367112, 0.9943478, -1.0601819, 1.0929997
3: -0.2629817, 1.1104673, -0.2404699, 1.0253115, -1.2882931, 1.3509372
4: -0.2787794, 0.9093835, -0.2202464, 0.8539461, -1.1327255, 1.1296300

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47

Time for candidate selection: 1.62 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6450345, upper bound: 0.6365467
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6450345, upper bound: 0.6365467
time: 0.36 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0022838, 0.9603593, -0.0025742, 1.0353181, -1.0330343, 0.9629335
1: -0.1017990, 1.2228236, -0.1087742, 1.3095722, -1.4113712, 1.3315978
2: -0.0658340, 1.0562885, -0.0731823, 1.1363764, -1.2022104, 1.1294708
3: -0.2629817, 1.1104673, -0.2704394, 1.1744311, -1.4374127, 1.3809067
4: -0.2787794, 0.9093835, -0.2936380, 0.9576517, -1.2364311, 1.2030215

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 5

Time for candidate selection: 1.63 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6474423, upper bound: 0.6468780
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6474423, upper bound: 0.6468780
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, 0.0248351, 0.8239982, -0.8545651, 0.6458231
1: -0.1440713, 0.8347254, -0.0720310, 1.0479953, -1.1920667, 0.9067564
2: -0.0742025, 0.8280591, -0.0307086, 0.9141164, -0.9883189, 0.8587677
3: -0.2901301, 0.8312570, -0.2340453, 0.9604603, -1.2505904, 1.0653024
4: -0.2451730, 0.8507983, -0.2052969, 0.8165774, -1.0617504, 1.0560951

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6269718, upper bound: 0.6893970
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47

Time for candidate selection: 2.34 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6420073, upper bound: 0.6936089
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6420073, upper bound: 0.6669338
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, -0.0167825, 0.9250188, -0.9555857, 0.6874408
1: -0.1440713, 0.8347254, -0.1252913, 1.1663461, -1.3104174, 0.9600167
2: -0.0742025, 0.8280591, -0.0773900, 1.0296290, -1.1038315, 0.9054491
3: -0.2901301, 0.8312570, -0.2777247, 1.0633786, -1.3535087, 1.1089818
4: -0.2451730, 0.8507983, -0.2678475, 0.8997647, -1.1449378, 1.1186459

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6269718, upper bound: 0.6668439
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47

Time for candidate selection: 2.40 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6420073, upper bound: 0.6669338
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6540379, upper bound: 0.6669338
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, 0.0006924, 0.9639854, -0.9527974, 0.5641692
1: -0.0944340, 0.7130072, -0.1038828, 1.2270458, -1.3214798, 0.8168900
2: -0.0312903, 0.7124612, -0.0676236, 1.0606856, -1.0919759, 0.7800848
3: -0.2500148, 0.7301772, -0.2647138, 1.1143060, -1.3643208, 0.9948910
4: -0.1787794, 0.7720095, -0.2805611, 0.9134201, -1.0921994, 1.0525706

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6229196, upper bound: 0.6752626
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47

Time for candidate selection: 2.39 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6897226
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6682847
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, 0.0006924, 0.9639854, -0.9945524, 0.6699659
1: -0.1440713, 0.8347254, -0.1038828, 1.2270458, -1.3711171, 0.9386082
2: -0.0742025, 0.8280591, -0.0676236, 1.0606856, -1.1348882, 0.8956828
3: -0.2901301, 0.8312570, -0.2647138, 1.1143060, -1.4044361, 1.0959709
4: -0.2451730, 0.8507983, -0.2805611, 0.9134201, -1.1585931, 1.1313593

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6229196, upper bound: 0.6752626
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47

Time for candidate selection: 2.22 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6897226
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6682847
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, -0.0360885, 1.0472777, -1.0360897, 0.6009500
1: -0.0944340, 0.7130072, -0.1492593, 1.3218184, -1.4162524, 0.8622665
2: -0.0312903, 0.7124612, -0.1066136, 1.1590223, -1.1903126, 0.8190749
3: -0.2500148, 0.7301772, -0.3045232, 1.1983278, -1.4483426, 1.0347004
4: -0.1787794, 0.7720095, -0.3275616, 0.9970202, -1.1757996, 1.0995711

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6229196, upper bound: 0.6595865
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 5

Time for candidate selection: 2.15 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6682847
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6682847
time: 0.34 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, -0.0360885, 1.0472777, -1.0778446, 0.7067467
1: -0.1440713, 0.8347254, -0.1492593, 1.3218184, -1.4658897, 0.9839847
2: -0.0742025, 0.8280591, -0.1066136, 1.1590223, -1.2332249, 0.9346728
3: -0.2901301, 0.8312570, -0.3045232, 1.1983278, -1.4884579, 1.1357803
4: -0.2451730, 0.8507983, -0.3275616, 0.9970202, -1.2421932, 1.1783600

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6229196, upper bound: 0.6595865
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 5

Time for candidate selection: 2.19 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6682847
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6682847
time: 0.34 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, 0.0006924, 0.9639854, -0.9527974, 0.5641692
1: -0.0944340, 0.7130072, -0.1038828, 1.2270458, -1.3214798, 0.8168900
2: -0.0312903, 0.7124612, -0.0676236, 1.0606856, -1.0919759, 0.7800848
3: -0.2500148, 0.7301772, -0.2647138, 1.1143060, -1.3643208, 0.9948910
4: -0.1787794, 0.7720095, -0.2805611, 0.9134201, -1.0921994, 1.0525706

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6229196, upper bound: 0.6752626
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47

Time for candidate selection: 2.19 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6897226
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6682847
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, 0.0006924, 0.9639854, -0.9945524, 0.6699659
1: -0.1440713, 0.8347254, -0.1038828, 1.2270458, -1.3711171, 0.9386082
2: -0.0742025, 0.8280591, -0.0676236, 1.0606856, -1.1348882, 0.8956828
3: -0.2901301, 0.8312570, -0.2647138, 1.1143060, -1.4044361, 1.0959709
4: -0.2451730, 0.8507983, -0.2805611, 0.9134201, -1.1585931, 1.1313593

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6229196, upper bound: 0.6802993
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47

Time for candidate selection: 2.19 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6977943
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6763565
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, -0.0360885, 1.0472777, -1.0360897, 0.6009500
1: -0.0944340, 0.7130072, -0.1492593, 1.3218184, -1.4162524, 0.8622665
2: -0.0312903, 0.7124612, -0.1066136, 1.1590223, -1.1903126, 0.8190749
3: -0.2500148, 0.7301772, -0.3045232, 1.1983278, -1.4483426, 1.0347004
4: -0.1787794, 0.7720095, -0.3275616, 0.9970202, -1.1757996, 1.0995711

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6229196, upper bound: 0.6595865
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 5

Time for candidate selection: 2.25 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6682847
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6682847
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, -0.0360885, 1.0472777, -1.0778446, 0.7067467
1: -0.1440713, 0.8347254, -0.1492593, 1.3218184, -1.4658897, 0.9839847
2: -0.0742025, 0.8280591, -0.1066136, 1.1590223, -1.2332249, 0.9346728
3: -0.2901301, 0.8312570, -0.3045232, 1.1983278, -1.4884579, 1.1357803
4: -0.2451730, 0.8507983, -0.3275616, 0.9970202, -1.2421932, 1.1783600

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6229196, upper bound: 0.6595865
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 5

Time for candidate selection: 2.27 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6763565
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6599831, upper bound: 0.6763565
time: 0.39 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0185366, 0.5137528, 0.0073309, 0.6935004, -0.6749638, 0.5064219
1: -0.0853710, 0.6410187, -0.0999169, 0.8545845, -0.9399555, 0.7409357
2: -0.0191109, 0.6585484, -0.0374527, 0.8373916, -0.8565025, 0.6960011
3: -0.2388380, 0.6765358, -0.2552719, 0.8256997, -1.0645378, 0.9318078
4: -0.1577795, 0.7368335, -0.1935482, 0.8381348, -0.9959143, 0.9303817

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 47

Time for candidate selection: 1.46 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6527988, upper bound: 0.6783239
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6662293, upper bound: 0.6621957
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0387027, 0.4375544, 0.0073309, 0.6935004, -0.6547977, 0.4302235
1: -0.0613070, 0.5365020, -0.0999169, 0.8545845, -0.9158914, 0.6364189
2: 0.0174038, 0.5755119, -0.0374527, 0.8373916, -0.8199878, 0.6129646
3: -0.2170205, 0.5777811, -0.2552719, 0.8256997, -1.0427203, 0.8330530
4: -0.1004417, 0.6784918, -0.1935482, 0.8381348, -0.9385765, 0.8720400

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 47

Time for candidate selection: 1.57 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6527988, upper bound: 0.6788786
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6662293, upper bound: 0.6628001
time: 0.37 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0185366, 0.5137528, 0.0294173, 0.5563495, -0.5378129, 0.4843355
1: -0.0853710, 0.6410187, -0.0742822, 0.6929038, -0.7782748, 0.7153009
2: -0.0191109, 0.6585484, 0.0004599, 0.7029566, -0.7220675, 0.6580884
3: -0.2388380, 0.6765358, -0.2323704, 0.6962168, -0.9350548, 0.9089062
4: -0.1577795, 0.7368335, -0.1339579, 0.7538315, -0.9116110, 0.8707913

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5

Time for candidate selection: 1.45 seconds

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6873844, upper bound: 0.6801620
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6873844, upper bound: 0.6801620
time: 0.39 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0387027, 0.4375544, 0.0294173, 0.5563495, -0.5176468, 0.4081371
1: -0.0613070, 0.5365020, -0.0742822, 0.6929038, -0.7542108, 0.6107842
2: 0.0174038, 0.5755119, 0.0004599, 0.7029566, -0.6855527, 0.5750520
3: -0.2170205, 0.5777811, -0.2323704, 0.6962168, -0.9132373, 0.8101515
4: -0.1004417, 0.6784918, -0.1339579, 0.7538315, -0.8542732, 0.8124496

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5

Time for candidate selection: 1.45 seconds

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6873844, upper bound: 0.6788838
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6873844, upper bound: 0.6813017
time: 0.37 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0860746, 0.7981433, 0.0073309, 0.6935004, -0.7795750, 0.7908124
1: -0.2016160, 0.9745648, -0.0999169, 0.8545845, -1.0562005, 1.0744817
2: -0.1532123, 0.9791473, -0.0374527, 0.8373916, -0.9906039, 1.0165999
3: -0.3256128, 0.9652334, -0.2552719, 0.8256997, -1.1513126, 1.2205054
4: -0.3422897, 0.9691569, -0.1935482, 0.8381348, -1.1804245, 1.1627052

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 47

Time for candidate selection: 1.45 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6368137, upper bound: 0.6753682
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6453810, upper bound: 0.6589475
time: 0.38 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0603211, 0.6870018, 0.0073309, 0.6935004, -0.7538215, 0.6796709
1: -0.1749001, 0.8297409, -0.0999169, 0.8545845, -1.0294845, 0.9296579
2: -0.1159739, 0.8579424, -0.0374527, 0.8373916, -0.9533656, 0.8953951
3: -0.3003569, 0.8422859, -0.2552719, 0.8256997, -1.1260567, 1.0975578
4: -0.2768354, 0.8908350, -0.1935482, 0.8381348, -1.1149702, 1.0843832

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 47

Time for candidate selection: 1.46 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6368137, upper bound: 0.6753682
time: 0.36 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6453810, upper bound: 0.6627969
time: 0.38 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0860746, 0.7981433, 0.0294173, 0.5563495, -0.6424241, 0.7687260
1: -0.2016160, 0.9745648, -0.0742822, 0.6929038, -0.8945199, 1.0488470
2: -0.1532123, 0.9791473, 0.0004599, 0.7029566, -0.8561689, 0.9786873
3: -0.3256128, 0.9652334, -0.2323704, 0.6962168, -1.0218296, 1.1976038
4: -0.3422897, 0.9691569, -0.1339579, 0.7538315, -1.0961212, 1.1031148

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5

Time for candidate selection: 1.45 seconds

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6594252, upper bound: 0.6765326
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6594252, upper bound: 0.6765326
time: 0.40 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0603211, 0.6870018, 0.0294173, 0.5563495, -0.6166706, 0.6575845
1: -0.1749001, 0.8297409, -0.0742822, 0.6929038, -0.8678039, 0.9040231
2: -0.1159739, 0.8579424, 0.0004599, 0.7029566, -0.8189305, 0.8574825
3: -0.3003569, 0.8422859, -0.2323704, 0.6962168, -0.9965737, 1.0746562
4: -0.2768354, 0.8908350, -0.1339579, 0.7538315, -1.0306669, 1.0247929

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5

Time for candidate selection: 1.48 seconds

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6338961, upper bound: 0.6753682
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6443170, upper bound: 0.6659172
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0158405, 0.4982040, 0.0061345, 0.6507710, -0.6349305, 0.4920695
1: -0.0890739, 0.6250523, -0.0976210, 0.8106803, -0.8997542, 0.7226732
2: -0.0220215, 0.6451663, -0.0417109, 0.7880348, -0.8100563, 0.6868772
3: -0.2441075, 0.6654664, -0.2534082, 0.7965333, -1.0406408, 0.9188746
4: -0.1598573, 0.7338556, -0.2023642, 0.8128391, -0.9726964, 0.9362198

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6373208, upper bound: 0.6724766
time: 0.40 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47

Time for candidate selection: 2.33 seconds

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6349868, upper bound: 0.6699629
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6349868, upper bound: 0.6699629
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0892661, 0.7843834, 0.0061345, 0.6507710, -0.7400371, 0.7782488
1: -0.2051971, 0.9613039, -0.0976210, 0.8106803, -1.0158774, 1.0589249
2: -0.1573753, 0.9656872, -0.0417109, 0.7880348, -0.9454101, 1.0073980
3: -0.3306475, 0.9540346, -0.2534082, 0.7965333, -1.1271808, 1.2074428
4: -0.3440671, 0.9667487, -0.2023642, 0.8128391, -1.1569061, 1.1691129

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6347579, upper bound: 0.6698976
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47

Time for candidate selection: 2.32 seconds

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6349868, upper bound: 0.6699629
time: 0.36 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6349868, upper bound: 0.6699629
time: 0.36 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 4.85 seconds
IS_A1_B1_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 4.85
Output dim: 0, lower bound: -0.6341388, upper bound: 0.6341388
IS_A1_B1_A1_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 4.85
Output dim: 0, lower bound: -0.6341388, upper bound: 0.6341388
IS_A1_B1_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 4.85
Output dim: 0, lower bound: -0.6341388, upper bound: 0.6450345
IS_A1_B1_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 4.85
Output dim: 0, lower bound: -0.6341388, upper bound: 0.6450345
IS_A1_B1_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 4.85
Output dim: 0, lower bound: -0.6450345, upper bound: 0.6365467
IS_A1_B1_A1_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 4.85
Output dim: 0, lower bound: -0.6450345, upper bound: 0.6365467
IS_A1_B1_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 4.85
Output dim: 0, lower bound: -0.6474423, upper bound: 0.6468780
IS_A1_B1_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 4.85
Output dim: 0, lower bound: -0.6474423, upper bound: 0.6468780
IS_A2_B1_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.85
Output dim: 0, lower bound: -0.6420073, upper bound: 0.6936089
IS_A2_B1_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.85
Output dim: 0, lower bound: -0.6420073, upper bound: 0.6669338
IS_A2_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.85
Output dim: 0, lower bound: -0.6420073, upper bound: 0.6669338
IS_A2_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.85
Output dim: 0, lower bound: -0.6540379, upper bound: 0.6669338
IS_A2_B1_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.85
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6897226
IS_A2_B1_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.85
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6682847
IS_A2_B1_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.85
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6897226
IS_A2_B1_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.85
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6682847
IS_A2_B1_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.85
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6682847
IS_A2_B1_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.85
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6682847
IS_A2_B1_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.85
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6682847
IS_A2_B1_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.85
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6682847
IS_A2_B1_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.85
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6897226
IS_A2_B1_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.85
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6682847
IS_A2_B1_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.85
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6977943
IS_A2_B1_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.85
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6763565
IS_A2_B1_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.85
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6682847
IS_A2_B1_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.85
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6682847
IS_A2_B1_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.85
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6763565
IS_A2_B1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.85
Output dim: 0, lower bound: -0.6599831, upper bound: 0.6763565
IS_A2_B2_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.85
Output dim: 0, lower bound: -0.6527988, upper bound: 0.6783239
IS_A2_B2_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.85
Output dim: 0, lower bound: -0.6662293, upper bound: 0.6621957
IS_A2_B2_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.85
Output dim: 0, lower bound: -0.6527988, upper bound: 0.6788786
IS_A2_B2_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.85
Output dim: 0, lower bound: -0.6662293, upper bound: 0.6628001
IS_A2_B2_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.85
Output dim: 0, lower bound: -0.6873844, upper bound: 0.6801620
IS_A2_B2_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.85
Output dim: 0, lower bound: -0.6873844, upper bound: 0.6801620
IS_A2_B2_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.85
Output dim: 0, lower bound: -0.6873844, upper bound: 0.6788838
IS_A2_B2_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.85
Output dim: 0, lower bound: -0.6873844, upper bound: 0.6813017
IS_A2_B2_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.85
Output dim: 0, lower bound: -0.6368137, upper bound: 0.6753682
IS_A2_B2_A1_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 4.85
Output dim: 0, lower bound: -0.6453810, upper bound: 0.6589475
IS_A2_B2_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.85
Output dim: 0, lower bound: -0.6368137, upper bound: 0.6753682
IS_A2_B2_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.85
Output dim: 0, lower bound: -0.6453810, upper bound: 0.6627969
IS_A2_B2_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.85
Output dim: 0, lower bound: -0.6594252, upper bound: 0.6765326
IS_A2_B2_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.85
Output dim: 0, lower bound: -0.6594252, upper bound: 0.6765326
IS_A2_B2_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.85
Output dim: 0, lower bound: -0.6338961, upper bound: 0.6753682
IS_A2_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.85
Output dim: 0, lower bound: -0.6443170, upper bound: 0.6659172
IS_A2_B2_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.85
Output dim: 0, lower bound: -0.6349868, upper bound: 0.6699629
IS_A2_B2_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.85
Output dim: 0, lower bound: -0.6349868, upper bound: 0.6699629
IS_A2_B2_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.85
Output dim: 0, lower bound: -0.6349868, upper bound: 0.6699629
IS_A2_B2_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.85
Output dim: 0, lower bound: -0.6349868, upper bound: 0.6699629

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, 0.0248351, 0.8239982, -0.8545651, 0.6458231
1: -0.1440713, 0.8347254, -0.0720310, 1.0479953, -1.1920667, 0.9067564
2: -0.0742025, 0.8280591, -0.0307086, 0.9141164, -0.9883189, 0.8587677
3: -0.2901301, 0.8312570, -0.2340453, 0.9604603, -1.2505904, 1.0653024
4: -0.2451730, 0.8507983, -0.2052969, 0.8165774, -1.0617504, 1.0560951

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5

Time for candidate selection: 1.88 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6420073, upper bound: 0.6577405
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6420073, upper bound: 0.6669338
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, -0.0167825, 0.9152751, -0.9458420, 0.6874408
1: -0.1440713, 0.8347254, -0.1252913, 1.1558418, -1.2999132, 0.9600167
2: -0.0742025, 0.8280591, -0.0773900, 1.0191219, -1.0933244, 0.9054491
3: -0.2901301, 0.8312570, -0.2777247, 1.0548213, -1.3449514, 1.1089818
4: -0.2451730, 0.8507983, -0.2678475, 0.8919395, -1.1371124, 1.1186459

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5

Time for candidate selection: 1.92 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6420073, upper bound: 0.6577405
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6420073, upper bound: 0.6669338
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, 0.0248351, 0.8239982, -0.8545651, 0.6458231
1: -0.1440713, 0.8347254, -0.0720310, 1.0479953, -1.1920667, 0.9067564
2: -0.0742025, 0.8280591, -0.0307086, 0.9141164, -0.9883189, 0.8587677
3: -0.2901301, 0.8312570, -0.2340453, 0.9604603, -1.2505904, 1.0653024
4: -0.2451730, 0.8507983, -0.2052969, 0.8165774, -1.0617504, 1.0560951

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5

Time for candidate selection: 1.88 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6420073, upper bound: 0.6577405
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6420073, upper bound: 0.6669338
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, -0.0167825, 0.9250188, -0.9555857, 0.6874408
1: -0.1440713, 0.8347254, -0.1252913, 1.1663461, -1.3104174, 0.9600167
2: -0.0742025, 0.8280591, -0.0773900, 1.0296290, -1.1038315, 0.9054491
3: -0.2901301, 0.8312570, -0.2777247, 1.0633786, -1.3535087, 1.1089818
4: -0.2451730, 0.8507983, -0.2678475, 0.8997647, -1.1449378, 1.1186459

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5

Time for candidate selection: 1.89 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6540380, upper bound: 0.6577405
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6540380, upper bound: 0.6669338
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, 0.0006924, 0.9639854, -0.9527974, 0.5641692
1: -0.0944340, 0.7130072, -0.1038828, 1.2270458, -1.3214798, 0.8168900
2: -0.0312903, 0.7124612, -0.0676236, 1.0606856, -1.0919759, 0.7800848
3: -0.2500148, 0.7301772, -0.2647138, 1.1143060, -1.3643208, 0.9948910
4: -0.1787794, 0.7720095, -0.2805611, 0.9134201, -1.0921994, 1.0525706

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6533288, upper bound: 0.6682771
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5

Time for candidate selection: 2.28 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6656445, upper bound: 0.6682847
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6682847
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, -0.0360885, 1.0472777, -1.0360897, 0.6009500
1: -0.0944340, 0.7130072, -0.1492593, 1.3218184, -1.4162524, 0.8622665
2: -0.0312903, 0.7124612, -0.1066136, 1.1590223, -1.1903126, 0.8190749
3: -0.2500148, 0.7301772, -0.3045232, 1.1983278, -1.4483426, 1.0347004
4: -0.1787794, 0.7720095, -0.3275616, 0.9970202, -1.1757996, 1.0995711

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6533288, upper bound: 0.6682771
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5

Time for candidate selection: 2.31 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6656445, upper bound: 0.6577405
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6682847
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, 0.0006924, 0.9639854, -0.9945524, 0.6699659
1: -0.1440713, 0.8347254, -0.1038828, 1.2270458, -1.3711171, 0.9386082
2: -0.0742025, 0.8280591, -0.0676236, 1.0606856, -1.1348882, 0.8956828
3: -0.2901301, 0.8312570, -0.2647138, 1.1143060, -1.4044361, 1.0959709
4: -0.2451730, 0.8507983, -0.2805611, 0.9134201, -1.1585931, 1.1313593

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6289553, upper bound: 0.6682771
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5

Time for candidate selection: 2.36 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6682847
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6682847
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, -0.0360885, 1.0472777, -1.0778446, 0.7067467
1: -0.1440713, 0.8347254, -0.1492593, 1.3218184, -1.4658897, 0.9839847
2: -0.0742025, 0.8280591, -0.1066136, 1.1590223, -1.2332249, 0.9346728
3: -0.2901301, 0.8312570, -0.3045232, 1.1983278, -1.4884579, 1.1357803
4: -0.2451730, 0.8507983, -0.3275616, 0.9970202, -1.2421932, 1.1783600

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6289553, upper bound: 0.6682771
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5

Time for candidate selection: 2.31 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6682847
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6682847
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, 0.0006924, 0.9639854, -0.9527974, 0.5641692
1: -0.0944340, 0.7130072, -0.1038828, 1.2270458, -1.3214798, 0.8168900
2: -0.0312903, 0.7124612, -0.0676236, 1.0606856, -1.0919759, 0.7800848
3: -0.2500148, 0.7301772, -0.2647138, 1.1143060, -1.3643208, 0.9948910
4: -0.1787794, 0.7720095, -0.2805611, 0.9134201, -1.0921994, 1.0525706

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6533288, upper bound: 0.6682771
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5

Time for candidate selection: 2.28 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6656445, upper bound: 0.6577405
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6682847
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, -0.0360885, 1.0472777, -1.0360897, 0.6009500
1: -0.0944340, 0.7130072, -0.1492593, 1.3218184, -1.4162524, 0.8622665
2: -0.0312903, 0.7124612, -0.1066136, 1.1590223, -1.1903126, 0.8190749
3: -0.2500148, 0.7301772, -0.3045232, 1.1983278, -1.4483426, 1.0347004
4: -0.1787794, 0.7720095, -0.3275616, 0.9970202, -1.1757996, 1.0995711

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6533288, upper bound: 0.6682771
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5

Time for candidate selection: 2.29 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6845282, upper bound: 0.6682847
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6599831, upper bound: 0.6682847
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, 0.0006924, 0.9639854, -0.9945524, 0.6699659
1: -0.1440713, 0.8347254, -0.1038828, 1.2270458, -1.3711171, 0.9386082
2: -0.0742025, 0.8280591, -0.0676236, 1.0606856, -1.1348882, 0.8956828
3: -0.2901301, 0.8312570, -0.2647138, 1.1143060, -1.4044361, 1.0959709
4: -0.2451730, 0.8507983, -0.2805611, 0.9134201, -1.1585931, 1.1313593

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6289553, upper bound: 0.6682771
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5

Time for candidate selection: 2.32 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6682847
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6682847
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, -0.0360885, 1.0472777, -1.0778446, 0.7067467
1: -0.1440713, 0.8347254, -0.1492593, 1.3218184, -1.4658897, 0.9839847
2: -0.0742025, 0.8280591, -0.1066136, 1.1590223, -1.2332249, 0.9346728
3: -0.2901301, 0.8312570, -0.3045232, 1.1983278, -1.4884579, 1.1357803
4: -0.2451730, 0.8507983, -0.3275616, 0.9970202, -1.2421932, 1.1783600

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6289553, upper bound: 0.6682771
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5

Time for candidate selection: 2.34 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6599831, upper bound: 0.6682847
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6599831, upper bound: 0.6682847
time: 0.44 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, 0.0006924, 0.9639854, -0.9527974, 0.5641692
1: -0.0944340, 0.7130072, -0.1038828, 1.2270458, -1.3214798, 0.8168900
2: -0.0312903, 0.7124612, -0.0676236, 1.0606856, -1.0919759, 0.7800848
3: -0.2500148, 0.7301772, -0.2647138, 1.1143060, -1.3643208, 0.9948910
4: -0.1787794, 0.7720095, -0.2805611, 0.9134201, -1.0921994, 1.0525706

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6533288, upper bound: 0.6682771
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5

Time for candidate selection: 2.34 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6656445, upper bound: 0.6682847
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6682847
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, -0.0360885, 1.0472777, -1.0360897, 0.6009500
1: -0.0944340, 0.7130072, -0.1492593, 1.3218184, -1.4162524, 0.8622665
2: -0.0312903, 0.7124612, -0.1066136, 1.1590223, -1.1903126, 0.8190749
3: -0.2500148, 0.7301772, -0.3045232, 1.1983278, -1.4483426, 1.0347004
4: -0.1787794, 0.7720095, -0.3275616, 0.9970202, -1.1757996, 1.0995711

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6533288, upper bound: 0.6682771
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5

Time for candidate selection: 2.39 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6656445, upper bound: 0.6577405
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6682847
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, 0.0006924, 0.9639854, -0.9945524, 0.6699659
1: -0.1440713, 0.8347254, -0.1038828, 1.2270458, -1.3711171, 0.9386082
2: -0.0742025, 0.8280591, -0.0676236, 1.0606856, -1.1348882, 0.8956828
3: -0.2901301, 0.8312570, -0.2647138, 1.1143060, -1.4044361, 1.0959709
4: -0.2451730, 0.8507983, -0.2805611, 0.9134201, -1.1585931, 1.1313593

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6289553, upper bound: 0.6763565
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5

Time for candidate selection: 2.35 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6682847
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6763565
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, -0.0360885, 1.0472777, -1.0778446, 0.7067467
1: -0.1440713, 0.8347254, -0.1492593, 1.3218184, -1.4658897, 0.9839847
2: -0.0742025, 0.8280591, -0.1066136, 1.1590223, -1.2332249, 0.9346728
3: -0.2901301, 0.8312570, -0.3045232, 1.1983278, -1.4884579, 1.1357803
4: -0.2451730, 0.8507983, -0.3275616, 0.9970202, -1.2421932, 1.1783600

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6289553, upper bound: 0.6763565
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5

Time for candidate selection: 2.41 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6682847
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6763565
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, 0.0006924, 0.9639854, -0.9527974, 0.5641692
1: -0.0944340, 0.7130072, -0.1038828, 1.2270458, -1.3214798, 0.8168900
2: -0.0312903, 0.7124612, -0.0676236, 1.0606856, -1.0919759, 0.7800848
3: -0.2500148, 0.7301772, -0.2647138, 1.1143060, -1.3643208, 0.9948910
4: -0.1787794, 0.7720095, -0.2805611, 0.9134201, -1.0921994, 1.0525706

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6533288, upper bound: 0.6682771
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5

Time for candidate selection: 2.35 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6656445, upper bound: 0.6577405
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6682847
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, -0.0360885, 1.0472777, -1.0360897, 0.6009500
1: -0.0944340, 0.7130072, -0.1492593, 1.3218184, -1.4162524, 0.8622665
2: -0.0312903, 0.7124612, -0.1066136, 1.1590223, -1.1903126, 0.8190749
3: -0.2500148, 0.7301772, -0.3045232, 1.1983278, -1.4483426, 1.0347004
4: -0.1787794, 0.7720095, -0.3275616, 0.9970202, -1.1757996, 1.0995711

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6533288, upper bound: 0.6682771
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5

Time for candidate selection: 2.35 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6845282, upper bound: 0.6682847
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6599831, upper bound: 0.6682847
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, 0.0006924, 0.9639854, -0.9945524, 0.6699659
1: -0.1440713, 0.8347254, -0.1038828, 1.2270458, -1.3711171, 0.9386082
2: -0.0742025, 0.8280591, -0.0676236, 1.0606856, -1.1348882, 0.8956828
3: -0.2901301, 0.8312570, -0.2647138, 1.1143060, -1.4044361, 1.0959709
4: -0.2451730, 0.8507983, -0.2805611, 0.9134201, -1.1585931, 1.1313593

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6289553, upper bound: 0.6763565
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5

Time for candidate selection: 2.43 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6682847
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6435072, upper bound: 0.6763565
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, -0.0360885, 1.0472777, -1.0778446, 0.7067467
1: -0.1440713, 0.8347254, -0.1492593, 1.3218184, -1.4658897, 0.9839847
2: -0.0742025, 0.8280591, -0.1066136, 1.1590223, -1.2332249, 0.9346728
3: -0.2901301, 0.8312570, -0.3045232, 1.1983278, -1.4884579, 1.1357803
4: -0.2451730, 0.8507983, -0.3275616, 0.9970202, -1.2421932, 1.1783600

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6289553, upper bound: 0.6763565
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5

Time for candidate selection: 2.38 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6599831, upper bound: 0.6682847
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6599831, upper bound: 0.6763565
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0185366, 0.5137528, 0.0136061, 0.5909360, -0.5723994, 0.5001467
1: -0.0853710, 0.6410187, -0.0912752, 0.7387895, -0.8241605, 0.7322940
2: -0.0191109, 0.6585484, -0.0286748, 0.7339988, -0.7531098, 0.6872232
3: -0.2388380, 0.6765358, -0.2452104, 0.7454442, -0.9842821, 0.9217463
4: -0.1577795, 0.7368335, -0.1770196, 0.7780294, -0.9358089, 0.9138530

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47

Time for candidate selection: 1.66 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6567166, upper bound: 0.6529388
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6567166, upper bound: 0.6621957
time: 0.39 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0185366, 0.5137528, -0.0276310, 0.7056582, -0.6871216, 0.5413839
1: -0.0853710, 0.6410187, -0.1400287, 0.8686816, -0.9540526, 0.7810475
2: -0.0191109, 0.6585484, -0.0711074, 0.8583306, -0.8774415, 0.7296557
3: -0.2388380, 0.6765358, -0.2846189, 0.8510476, -1.0898856, 0.9611547
4: -0.1577795, 0.7368335, -0.2430694, 0.8599038, -1.0176833, 0.9799029

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47

Time for candidate selection: 1.62 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6670627, upper bound: 0.6529388
time: 0.39 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6670627, upper bound: 0.6621957
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0387027, 0.4375544, 0.0136061, 0.5909360, -0.5522333, 0.4239483
1: -0.0613070, 0.5365020, -0.0912752, 0.7387895, -0.8000965, 0.6277772
2: 0.0174038, 0.5755119, -0.0286748, 0.7339988, -0.7165950, 0.6041868
3: -0.2170205, 0.5777811, -0.2452104, 0.7454442, -0.9624647, 0.8229915
4: -0.1004417, 0.6784918, -0.1770196, 0.7780294, -0.8784711, 0.8555114

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5

Time for candidate selection: 1.64 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6527988, upper bound: 0.6788786
time: 0.40 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6414657, upper bound: 0.6788599
time: 0.40 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0387027, 0.4375544, -0.0276310, 0.7056582, -0.6669555, 0.4651855
1: -0.0613070, 0.5365020, -0.1400287, 0.8686816, -0.9299885, 0.6765307
2: 0.0174038, 0.5755119, -0.0711074, 0.8583306, -0.8409268, 0.6466193
3: -0.2170205, 0.5777811, -0.2846189, 0.8510476, -1.0680681, 0.8623999
4: -0.1004417, 0.6784918, -0.2430694, 0.8599038, -0.9603455, 0.9215612

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5

Time for candidate selection: 1.60 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6662293, upper bound: 0.6628001
time: 0.40 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6534066, upper bound: 0.6627403
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0185366, 0.5137528, 0.0387027, 0.4375544, -0.4190179, 0.4750501
1: -0.0853710, 0.6410187, -0.0613070, 0.5365020, -0.6218730, 0.7023257
2: -0.0191109, 0.6585484, 0.0174038, 0.5755119, -0.5946229, 0.6411445
3: -0.2388380, 0.6765358, -0.2170205, 0.5777811, -0.8166190, 0.8935564
4: -0.1577795, 0.7368335, -0.1004417, 0.6784918, -0.8362712, 0.8372751

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47

Time for candidate selection: 1.62 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6870463, upper bound: 0.6534791
time: 0.39 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6671286, upper bound: 0.6630272
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0185366, 0.5137528, -0.0538366, 0.6856053, -0.6670687, 0.5675894
1: -0.0853710, 0.6410187, -0.1734412, 0.8282951, -0.9136661, 0.8144599
2: -0.0191109, 0.6585484, -0.1139090, 0.8565001, -0.8756111, 0.7724574
3: -0.2388380, 0.6765358, -0.2983873, 0.8408561, -1.0796940, 0.9749231
4: -0.1577795, 0.7368335, -0.2739153, 0.8889887, -1.0467682, 1.0107487

Time for backsubstitution: 2.00 seconds
Binary search (step 1): status=Status.UNKNOWN, low=0.0017312, high=0.0467529, mid=0.0467529, abs_max=0.7819017171859741
rel_dist={0: [-0.7029083079558228, 0.7029083079558232]}

## Binary search (step 2) starts
Candidate diff: 0.0242421


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6933877, upper bound: 0.6862236
time: 0.33 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6937108, upper bound: 0.6937108
time: 0.36 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.86 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.86
Output dim: 0, lower bound: -0.6933877, upper bound: 0.6862236
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.86
Output dim: 0, lower bound: -0.6937108, upper bound: 0.6937108

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0017378, 0.9577341, -0.0226004, 0.6992232, -0.7009610, 0.9803345
1: -0.1117451, 1.2035816, -0.1409330, 0.8667333, -0.9784784, 1.3445146
2: -0.0625789, 1.0695953, -0.0639827, 0.8644909, -0.9270698, 1.1335781
3: -0.2743888, 1.0868173, -0.3000648, 0.8557575, -1.1301463, 1.3868821
4: -0.2543875, 0.9355542, -0.2339711, 0.8990978, -1.1534853, 1.1695254

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6862236, upper bound: 0.6862236
time: 0.32 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6862236, upper bound: 0.6862236
time: 0.33 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0124662, 0.7046853, -0.0323386, 0.7495631, -0.7620293, 0.7370239
1: -0.1260490, 0.8739170, -0.1552227, 0.9255341, -1.0515832, 1.0291396
2: -0.0592167, 0.8673251, -0.0741289, 0.9267865, -0.9860032, 0.9414539
3: -0.2830637, 0.8561093, -0.3143692, 0.9027434, -1.1858070, 1.1704785
4: -0.2274244, 0.8807485, -0.2515757, 0.9410419, -1.1684663, 1.1323242

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6862236, upper bound: 0.6933184
time: 0.33 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6862236, upper bound: 0.6937108
time: 0.36 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.57 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.57
Output dim: 0, lower bound: -0.6862236, upper bound: 0.6862236
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.57
Output dim: 0, lower bound: -0.6862236, upper bound: 0.6862236
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.57
Output dim: 0, lower bound: -0.6862236, upper bound: 0.6933184
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.57
Output dim: 0, lower bound: -0.6862236, upper bound: 0.6937108

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0017378, 0.9577341, -0.0017378, 0.9577341, -0.9594719, 0.9594719
1: -0.1117451, 1.2035816, -0.1117451, 1.2035816, -1.3153267, 1.3153267
2: -0.0625789, 1.0695953, -0.0625789, 1.0695953, -1.1321743, 1.1321743
3: -0.2743888, 1.0868173, -0.2743888, 1.0868173, -1.3612061, 1.3612061
4: -0.2543875, 0.9355542, -0.2543875, 0.9355542, -1.1899416, 1.1899416

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6756641, upper bound: 0.6516575
time: 0.30 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6609718, upper bound: 0.6438791
time: 0.29 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0017378, 0.9577341, -0.0124662, 0.7046853, -0.7064232, 0.9702003
1: -0.1117451, 1.2035816, -0.1260490, 0.8739170, -0.9856621, 1.3296306
2: -0.0625789, 1.0695953, -0.0592167, 0.8673251, -0.9299040, 1.1288121
3: -0.2743888, 1.0868173, -0.2830637, 0.8561093, -1.1304981, 1.3698809
4: -0.2543875, 0.9355542, -0.2274244, 0.8807485, -1.1351360, 1.1629786

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6756641, upper bound: 0.6516575
time: 0.29 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6609718, upper bound: 0.6438791
time: 0.31 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0124662, 0.7046853, -0.0017378, 0.9577341, -0.9702003, 0.7064232
1: -0.1260490, 0.8739170, -0.1117451, 1.2035816, -1.3296306, 0.9856621
2: -0.0592167, 0.8673251, -0.0625789, 1.0695953, -1.1288121, 0.9299040
3: -0.2830637, 0.8561093, -0.2743888, 1.0868173, -1.3698809, 1.1304981
4: -0.2274244, 0.8807485, -0.2543875, 0.9355542, -1.1629786, 1.1351360

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6851157, upper bound: 0.6881240
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6855405, upper bound: 0.6920244
time: 0.36 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0124662, 0.7046853, -0.0124662, 0.7046853, -0.7171515, 0.7171515
1: -0.1260490, 0.8739170, -0.1260490, 0.8739170, -0.9999660, 0.9999660
2: -0.0592167, 0.8673251, -0.0592167, 0.8673251, -0.9265418, 0.9265418
3: -0.2830637, 0.8561093, -0.2830637, 0.8561093, -1.1391729, 1.1391729
4: -0.2274244, 0.8807485, -0.2274244, 0.8807485, -1.1081729, 1.1081729

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6851157, upper bound: 0.6917761
time: 0.32 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6855405, upper bound: 0.6920244
time: 0.34 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.19 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 0, lower bound: -0.6756641, upper bound: 0.6516575
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 3.19
Output dim: 0, lower bound: -0.6609718, upper bound: 0.6438791
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 0, lower bound: -0.6756641, upper bound: 0.6516575
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 3.19
Output dim: 0, lower bound: -0.6609718, upper bound: 0.6438791
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 0, lower bound: -0.6851157, upper bound: 0.6881240
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 0, lower bound: -0.6855405, upper bound: 0.6920244
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 0, lower bound: -0.6851157, upper bound: 0.6917761
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 0, lower bound: -0.6855405, upper bound: 0.6920244

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0000615, 0.9523284, -0.0017378, 0.9577341, -0.9576726, 0.9540663
1: -0.1095166, 1.1974459, -0.1117451, 1.2035816, -1.3130982, 1.3091910
2: -0.0606787, 1.0631847, -0.0625789, 1.0695953, -1.1302741, 1.1257637
3: -0.2725427, 1.0815248, -0.2743888, 1.0868173, -1.3593600, 1.3559136
4: -0.2520238, 0.9301934, -0.2543875, 0.9355542, -1.1875780, 1.1845809

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6639844, upper bound: 0.6639844
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6639844, upper bound: 0.6639844
time: 0.31 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0000615, 0.9523284, -0.0124662, 0.7046853, -0.7046238, 0.9647946
1: -0.1095166, 1.1974459, -0.1260490, 0.8739170, -0.9834336, 1.3234949
2: -0.0606787, 1.0631847, -0.0592167, 0.8673251, -0.9280038, 1.1224015
3: -0.2725427, 1.0815248, -0.2830637, 0.8561093, -1.1286520, 1.3645885
4: -0.2520238, 0.9301934, -0.2274244, 0.8807485, -1.1327723, 1.1576178

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6609718, upper bound: 0.6438791
time: 0.30 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6609718, upper bound: 0.6438791
time: 0.28 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0054424, 0.6499398, -0.0017378, 0.9577341, -0.9522917, 0.6516776
1: -0.1022949, 0.8109143, -0.1117451, 1.2035816, -1.3058765, 0.9226594
2: -0.0393627, 0.7986160, -0.0625789, 1.0695953, -1.1089580, 0.8611949
3: -0.2590857, 0.7998861, -0.2743888, 1.0868173, -1.3459029, 1.0742749
4: -0.1939640, 0.8231770, -0.2543875, 0.9355542, -1.1295183, 1.0775645

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6722866, upper bound: 0.6876349
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6722866, upper bound: 0.6881240
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0004525, 0.7221838, 0.0016496, 0.9380729, -0.9376204, 0.7205342
1: -0.1048932, 0.8939738, -0.1070411, 1.1816590, -1.2865522, 1.0010149
2: -0.0500441, 0.8598962, -0.0588033, 1.0435266, -1.0935707, 0.9186995
3: -0.2612641, 0.8577256, -0.2689416, 1.0690438, -1.3303078, 1.1266673
4: -0.2176540, 0.8560207, -0.2483846, 0.9189640, -1.1366179, 1.1044054

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6515660, upper bound: 0.6751746
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6438791, upper bound: 0.6607286
time: 0.37 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0054424, 0.6499398, -0.0124662, 0.7046853, -0.6992429, 0.6624060
1: -0.1022949, 0.8109143, -0.1260490, 0.8739170, -0.9762119, 0.9369633
2: -0.0393627, 0.7986160, -0.0592167, 0.8673251, -0.9066877, 0.8578327
3: -0.2590857, 0.7998861, -0.2830637, 0.8561093, -1.1151949, 1.0829498
4: -0.1939640, 0.8231770, -0.2274244, 0.8807485, -1.0747125, 1.0506014

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6717107, upper bound: 0.6584709
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6798252, upper bound: 0.6917454
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6798252, upper bound: 0.6917725
time: 0.35 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0004525, 0.7221838, -0.0050101, 0.6788989, -0.6784464, 0.7271940
1: -0.1048932, 0.8939738, -0.1146951, 0.8445600, -0.9494532, 1.0086689
2: -0.0500441, 0.8598962, -0.0525062, 0.8296809, -0.8797249, 0.9124024
3: -0.2612641, 0.8577256, -0.2720425, 0.8305112, -1.0917752, 1.1297681
4: -0.2176540, 0.8560207, -0.2159448, 0.8508037, -1.0684577, 1.0719655

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6515660, upper bound: 0.6746649
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6408665, upper bound: 0.6408665
time: 0.34 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.41 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.41
Output dim: 0, lower bound: -0.6639844, upper bound: 0.6639844
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.41
Output dim: 0, lower bound: -0.6639844, upper bound: 0.6639844
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 3.41
Output dim: 0, lower bound: -0.6609718, upper bound: 0.6438791
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 3.41
Output dim: 0, lower bound: -0.6609718, upper bound: 0.6438791
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.41
Output dim: 0, lower bound: -0.6722866, upper bound: 0.6876349
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.41
Output dim: 0, lower bound: -0.6722866, upper bound: 0.6881240
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.41
Output dim: 0, lower bound: -0.6515660, upper bound: 0.6751746
IS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 3.41
Output dim: 0, lower bound: -0.6438791, upper bound: 0.6607286
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.41
Output dim: 0, lower bound: -0.6798252, upper bound: 0.6917454
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.41
Output dim: 0, lower bound: -0.6798252, upper bound: 0.6917725
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.41
Output dim: 0, lower bound: -0.6515660, upper bound: 0.6746649
IS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 3.41
Output dim: 0, lower bound: -0.6408665, upper bound: 0.6408665

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0000615, 0.9523284, 0.0000615, 0.9523284, -0.9522669, 0.9522669
1: -0.1095166, 1.1974459, -0.1095166, 1.1974459, -1.3069625, 1.3069625
2: -0.0606787, 1.0631847, -0.0606787, 1.0631847, -1.1238635, 1.1238635
3: -0.2725427, 1.0815248, -0.2725427, 1.0815248, -1.3540676, 1.3540676
4: -0.2520238, 0.9301934, -0.2520238, 0.9301934, -1.1822172, 1.1822172

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6727555, upper bound: 0.5879348
time: 0.31 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6727555, upper bound: 0.6714281
time: 0.33 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0000615, 0.9523284, -0.0512598, 1.8787861, -1.8787246, 1.0035882
1: -0.1095166, 1.1974459, -0.1754494, 2.2964354, -2.4059520, 1.3728952
2: -0.0606787, 1.0631847, -0.1450286, 2.0645037, -2.1251824, 1.2082133
3: -0.2725427, 1.0815248, -0.3347113, 1.8700678, -2.1426105, 1.4162362
4: -0.2520238, 0.9301934, -0.4980223, 1.5064290, -1.7584528, 1.4282157

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6727555, upper bound: 0.5879348
time: 0.31 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6781872, upper bound: 0.6714281
time: 0.34 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0054424, 0.6499398, 0.0197756, 0.9062033, -0.9007609, 0.6301641
1: -0.1022949, 0.8109143, -0.0792882, 1.1450822, -1.2473772, 0.8902025
2: -0.0393627, 0.7986160, -0.0386198, 1.0002036, -1.0395663, 0.8372357
3: -0.2590857, 0.7998861, -0.2423158, 1.0302336, -1.2893193, 1.0422019
4: -0.1939640, 0.8231770, -0.2225227, 0.8589398, -1.0529038, 1.0456997

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5

Time for candidate selection: 1.75 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6775245, upper bound: 0.6502687
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6537549, upper bound: 0.6603057
time: 0.32 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0054424, 0.6499398, -0.0046289, 1.0400095, -1.0345671, 0.6545687
1: -0.1022949, 0.8109143, -0.1109180, 1.3149409, -1.4172359, 0.9218323
2: -0.0393627, 0.7986160, -0.0750914, 1.1419265, -1.1812892, 0.8737074
3: -0.2590857, 0.7998861, -0.2722509, 1.1791039, -1.4381895, 1.0721370
4: -0.1939640, 0.8231770, -0.2956257, 0.9621377, -1.1561017, 1.1188027

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5

Time for candidate selection: 1.74 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6775245, upper bound: 0.6502687
time: 0.33 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6537549, upper bound: 0.6656294
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0004525, 0.7221838, 0.0033271, 0.9327366, -0.9322841, 0.7188567
1: -0.1048932, 0.8939738, -0.1048460, 1.1755776, -1.2804708, 0.9988198
2: -0.0500441, 0.8598962, -0.0569408, 1.0371912, -1.0872352, 0.9168370
3: -0.2612641, 0.8577256, -0.2671270, 1.0637906, -1.3250546, 1.1248527
4: -0.2176540, 0.8560207, -0.2460732, 0.9137013, -1.1313553, 1.1020939

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6438791, upper bound: 0.6607286
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6438791, upper bound: 0.6607286
time: 0.35 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0054424, 0.6499398, 0.0054424, 0.6499398, -0.6444974, 0.6444974
1: -0.1022949, 0.8109143, -0.1022949, 0.8109143, -0.9132092, 0.9132092
2: -0.0393627, 0.7986160, -0.0393627, 0.7986160, -0.8379787, 0.8379787
3: -0.2590857, 0.7998861, -0.2590857, 0.7998861, -1.0589718, 1.0589718
4: -0.1939640, 0.8231770, -0.1939640, 0.8231770, -1.0171410, 1.0171410

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5

Time for candidate selection: 1.80 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6784436, upper bound: 0.6697954
time: 0.40 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6598466, upper bound: 0.6644795
time: 0.37 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0054424, 0.6499398, 0.0004525, 0.7221838, -0.7167414, 0.6494873
1: -0.1022949, 0.8109143, -0.1048932, 0.8939738, -0.9962687, 0.9158075
2: -0.0393627, 0.7986160, -0.0500441, 0.8598962, -0.8992589, 0.8486601
3: -0.2590857, 0.7998861, -0.2612641, 0.8577256, -1.1168113, 1.0611502
4: -0.1939640, 0.8231770, -0.2176540, 0.8560207, -1.0499847, 1.0408310

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5

Time for candidate selection: 1.75 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6784436, upper bound: 0.6697954
time: 0.36 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6598466, upper bound: 0.6644795
time: 0.35 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0004525, 0.7221838, -0.0032587, 0.6730454, -0.6725929, 0.7254425
1: -0.1048932, 0.8939738, -0.1126518, 0.8378215, -0.9427147, 1.0066257
2: -0.0500441, 0.8598962, -0.0505674, 0.8232774, -0.8733214, 0.9104636
3: -0.2612641, 0.8577256, -0.2704487, 0.8248124, -1.0860765, 1.1281743
4: -0.2176540, 0.8560207, -0.2129583, 0.8459927, -1.0636468, 1.0689790

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6408665, upper bound: 0.6408665
time: 0.34 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6408665, upper bound: 0.6408665
time: 0.33 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.13 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.13
Output dim: 0, lower bound: -0.6727555, upper bound: 0.5879348
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.13
Output dim: 0, lower bound: -0.6727555, upper bound: 0.6714281
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.13
Output dim: 0, lower bound: -0.6727555, upper bound: 0.5879348
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.13
Output dim: 0, lower bound: -0.6781872, upper bound: 0.6714281
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.13
Output dim: 0, lower bound: -0.6775245, upper bound: 0.6502687
IS_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.13
Output dim: 0, lower bound: -0.6537549, upper bound: 0.6603057
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.13
Output dim: 0, lower bound: -0.6775245, upper bound: 0.6502687
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.13
Output dim: 0, lower bound: -0.6537549, upper bound: 0.6656294
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.13
Output dim: 0, lower bound: -0.6438791, upper bound: 0.6607286
IS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.13
Output dim: 0, lower bound: -0.6438791, upper bound: 0.6607286
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.13
Output dim: 0, lower bound: -0.6784436, upper bound: 0.6697954
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.13
Output dim: 0, lower bound: -0.6598466, upper bound: 0.6644795
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.13
Output dim: 0, lower bound: -0.6784436, upper bound: 0.6697954
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.13
Output dim: 0, lower bound: -0.6598466, upper bound: 0.6644795
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.13
Output dim: 0, lower bound: -0.6408665, upper bound: 0.6408665
IS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.13
Output dim: 0, lower bound: -0.6408665, upper bound: 0.6408665

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0213993, 0.9013143, 0.0000615, 0.9523284, -0.9309292, 0.9012527
1: -0.0770538, 1.1394897, -0.1095166, 1.1974459, -1.2744997, 1.2490063
2: -0.0367112, 0.9943478, -0.0606787, 1.0631847, -1.0998960, 1.0550265
3: -0.2404699, 1.0253115, -0.2725427, 1.0815248, -1.3219948, 1.2978542
4: -0.2202464, 0.8539461, -0.2520238, 0.9301934, -1.1504399, 1.1059699

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6685265, upper bound: 0.6685265
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6685265, upper bound: 0.6685265
time: 0.32 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0028670, 1.0353181, 0.0033271, 0.9327366, -0.9356036, 1.0319910
1: -0.1088188, 1.3095722, -0.1048460, 1.1755776, -1.2843964, 1.4144182
2: -0.0732558, 1.1363764, -0.0569408, 1.0371912, -1.1104469, 1.1933172
3: -0.2704966, 1.1744311, -0.2671270, 1.0637906, -1.3342872, 1.4415581
4: -0.2937192, 0.9576517, -0.2460732, 0.9137013, -1.2074205, 1.2037250

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6685265, upper bound: 0.6807884
time: 0.31 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6685265, upper bound: 0.6858741
time: 0.31 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0213993, 0.9013143, -0.0512598, 1.8787861, -1.8573868, 0.9525740
1: -0.0770538, 1.1394897, -0.1754494, 2.2964354, -2.3734891, 1.3149390
2: -0.0367112, 0.9943478, -0.1450286, 2.0645037, -2.1012149, 1.1393764
3: -0.2404699, 1.0253115, -0.3347113, 1.8700678, -2.1105378, 1.3600228
4: -0.2202464, 0.8539461, -0.4980223, 1.5064290, -1.7266754, 1.3519684

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47

Time for candidate selection: 1.68 seconds

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6675757, upper bound: 0.5874962
time: 0.33 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 5

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 47

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5
type: B, layer: 5, pos: 1
type: B, layer: 5, pos: 8
type: B, layer: 5, pos: 22
type: B, layer: 5, pos: 27
type: B, layer: 5, pos: 40
type: B, layer: 5, pos: 21
type: B, layer: 5, pos: 25

Time for candidate selection: 4.59 seconds

### Candidate
type: B, layer: 5, pos: 1

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6658479, upper bound: 0.5700364
time: 0.30 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6716385, upper bound: 0.5878555
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0028670, 1.0353181, -0.0426288, 1.8511992, -1.8540661, 1.0779469
1: -0.1088188, 1.3095722, -0.1642268, 2.2651000, -2.3739188, 1.4737990
2: -0.0732558, 1.1363764, -0.1343973, 2.0305896, -2.1038454, 1.2707736
3: -0.2704966, 1.1744311, -0.3265591, 1.8406374, -2.1111341, 1.5009902
4: -0.2937192, 0.9576517, -0.4800941, 1.4741251, -1.7678443, 1.4377458

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6400006, upper bound: 0.6714281
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6441794, upper bound: 0.6534000
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, 0.0197756, 0.9062033, -0.8950152, 0.5450859
1: -0.0944340, 0.7130072, -0.0792882, 1.1450822, -1.2395163, 0.7922955
2: -0.0312903, 0.7124612, -0.0386198, 1.0002036, -1.0314939, 0.7510810
3: -0.2500148, 0.7301772, -0.2423158, 1.0302336, -1.2802484, 0.9724930
4: -0.1787794, 0.7720095, -0.2225227, 0.8589398, -1.0377191, 0.9945322

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6516224, upper bound: 0.6384161
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47

Time for candidate selection: 2.04 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6415780, upper bound: 0.6502687
time: 0.32 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6415780, upper bound: 0.6502687
time: 0.32 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, -0.0046289, 1.0400095, -1.0288215, 0.5694904
1: -0.0944340, 0.7130072, -0.1109180, 1.3149409, -1.4093750, 0.8239253
2: -0.0312903, 0.7124612, -0.0750914, 1.1419265, -1.1732168, 0.7875526
3: -0.2500148, 0.7301772, -0.2722509, 1.1791039, -1.4291186, 1.0024281
4: -0.1787794, 0.7720095, -0.2956257, 0.9621377, -1.1409171, 1.0676352

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 5

Time for candidate selection: 1.78 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6423504, upper bound: 0.6573603
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6423504, upper bound: 0.6573605
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, -0.0046289, 1.0400095, -1.0705764, 0.6752871
1: -0.1440713, 0.8347254, -0.1109180, 1.3149409, -1.4590123, 0.9456434
2: -0.0742025, 0.8280591, -0.0750914, 1.1419265, -1.2161291, 0.9031505
3: -0.2901301, 0.8312570, -0.2722509, 1.1791039, -1.4692340, 1.1035080
4: -0.2451730, 0.8507983, -0.2956257, 0.9621377, -1.2073107, 1.1464241

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 5

Time for candidate selection: 1.78 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6423504, upper bound: 0.6656294
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6423504, upper bound: 0.6656294
time: 0.34 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0158405, 0.4982040, 0.0054424, 0.6499398, -0.6340992, 0.4927616
1: -0.0890739, 0.6250523, -0.1022949, 0.8109143, -0.8999882, 0.7273472
2: -0.0220215, 0.6451663, -0.0393627, 0.7986160, -0.8206375, 0.6845290
3: -0.2441075, 0.6654664, -0.2590857, 0.7998861, -1.0439936, 0.9245521
4: -0.1598573, 0.7338556, -0.1939640, 0.8231770, -0.9830343, 0.9278196

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6812918, upper bound: 0.6740107
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6850497, upper bound: 0.6763963
time: 0.37 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0892951, 0.7884323, 0.0054424, 0.6499398, -0.7392349, 0.7829899
1: -0.2055373, 0.9655911, -0.1022949, 0.8109143, -1.0164516, 1.0678861
2: -0.1576910, 0.9697834, -0.0393627, 0.7986160, -0.9563070, 1.0091461
3: -0.3306990, 0.9586977, -0.2590857, 0.7998861, -1.1305851, 1.2177833
4: -0.3441186, 0.9710234, -0.1939640, 0.8231770, -1.1672956, 1.1649873

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6668777, upper bound: 0.6718276
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6706520, upper bound: 0.6761571
time: 0.35 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0158405, 0.4982040, 0.0004525, 0.7221838, -0.7063433, 0.4977515
1: -0.0890739, 0.6250523, -0.1048932, 0.8939738, -0.9830477, 0.7299455
2: -0.0220215, 0.6451663, -0.0500441, 0.8598962, -0.8819177, 0.6952104
3: -0.2441075, 0.6654664, -0.2612641, 0.8577256, -1.1018331, 0.9267305
4: -0.1598573, 0.7338556, -0.2176540, 0.8560207, -1.0158780, 0.9515096

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6652464, upper bound: 0.6475457
time: 0.40 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47

Time for candidate selection: 2.21 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6458655, upper bound: 0.6691407
time: 0.34 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6553430, upper bound: 0.6508301
time: 0.40 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0892951, 0.7884323, 0.0004525, 0.7221838, -0.8114790, 0.7879798
1: -0.2055373, 0.9655911, -0.1048932, 0.8939738, -1.0995111, 1.0704844
2: -0.1576910, 0.9697834, -0.0500441, 0.8598962, -1.0175872, 1.0198275
3: -0.3306990, 0.9586977, -0.2612641, 0.8577256, -1.1884246, 1.2199618
4: -0.3441186, 0.9710234, -0.2176540, 0.8560207, -1.2001393, 1.1886773

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6589704, upper bound: 0.6483851
time: 0.33 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47

Time for candidate selection: 2.16 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6309964, upper bound: 0.6644795
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6405375, upper bound: 0.6466901
time: 0.37 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.65 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 0, lower bound: -0.6685265, upper bound: 0.6685265
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 0, lower bound: -0.6685265, upper bound: 0.6685265
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 0, lower bound: -0.6685265, upper bound: 0.6807884
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 0, lower bound: -0.6685265, upper bound: 0.6858741
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 0, lower bound: -0.6658479, upper bound: 0.5700364
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 0, lower bound: -0.6716385, upper bound: 0.5878555
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 0, lower bound: -0.6400006, upper bound: 0.6714281
IS_A1_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.65
Output dim: 0, lower bound: -0.6441794, upper bound: 0.6534000
IS_A2_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.65
Output dim: 0, lower bound: -0.6415780, upper bound: 0.6502687
IS_A2_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.65
Output dim: 0, lower bound: -0.6415780, upper bound: 0.6502687
IS_A2_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.65
Output dim: 0, lower bound: -0.6423504, upper bound: 0.6573603
IS_A2_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.65
Output dim: 0, lower bound: -0.6423504, upper bound: 0.6573605
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 0, lower bound: -0.6423504, upper bound: 0.6656294
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 0, lower bound: -0.6423504, upper bound: 0.6656294
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 0, lower bound: -0.6812918, upper bound: 0.6740107
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 0, lower bound: -0.6850497, upper bound: 0.6763963
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 0, lower bound: -0.6668777, upper bound: 0.6718276
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 0, lower bound: -0.6706520, upper bound: 0.6761571
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 0, lower bound: -0.6458655, upper bound: 0.6691407
IS_A2_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.65
Output dim: 0, lower bound: -0.6553430, upper bound: 0.6508301
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 0, lower bound: -0.6309964, upper bound: 0.6644795
IS_A2_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.65
Output dim: 0, lower bound: -0.6405375, upper bound: 0.6466901

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0213993, 0.9013143, 0.0213993, 0.9013143, -0.8799150, 0.8799150
1: -0.0770538, 1.1394897, -0.0770538, 1.1394897, -1.2165434, 1.2165434
2: -0.0367112, 0.9943478, -0.0367112, 0.9943478, -1.0310590, 1.0310590
3: -0.2404699, 1.0253115, -0.2404699, 1.0253115, -1.2657814, 1.2657814
4: -0.2202464, 0.8539461, -0.2202464, 0.8539461, -1.0741925, 1.0741925

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 47

Time for candidate selection: 1.34 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6719145, upper bound: 0.6341388
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6461180, upper bound: 0.6461180
time: 0.32 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0213993, 0.9013143, -0.0028670, 1.0353181, -1.0139189, 0.9041812
1: -0.0770538, 1.1394897, -0.1088188, 1.3095722, -1.3866260, 1.2483084
2: -0.0367112, 0.9943478, -0.0732558, 1.1363764, -1.1730876, 1.0676036
3: -0.2404699, 1.0253115, -0.2704966, 1.1744311, -1.4149010, 1.2958081
4: -0.2202464, 0.8539461, -0.2937192, 0.9576517, -1.1778982, 1.1476653

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 47

Time for candidate selection: 1.35 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6719145, upper bound: 0.6408700
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6461180, upper bound: 0.6500684
time: 0.31 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0028670, 1.0353181, 0.0213993, 0.9013143, -0.9041812, 1.0139189
1: -0.1088188, 1.3095722, -0.0770538, 1.1394897, -1.2483084, 1.3866260
2: -0.0732558, 1.1363764, -0.0367112, 0.9943478, -1.0676036, 1.1730876
3: -0.2704966, 1.1744311, -0.2404699, 1.0253115, -1.2958081, 1.4149010
4: -0.2937192, 0.9576517, -0.2202464, 0.8539461, -1.1476653, 1.1778982

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5

Time for candidate selection: 1.38 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6677960, upper bound: 0.6354123
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6461180, upper bound: 0.6507545
time: 0.36 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0028670, 1.0353181, -0.0025742, 1.0353181, -1.0381851, 1.0378923
1: -0.1088188, 1.3095722, -0.1087742, 1.3095722, -1.4183910, 1.4183464
2: -0.0732558, 1.1363764, -0.0731823, 1.1363764, -1.2096322, 1.2095587
3: -0.2704966, 1.1744311, -0.2704394, 1.1744311, -1.4449277, 1.4448705
4: -0.2937192, 0.9576517, -0.2936380, 0.9576517, -1.2513709, 1.2512897

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5

Time for candidate selection: 1.34 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6677960, upper bound: 0.6468781
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6461180, upper bound: 0.6461180
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0213993, 0.9013143, -0.0214136, 1.8105392, -1.7891400, 0.9227278
1: -0.0770538, 1.1394897, -0.1446762, 2.2199259, -2.2969797, 1.2841659
2: -0.0367112, 0.9943478, -0.1009064, 1.9879384, -2.0246496, 1.0952542
3: -0.2404699, 1.0253115, -0.3037958, 1.7980361, -2.0385060, 1.3291073
4: -0.2202464, 0.8539461, -0.4475898, 1.4428692, -1.6631156, 1.3015358

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6658396, upper bound: 0.5697126
time: 0.30 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 47

Time for candidate selection: 1.70 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6657021, upper bound: 0.5659192
time: 0.33 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6528675, upper bound: 0.5508613
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6658479, upper bound: 0.5700364
time: 0.33 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0213993, 0.9013143, -0.0401359, 1.8319054, -1.8105061, 0.9414501
1: -0.0770538, 1.1394897, -0.1517431, 2.1914821, -2.2685359, 1.2912327
2: -0.0367112, 0.9943478, -0.1223392, 2.0233612, -2.0600724, 1.1166871
3: -0.2404699, 1.0253115, -0.2837400, 1.7822971, -2.0227671, 1.3090515
4: -0.2202464, 0.8539461, -0.4533617, 1.5378227, -1.7580692, 1.3073078

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6716302, upper bound: 0.5845623
time: 0.31 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 47

Time for candidate selection: 1.69 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6706841, upper bound: 0.5839400
time: 0.33 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6583286, upper bound: 0.5852470
time: 0.33 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6714575, upper bound: 0.5878555
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0028670, 1.0353181, -0.0415804, 1.8474441, -1.8503110, 1.0768986
1: -0.1088188, 1.3095722, -0.1629195, 2.2608070, -2.3696258, 1.4724917
2: -0.0732558, 1.1363764, -0.1332316, 2.0263104, -2.0995662, 1.2696080
3: -0.2704966, 1.1744311, -0.3252528, 1.8368683, -2.1073649, 1.4996839
4: -0.2937192, 0.9576517, -0.4785490, 1.4706423, -1.7643615, 1.4362007

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6400006, upper bound: 0.6526529
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6400006, upper bound: 0.6534000
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, 0.0006924, 0.9639854, -0.9945524, 0.6699659
1: -0.1440713, 0.8347254, -0.1038828, 1.2270458, -1.3711171, 0.9386082
2: -0.0742025, 0.8280591, -0.0676236, 1.0606856, -1.1348882, 0.8956828
3: -0.2901301, 0.8312570, -0.2647138, 1.1143060, -1.4044361, 1.0959709
4: -0.2451730, 0.8507983, -0.2805611, 0.9134201, -1.1585931, 1.1313593

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5

Time for candidate selection: 1.88 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6423504, upper bound: 0.6573603
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6423504, upper bound: 0.6656294
time: 0.33 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, -0.0360885, 1.0472777, -1.0778446, 0.7067467
1: -0.1440713, 0.8347254, -0.1492593, 1.3218184, -1.4658897, 0.9839847
2: -0.0742025, 0.8280591, -0.1066136, 1.1590223, -1.2332249, 0.9346728
3: -0.2901301, 0.8312570, -0.3045232, 1.1983278, -1.4884579, 1.1357803
4: -0.2451730, 0.8507983, -0.3275616, 0.9970202, -1.2421932, 1.1783600

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5

Time for candidate selection: 1.94 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6577101, upper bound: 0.6573603
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6577102, upper bound: 0.6656292
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0175722, 0.4906744, 0.0073309, 0.6935004, -0.6759282, 0.4833435
1: -0.0868347, 0.6162290, -0.0999169, 0.8545845, -0.9414191, 0.7161459
2: -0.0197408, 0.6367096, -0.0374527, 0.8373916, -0.8571324, 0.6741623
3: -0.2417490, 0.6583002, -0.2552719, 0.8256997, -1.0674489, 0.9135721
4: -0.1565087, 0.7273067, -0.1935482, 0.8381348, -0.9946435, 0.9208549

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6812918, upper bound: 0.6739252
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6812918, upper bound: 0.6740107
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0228415, 0.4731701, 0.0294173, 0.5563495, -0.5335081, 0.4437528
1: -0.0810974, 0.5906536, -0.0742822, 0.6929038, -0.7740012, 0.6649358
2: -0.0101569, 0.6171860, 0.0004599, 0.7029566, -0.7131134, 0.6167261
3: -0.2359192, 0.6358365, -0.2323704, 0.6962168, -0.9321361, 0.8682069
4: -0.1423800, 0.7138430, -0.1339579, 0.7538315, -0.8962115, 0.8478009

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6849585, upper bound: 0.6751849
time: 0.34 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6849586, upper bound: 0.6763963
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0872858, 0.7793844, 0.0073309, 0.6935004, -0.7807862, 0.7720535
1: -0.2031336, 0.9545649, -0.0999169, 0.8545845, -1.0577180, 1.0544817
2: -0.1549540, 0.9597380, -0.0374527, 0.8373916, -0.9923456, 0.9971907
3: -0.3282795, 0.9494417, -0.2552719, 0.8256997, -1.1539793, 1.2047136
4: -0.3399453, 0.9635973, -0.1935482, 0.8381348, -1.1780801, 1.1571455

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6586301, upper bound: 0.6701873
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6586301, upper bound: 0.6718277
time: 0.38 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0807047, 0.7613738, 0.0294173, 0.5563495, -0.6370542, 0.7319565
1: -0.1971450, 0.9301616, -0.0742822, 0.6929038, -0.8900488, 1.0044438
2: -0.1462803, 0.9406436, 0.0004599, 0.7029566, -0.8492368, 0.9401837
3: -0.3223667, 0.9271001, -0.2323704, 0.6962168, -1.0185835, 1.1594704
4: -0.3254409, 0.9495716, -0.1339579, 0.7538315, -1.0792724, 1.0835295

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6586301, upper bound: 0.6709031
time: 0.36 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6614808, upper bound: 0.6581234
time: 0.34 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6599186, upper bound: 0.6596040
time: 0.34 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0158405, 0.4982040, 0.0061345, 0.6507710, -0.6349305, 0.4920695
1: -0.0890739, 0.6250523, -0.0976210, 0.8106803, -0.8997542, 0.7226732
2: -0.0220215, 0.6451663, -0.0417109, 0.7880348, -0.8100563, 0.6868772
3: -0.2441075, 0.6654664, -0.2534082, 0.7965333, -1.0406408, 0.9188746
4: -0.1598573, 0.7338556, -0.2023642, 0.8128391, -0.9726964, 0.9362198

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5

Time for candidate selection: 1.91 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6458655, upper bound: 0.6691407
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6309964, upper bound: 0.6644795
time: 0.39 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0892951, 0.7884323, 0.0061345, 0.6507710, -0.7400662, 0.7822978
1: -0.2055373, 0.9655911, -0.0976210, 0.8106803, -1.0162176, 1.0632122
2: -0.1576910, 0.9697834, -0.0417109, 0.7880348, -0.9457258, 1.0114943
3: -0.3306990, 0.9586977, -0.2534082, 0.7965333, -1.1272323, 1.2121059
4: -0.3441186, 0.9710234, -0.2023642, 0.8128391, -1.1569576, 1.1733875

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5

Time for candidate selection: 1.91 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6309964, upper bound: 0.6371391
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6309964, upper bound: 0.6466901
time: 0.37 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.43 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -0.6719145, upper bound: 0.6341388
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 0, lower bound: -0.6461180, upper bound: 0.6461180
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -0.6719145, upper bound: 0.6408700
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 0, lower bound: -0.6461180, upper bound: 0.6500684
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -0.6677960, upper bound: 0.6354123
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 0, lower bound: -0.6461180, upper bound: 0.6507545
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -0.6677960, upper bound: 0.6468781
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 0, lower bound: -0.6461180, upper bound: 0.6461180
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 0, lower bound: -0.6528675, upper bound: 0.5508613
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -0.6658479, upper bound: 0.5700364
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 0, lower bound: -0.6583286, upper bound: 0.5852470
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -0.6714575, upper bound: 0.5878555
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 0, lower bound: -0.6400006, upper bound: 0.6526529
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 0, lower bound: -0.6400006, upper bound: 0.6534000
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 0, lower bound: -0.6423504, upper bound: 0.6573603
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -0.6423504, upper bound: 0.6656294
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 0, lower bound: -0.6577101, upper bound: 0.6573603
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -0.6577102, upper bound: 0.6656292
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -0.6812918, upper bound: 0.6739252
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -0.6812918, upper bound: 0.6740107
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -0.6849585, upper bound: 0.6751849
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -0.6849586, upper bound: 0.6763963
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -0.6586301, upper bound: 0.6701873
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -0.6586301, upper bound: 0.6718277
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 0, lower bound: -0.6614808, upper bound: 0.6581234
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 0, lower bound: -0.6599186, upper bound: 0.6596040
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -0.6458655, upper bound: 0.6691407
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.43
Output dim: 0, lower bound: -0.6309964, upper bound: 0.6644795
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 0, lower bound: -0.6309964, upper bound: 0.6371391
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.43
Output dim: 0, lower bound: -0.6309964, upper bound: 0.6466901

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0265114, 0.8204293, 0.0213993, 0.9013143, -0.8748028, 0.7990301
1: -0.0697420, 1.0438113, -0.0770538, 1.1394897, -1.2092316, 1.1208651
2: -0.0287337, 0.9096682, -0.0367112, 0.9943478, -1.0230815, 0.9463794
3: -0.2322266, 0.9569638, -0.2404699, 1.0253115, -1.2575381, 1.1974337
4: -0.2029902, 0.8125998, -0.2202464, 0.8539461, -1.0569363, 1.0328462

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47

Time for candidate selection: 1.40 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6341388, upper bound: 0.6341388
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6341388, upper bound: 0.6341388
time: 0.30 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0265114, 0.8204293, -0.0028670, 1.0353181, -1.0088067, 0.8232963
1: -0.0697420, 1.0438113, -0.1088188, 1.3095722, -1.3793142, 1.1526301
2: -0.0287337, 0.9096682, -0.0732558, 1.1363764, -1.1651101, 0.9829240
3: -0.2322266, 0.9569638, -0.2704966, 1.1744311, -1.4066577, 1.2274604
4: -0.2029902, 0.8125998, -0.2937192, 0.9576517, -1.1606419, 1.1063190

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 5

Time for candidate selection: 1.39 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6341388, upper bound: 0.6408698
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6341388, upper bound: 0.6341388
time: 0.33 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0022838, 0.9603593, 0.0213993, 0.9013143, -0.8990304, 0.9389601
1: -0.1017990, 1.2228236, -0.0770538, 1.1394897, -1.2412887, 1.2998774
2: -0.0658340, 1.0562885, -0.0367112, 0.9943478, -1.0601819, 1.0929997
3: -0.2629817, 1.1104673, -0.2404699, 1.0253115, -1.2882931, 1.3509372
4: -0.2787794, 0.9093835, -0.2202464, 0.8539461, -1.1327255, 1.1296300

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47

Time for candidate selection: 1.53 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6341388, upper bound: 0.6354121
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6341388, upper bound: 0.6354121
time: 0.36 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0022838, 0.9603593, -0.0025742, 1.0353181, -1.0330343, 0.9629335
1: -0.1017990, 1.2228236, -0.1087742, 1.3095722, -1.4113712, 1.3315978
2: -0.0658340, 1.0562885, -0.0731823, 1.1363764, -1.2022104, 1.1294708
3: -0.2629817, 1.1104673, -0.2704394, 1.1744311, -1.4374127, 1.3809067
4: -0.2787794, 0.9093835, -0.2936380, 0.9576517, -1.2364311, 1.2030215

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 5

Time for candidate selection: 1.45 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6474426, upper bound: 0.6468780
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6474426, upper bound: 0.6468780
time: 0.36 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0407588, 0.7573676, -0.0214136, 1.8105392, -1.7697804, 0.7787812
1: -0.0475807, 0.9557493, -0.1446762, 2.2199259, -2.2675066, 1.1004255
2: -0.0055110, 0.8499758, -0.1009064, 1.9879384, -1.9934494, 0.9508822
3: -0.2086923, 0.8823092, -0.3037958, 1.7980361, -2.0067284, 1.1861050
4: -0.1635908, 0.7613368, -0.4475898, 1.4428692, -1.6064600, 1.2089266

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47

Time for candidate selection: 1.81 seconds

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 5

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 47

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5
type: B, layer: 5, pos: 1
type: B, layer: 5, pos: 8
type: B, layer: 5, pos: 22
type: B, layer: 5, pos: 40
type: B, layer: 5, pos: 21
type: B, layer: 5, pos: 25

Time for candidate selection: 4.49 seconds

### Candidate
type: B, layer: 5, pos: 1

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 25

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 7
type: B, layer: 7, pos: 11
type: B, layer: 7, pos: 23
type: B, layer: 7, pos: 40
type: B, layer: 7, pos: 1
type: B, layer: 7, pos: 27
type: B, layer: 7, pos: 16
type: B, layer: 7, pos: 21
type: B, layer: 7, pos: 45
type: B, layer: 7, pos: 43
type: B, layer: 7, pos: 19

Time for candidate selection: 7.26 seconds

### Candidate
type: B, layer: 7, pos: 11

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6658479, upper bound: 0.5692785
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 1

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6457254, upper bound: 0.5393889
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 43

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 7, pos: 19

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 9
type: B, layer: 9, pos: 33
type: B, layer: 9, pos: 5
type: B, layer: 9, pos: 46
type: B, layer: 9, pos: 32
type: B, layer: 9, pos: 0
type: B, layer: 9, pos: 23
type: B, layer: 9, pos: 7
type: B, layer: 9, pos: 9
type: B, layer: 9, pos: 36
type: B, layer: 9, pos: 43
type: B, layer: 9, pos: 20
type: B, layer: 9, pos: 14
type: B, layer: 9, pos: 28
type: B, layer: 9, pos: 25
type: B, layer: 9, pos: 27
type: B, layer: 9, pos: 16
type: B, layer: 9, pos: 3
type: B, layer: 9, pos: 38
type: B, layer: 9, pos: 24
type: B, layer: 9, pos: 30
type: B, layer: 9, pos: 15
type: B, layer: 9, pos: 18
type: B, layer: 9, pos: 17

Time for candidate selection: 12.61 seconds

### Candidate
type: B, layer: 9, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6602154, upper bound: 0.5539915
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6478880, upper bound: 0.5574248
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0407588, 0.7573676, -0.0401359, 1.8319054, -1.7911465, 0.7975035
1: -0.0475807, 0.9557493, -0.1517431, 2.1914821, -2.2390628, 1.1074923
2: -0.0055110, 0.8499758, -0.1223392, 2.0233612, -2.0288723, 0.9723151
3: -0.2086923, 0.8823092, -0.2837400, 1.7822971, -1.9909894, 1.1660492
4: -0.1635908, 0.7613368, -0.4533617, 1.5378227, -1.7014135, 1.2146986

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 5

Time for candidate selection: 1.80 seconds

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 47

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6566896, upper bound: 0.5874231
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 5

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5
type: B, layer: 5, pos: 1
type: B, layer: 5, pos: 8
type: B, layer: 5, pos: 40
type: B, layer: 5, pos: 22
type: B, layer: 5, pos: 21
type: B, layer: 5, pos: 25

Time for candidate selection: 4.78 seconds

### Candidate
type: B, layer: 5, pos: 1

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6572607, upper bound: 0.5863376
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6652642, upper bound: 0.5862698
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, 0.0006924, 0.9639854, -0.9945524, 0.6699659
1: -0.1440713, 0.8347254, -0.1038828, 1.2270458, -1.3711171, 0.9386082
2: -0.0742025, 0.8280591, -0.0676236, 1.0606856, -1.1348882, 0.8956828
3: -0.2901301, 0.8312570, -0.2647138, 1.1143060, -1.4044361, 1.0959709
4: -0.2451730, 0.8507983, -0.2805611, 0.9134201, -1.1585931, 1.1313593

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47

Time for candidate selection: 1.88 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6423504, upper bound: 0.6873994
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6423504, upper bound: 0.6656294
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, -0.0360885, 1.0472777, -1.0778446, 0.7067467
1: -0.1440713, 0.8347254, -0.1492593, 1.3218184, -1.4658897, 0.9839847
2: -0.0742025, 0.8280591, -0.1066136, 1.1590223, -1.2332249, 0.9346728
3: -0.2901301, 0.8312570, -0.3045232, 1.1983278, -1.4884579, 1.1357803
4: -0.2451730, 0.8507983, -0.3275616, 0.9970202, -1.2421932, 1.1783600

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 5

Time for candidate selection: 1.90 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6423504, upper bound: 0.6656292
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6577101, upper bound: 0.6656292
time: 0.38 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0185366, 0.5137528, 0.0073309, 0.6935004, -0.6749638, 0.5064219
1: -0.0853710, 0.6410187, -0.0999169, 0.8545845, -0.9399555, 0.7409357
2: -0.0191109, 0.6585484, -0.0374527, 0.8373916, -0.8565025, 0.6960011
3: -0.2388380, 0.6765358, -0.2552719, 0.8256997, -1.0645378, 0.9318078
4: -0.1577795, 0.7368335, -0.1935482, 0.8381348, -0.9959143, 0.9303817

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 47

Time for candidate selection: 1.43 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6448742, upper bound: 0.6733739
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6593579, upper bound: 0.6572456
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0387027, 0.4375544, 0.0073309, 0.6935004, -0.6547977, 0.4302235
1: -0.0613070, 0.5365020, -0.0999169, 0.8545845, -0.9158914, 0.6364189
2: 0.0174038, 0.5755119, -0.0374527, 0.8373916, -0.8199878, 0.6129646
3: -0.2170205, 0.5777811, -0.2552719, 0.8256997, -1.0427203, 0.8330530
4: -0.1004417, 0.6784918, -0.1935482, 0.8381348, -0.9385765, 0.8720400

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 47

Time for candidate selection: 1.49 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6448742, upper bound: 0.6734988
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6593579, upper bound: 0.6574420
time: 0.34 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0188074, 0.5137528, 0.0294173, 0.5563495, -0.5375421, 0.4843355
1: -0.0853209, 0.6410187, -0.0742822, 0.6929038, -0.7782248, 0.7153009
2: -0.0190291, 0.6585484, 0.0004599, 0.7029566, -0.7219857, 0.6580884
3: -0.2387748, 0.6765358, -0.2323704, 0.6962168, -0.9349916, 0.9089062
4: -0.1576672, 0.7368335, -0.1339579, 0.7538315, -0.9114987, 0.8707913

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5

Time for candidate selection: 1.43 seconds

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6812918, upper bound: 0.6751848
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6812918, upper bound: 0.6751848
time: 0.37 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0387027, 0.4375544, 0.0294173, 0.5563495, -0.5176468, 0.4081371
1: -0.0613070, 0.5365020, -0.0742822, 0.6929038, -0.7542108, 0.6107842
2: 0.0174038, 0.5755119, 0.0004599, 0.7029566, -0.6855527, 0.5750520
3: -0.2170205, 0.5777811, -0.2323704, 0.6962168, -0.9132373, 0.8101515
4: -0.1004417, 0.6784918, -0.1339579, 0.7538315, -0.8542732, 0.8124496

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5

Time for candidate selection: 1.46 seconds

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6812918, upper bound: 0.6763963
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6812918, upper bound: 0.6763963
time: 0.38 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0860746, 0.7981433, 0.0073328, 0.6934912, -0.7795658, 0.7908105
1: -0.2016160, 0.9745648, -0.0999153, 0.8545710, -1.0561870, 1.0744801
2: -0.1532123, 0.9791473, -0.0374498, 0.8373810, -0.9905933, 1.0165970
3: -0.3256128, 0.9652334, -0.2552688, 0.8256882, -1.1513010, 1.2205023
4: -0.3422897, 0.9691569, -0.1935434, 0.8381268, -1.1804165, 1.1627004

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 47

Time for candidate selection: 1.47 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6356293, upper bound: 0.6698522
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6451673, upper bound: 0.6534247
time: 0.37 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0603211, 0.6870018, 0.0073309, 0.6935004, -0.7538215, 0.6796709
1: -0.1749001, 0.8297409, -0.0999169, 0.8545845, -1.0294845, 0.9296579
2: -0.1159739, 0.8579424, -0.0374527, 0.8373916, -0.9533656, 0.8953951
3: -0.3003569, 0.8422859, -0.2552719, 0.8256997, -1.1260567, 1.0975578
4: -0.2768354, 0.8908350, -0.1935482, 0.8381348, -1.1149702, 1.0843832

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 47

Time for candidate selection: 1.49 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6356293, upper bound: 0.6715074
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6451673, upper bound: 0.6554943
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0158405, 0.4982040, 0.0061345, 0.6507710, -0.6349305, 0.4920695
1: -0.0890739, 0.6250523, -0.0976210, 0.8106803, -0.8997542, 0.7226732
2: -0.0220215, 0.6451663, -0.0417109, 0.7880348, -0.8100563, 0.6868772
3: -0.2441075, 0.6654664, -0.2534082, 0.7965333, -1.0406408, 0.9188746
4: -0.1598573, 0.7338556, -0.2023642, 0.8128391, -0.9726964, 0.9362198

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6269174, upper bound: 0.6470828
time: 0.34 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47

Time for candidate selection: 2.29 seconds

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6309964, upper bound: 0.6644795
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6309964, upper bound: 0.6644795
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0892661, 0.7843834, 0.0061345, 0.6507710, -0.7400371, 0.7782488
1: -0.2051971, 0.9613039, -0.0976210, 0.8106803, -1.0158774, 1.0589249
2: -0.1573753, 0.9656872, -0.0417109, 0.7880348, -0.9454101, 1.0073980
3: -0.3306475, 0.9540346, -0.2534082, 0.7965333, -1.1271808, 1.2074428
4: -0.3440671, 0.9667487, -0.2023642, 0.8128391, -1.1569061, 1.1691129

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6269174, upper bound: 0.6470828
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47

Time for candidate selection: 2.36 seconds

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6309964, upper bound: 0.6644795
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6309964, upper bound: 0.6644795
time: 0.39 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 4.97 seconds
IS_A1_B1_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 4.97
Output dim: 0, lower bound: -0.6341388, upper bound: 0.6341388
IS_A1_B1_A1_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 4.97
Output dim: 0, lower bound: -0.6341388, upper bound: 0.6341388
IS_A1_B1_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 4.97
Output dim: 0, lower bound: -0.6341388, upper bound: 0.6408698
IS_A1_B1_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 4.97
Output dim: 0, lower bound: -0.6341388, upper bound: 0.6341388
IS_A1_B1_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 4.97
Output dim: 0, lower bound: -0.6341388, upper bound: 0.6354121
IS_A1_B1_A1_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 4.97
Output dim: 0, lower bound: -0.6341388, upper bound: 0.6354121
IS_A1_B1_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 4.97
Output dim: 0, lower bound: -0.6474426, upper bound: 0.6468780
IS_A1_B1_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 4.97
Output dim: 0, lower bound: -0.6474426, upper bound: 0.6468780
IS_A1_B1_A1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.97
Output dim: 0, lower bound: -0.6602154, upper bound: 0.5539915
IS_A1_B1_A1_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 4.97
Output dim: 0, lower bound: -0.6478880, upper bound: 0.5574248
IS_A1_B1_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.97
Output dim: 0, lower bound: -0.6572607, upper bound: 0.5863376
IS_A1_B1_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.97
Output dim: 0, lower bound: -0.6652642, upper bound: 0.5862698
IS_A2_B1_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.97
Output dim: 0, lower bound: -0.6423504, upper bound: 0.6873994
IS_A2_B1_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.97
Output dim: 0, lower bound: -0.6423504, upper bound: 0.6656294
IS_A2_B1_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.97
Output dim: 0, lower bound: -0.6423504, upper bound: 0.6656292
IS_A2_B1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.97
Output dim: 0, lower bound: -0.6577101, upper bound: 0.6656292
IS_A2_B2_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.97
Output dim: 0, lower bound: -0.6448742, upper bound: 0.6733739
IS_A2_B2_A1_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 4.97
Output dim: 0, lower bound: -0.6593579, upper bound: 0.6572456
IS_A2_B2_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.97
Output dim: 0, lower bound: -0.6448742, upper bound: 0.6734988
IS_A2_B2_A1_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 4.97
Output dim: 0, lower bound: -0.6593579, upper bound: 0.6574420
IS_A2_B2_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.97
Output dim: 0, lower bound: -0.6812918, upper bound: 0.6751848
IS_A2_B2_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.97
Output dim: 0, lower bound: -0.6812918, upper bound: 0.6751848
IS_A2_B2_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.97
Output dim: 0, lower bound: -0.6812918, upper bound: 0.6763963
IS_A2_B2_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.97
Output dim: 0, lower bound: -0.6812918, upper bound: 0.6763963
IS_A2_B2_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.97
Output dim: 0, lower bound: -0.6356293, upper bound: 0.6698522
IS_A2_B2_A1_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 4.97
Output dim: 0, lower bound: -0.6451673, upper bound: 0.6534247
IS_A2_B2_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.97
Output dim: 0, lower bound: -0.6356293, upper bound: 0.6715074
IS_A2_B2_A1_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 4.97
Output dim: 0, lower bound: -0.6451673, upper bound: 0.6554943
IS_A2_B2_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.97
Output dim: 0, lower bound: -0.6309964, upper bound: 0.6644795
IS_A2_B2_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.97
Output dim: 0, lower bound: -0.6309964, upper bound: 0.6644795
IS_A2_B2_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.97
Output dim: 0, lower bound: -0.6309964, upper bound: 0.6644795
IS_A2_B2_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.97
Output dim: 0, lower bound: -0.6309964, upper bound: 0.6644795

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0407588, 0.7573676, -0.0317793, 1.7519007, -1.7111418, 0.7891469
1: -0.0475807, 0.9557493, -0.1409086, 2.0930991, -2.1406798, 1.0966579
2: -0.0055110, 0.8499758, -0.1054173, 1.9403348, -1.9458458, 0.9553931
3: -0.2086923, 0.8823092, -0.2720395, 1.7009158, -1.9096081, 1.1543487
4: -0.1635908, 0.7613368, -0.4259328, 1.4818592, -1.6454500, 1.1872696

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6652642, upper bound: 0.5836699
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47

Time for candidate selection: 1.85 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6644078, upper bound: 0.5824353
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 47

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5
type: A, layer: 5, pos: 1
type: A, layer: 5, pos: 8
type: A, layer: 5, pos: 40
type: A, layer: 5, pos: 27
type: A, layer: 5, pos: 21
type: A, layer: 5, pos: 25

Time for candidate selection: 4.54 seconds

### Candidate
type: A, layer: 5, pos: 1

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6652642, upper bound: 0.5821520
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6652642, upper bound: 0.5862698
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, 0.0006924, 0.9639854, -0.9945524, 0.6699659
1: -0.1440713, 0.8347254, -0.1038828, 1.2270458, -1.3711171, 0.9386082
2: -0.0742025, 0.8280591, -0.0676236, 1.0606856, -1.1348882, 0.8956828
3: -0.2901301, 0.8312570, -0.2647138, 1.1143060, -1.4044361, 1.0959709
4: -0.2451730, 0.8507983, -0.2805611, 0.9134201, -1.1585931, 1.1313593

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5

Time for candidate selection: 2.03 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6423504, upper bound: 0.6573603
time: 0.45 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6423504, upper bound: 0.6656294
time: 0.34 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, -0.0360885, 1.0472777, -1.0778446, 0.7067467
1: -0.1440713, 0.8347254, -0.1492593, 1.3218184, -1.4658897, 0.9839847
2: -0.0742025, 0.8280591, -0.1066136, 1.1590223, -1.2332249, 0.9346728
3: -0.2901301, 0.8312570, -0.3045232, 1.1983278, -1.4884579, 1.1357803
4: -0.2451730, 0.8507983, -0.3275616, 0.9970202, -1.2421932, 1.1783600

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5

Time for candidate selection: 2.02 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6423504, upper bound: 0.6573603
time: 0.44 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6423504, upper bound: 0.6656294
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, 0.0006924, 0.9639854, -0.9945524, 0.6699659
1: -0.1440713, 0.8347254, -0.1038828, 1.2270458, -1.3711171, 0.9386082
2: -0.0742025, 0.8280591, -0.0676236, 1.0606856, -1.1348882, 0.8956828
3: -0.2901301, 0.8312570, -0.2647138, 1.1143060, -1.4044361, 1.0959709
4: -0.2451730, 0.8507983, -0.2805611, 0.9134201, -1.1585931, 1.1313593

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5

Time for candidate selection: 2.10 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6423504, upper bound: 0.6573603
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6423504, upper bound: 0.6656294
time: 0.34 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, -0.0360885, 1.0472777, -1.0778446, 0.7067467
1: -0.1440713, 0.8347254, -0.1492593, 1.3218184, -1.4658897, 0.9839847
2: -0.0742025, 0.8280591, -0.1066136, 1.1590223, -1.2332249, 0.9346728
3: -0.2901301, 0.8312570, -0.3045232, 1.1983278, -1.4884579, 1.1357803
4: -0.2451730, 0.8507983, -0.3275616, 0.9970202, -1.2421932, 1.1783600

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5

Time for candidate selection: 2.03 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6577101, upper bound: 0.6573603
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6577102, upper bound: 0.6656292
time: 0.40 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0185366, 0.5137528, 0.0136061, 0.5909360, -0.5723994, 0.5001467
1: -0.0853710, 0.6410187, -0.0912752, 0.7387895, -0.8241605, 0.7322940
2: -0.0191109, 0.6585484, -0.0286748, 0.7339988, -0.7531098, 0.6872232
3: -0.2388380, 0.6765358, -0.2452104, 0.7454442, -0.9842821, 0.9217463
4: -0.1577795, 0.7368335, -0.1770196, 0.7780294, -0.9358089, 0.9138530

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47

Time for candidate selection: 1.46 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6521280, upper bound: 0.6464990
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6521280, upper bound: 0.6572456
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0387027, 0.4375544, 0.0136061, 0.5909360, -0.5522333, 0.4239483
1: -0.0613070, 0.5365020, -0.0912752, 0.7387895, -0.8000965, 0.6277772
2: 0.0174038, 0.5755119, -0.0286748, 0.7339988, -0.7165950, 0.6041868
3: -0.2170205, 0.5777811, -0.2452104, 0.7454442, -0.9624647, 0.8229915
4: -0.1004417, 0.6784918, -0.1770196, 0.7780294, -0.8784711, 0.8555114

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5

Time for candidate selection: 1.50 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6448742, upper bound: 0.6734988
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6380948, upper bound: 0.6715074
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0188074, 0.5137528, 0.0387027, 0.4375544, -0.4187470, 0.4750501
1: -0.0853209, 0.6410187, -0.0613070, 0.5365020, -0.6218230, 0.7023257
2: -0.0190291, 0.6585484, 0.0174038, 0.5755119, -0.5945411, 0.6411445
3: -0.2387748, 0.6765358, -0.2170205, 0.5777811, -0.8165559, 0.8935564
4: -0.1576672, 0.7368335, -0.1004417, 0.6784918, -0.8361589, 0.8372751

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47

Time for candidate selection: 1.48 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6847764, upper bound: 0.6469858
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6647232, upper bound: 0.6580429
time: 0.39 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0188074, 0.5137528, -0.0538366, 0.6856053, -0.6667979, 0.5675894
1: -0.0853209, 0.6410187, -0.1734412, 0.8282951, -0.9136160, 0.8144599
2: -0.0190291, 0.6585484, -0.1139090, 0.8565001, -0.8755293, 0.7724574
3: -0.2387748, 0.6765358, -0.2983873, 0.8408561, -1.0796309, 0.9749231
4: -0.1576672, 0.7368335, -0.2739153, 0.8889887, -1.0466559, 1.0107487

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47

Time for candidate selection: 1.51 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6847764, upper bound: 0.6469858
time: 0.40 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6647232, upper bound: 0.6580429
time: 0.39 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 4.17 seconds
IS_A1_B1_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 4.17
Output dim: 0, lower bound: -0.6652642, upper bound: 0.5821520
IS_A1_B1_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.17
Output dim: 0, lower bound: -0.6652642, upper bound: 0.5862698
IS_A2_B1_A1_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.17
Output dim: 0, lower bound: -0.6423504, upper bound: 0.6573603
IS_A2_B1_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.17
Output dim: 0, lower bound: -0.6423504, upper bound: 0.6656294
IS_A2_B1_A1_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.17
Output dim: 0, lower bound: -0.6423504, upper bound: 0.6573603
IS_A2_B1_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.17
Output dim: 0, lower bound: -0.6423504, upper bound: 0.6656294
IS_A2_B1_A1_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.17
Output dim: 0, lower bound: -0.6423504, upper bound: 0.6573603
IS_A2_B1_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.17
Output dim: 0, lower bound: -0.6423504, upper bound: 0.6656294
IS_A2_B1_A1_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.17
Output dim: 0, lower bound: -0.6577101, upper bound: 0.6573603
IS_A2_B1_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.17
Output dim: 0, lower bound: -0.6577102, upper bound: 0.6656292
IS_A2_B2_A1_B1_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.17
Output dim: 0, lower bound: -0.6521280, upper bound: 0.6464990
IS_A2_B2_A1_B1_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 9, time: 4.17
Output dim: 0, lower bound: -0.6521280, upper bound: 0.6572456
IS_A2_B2_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 4.17
Output dim: 0, lower bound: -0.6448742, upper bound: 0.6734988
IS_A2_B2_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.17
Output dim: 0, lower bound: -0.6380948, upper bound: 0.6715074
IS_A2_B2_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 4.17
Output dim: 0, lower bound: -0.6847764, upper bound: 0.6469858
IS_A2_B2_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.17
Output dim: 0, lower bound: -0.6647232, upper bound: 0.6580429
IS_A2_B2_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 4.17
Output dim: 0, lower bound: -0.6847764, upper bound: 0.6469858
IS_A2_B2_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.17
Output dim: 0, lower bound: -0.6647232, upper bound: 0.6580429
IS_A2_B2_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.17
Output dim: 0, lower bound: -0.6812918, upper bound: 0.6763963
IS_A2_B2_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.17
Output dim: 0, lower bound: -0.6812918, upper bound: 0.6763963
IS_A2_B2_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.17
Output dim: 0, lower bound: -0.6356293, upper bound: 0.6698522
IS_A2_B2_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.17
Output dim: 0, lower bound: -0.6356293, upper bound: 0.6715074
IS_A2_B2_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.17
Output dim: 0, lower bound: -0.6309964, upper bound: 0.6644795
IS_A2_B2_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.17
Output dim: 0, lower bound: -0.6309964, upper bound: 0.6644795
IS_A2_B2_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.17
Output dim: 0, lower bound: -0.6309964, upper bound: 0.6644795
IS_A2_B2_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.17
Output dim: 0, lower bound: -0.6309964, upper bound: 0.6644795
Binary search (step 2): status=Status.UNKNOWN, low=0.0017312, high=0.0242421, mid=0.0242421, abs_max=0.7819017171859741
rel_dist={0: [-0.6965375425073153, 0.6965375425073155]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0017311790288658813
execution time: 1148.21 seconds
