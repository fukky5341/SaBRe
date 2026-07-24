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
execution time: IAR + LP analysis = 1.71 + 0.97 = 2.68 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.7126415, upper bound: 0.7126415


# Binary Search by BASE starts (time budget: 1197.32 seconds, max iter: 100)

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
Binary search time: 48.76 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0017311790288658813


# Individual Split (IS_dual) starts
Time budget: 1148.56 seconds

## Binary search (step 0) starts
Candidate diff: 0.0917747


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7096954, upper bound: 0.6862236
time: 0.31 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7092829, upper bound: 0.7092829
time: 0.30 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.81 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.81
Output dim: 0, lower bound: -0.7096954, upper bound: 0.6862236
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.81
Output dim: 0, lower bound: -0.7092829, upper bound: 0.7092829

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0017378, 0.9577341, -0.0318909, 0.7472411, -0.7489790, 0.9896250
1: -0.1117451, 1.2035816, -0.1545317, 0.9228499, -1.0345950, 1.3581133
2: -0.0625789, 1.0695953, -0.0736544, 0.9239269, -0.9865058, 1.1432498
3: -0.2743888, 1.0868173, -0.3136785, 0.9005898, -1.1749785, 1.4004958
4: -0.2543875, 0.9355542, -0.2507498, 0.9391413, -1.1935288, 1.1863041

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7083999, upper bound: 0.6851157
time: 0.32 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7072936, upper bound: 0.6855405
time: 0.32 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0124662, 0.7046853, -0.0323386, 0.7495631, -0.7620293, 0.7370239
1: -0.1260490, 0.8739170, -0.1552227, 0.9255341, -1.0515832, 1.0291396
2: -0.0592167, 0.8673251, -0.0741289, 0.9267865, -0.9860032, 0.9414539
3: -0.2830637, 0.8561093, -0.3143692, 0.9027434, -1.1858070, 1.1704785
4: -0.2274244, 0.8807485, -0.2515757, 0.9410419, -1.1684663, 1.1323242

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 8

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6862236, upper bound: 0.7092829
time: 0.33 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6862236, upper bound: 0.7092829
time: 0.30 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.67 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.67
Output dim: 0, lower bound: -0.7083999, upper bound: 0.6851157
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.67
Output dim: 0, lower bound: -0.7072936, upper bound: 0.6855405
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.67
Output dim: 0, lower bound: -0.6862236, upper bound: 0.7092829
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.67
Output dim: 0, lower bound: -0.6862236, upper bound: 0.7092829

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0017378, 0.9577341, -0.0114529, 0.6914618, -0.6931996, 0.9691870
1: -0.1117451, 1.2035816, -0.1282673, 0.8587967, -0.9705418, 1.3318489
2: -0.0625789, 1.0695953, -0.0526597, 0.8532972, -0.9158762, 1.1222551
3: -0.2743888, 1.0868173, -0.2873807, 0.8425156, -1.1169045, 1.3741980
4: -0.2543875, 0.9355542, -0.2159567, 0.8764606, -1.1308482, 1.1515110

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6788727, upper bound: 0.6810052
time: 0.32 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6788728, upper bound: 0.6851157
time: 0.32 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0011210, 0.9538052, -0.0200248, 0.7691019, -0.7702229, 0.9738300
1: -0.1108930, 1.1991937, -0.1372972, 0.9491606, -1.0600536, 1.3364909
2: -0.0618782, 1.0645330, -0.0651515, 0.9265456, -0.9884238, 1.1296844
3: -0.2734051, 1.0833064, -0.2961719, 0.9082789, -1.1816840, 1.3794783
4: -0.2532487, 0.9323727, -0.2432690, 0.9243544, -1.1776030, 1.1756418

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7071055, upper bound: 0.6513010
time: 0.31 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6745204, upper bound: 0.6520481
time: 0.32 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0124662, 0.7046853, -0.0017378, 0.9577341, -0.9702003, 0.7064232
1: -0.1260490, 0.8739170, -0.1117451, 1.2035816, -1.3296306, 0.9856621
2: -0.0592167, 0.8673251, -0.0625789, 1.0695953, -1.1288121, 0.9299040
3: -0.2830637, 0.8561093, -0.2743888, 1.0868173, -1.3698809, 1.1304981
4: -0.2274244, 0.8807485, -0.2543875, 0.9355542, -1.1629786, 1.1351360

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6851157, upper bound: 0.7078883
time: 0.33 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6855405, upper bound: 0.7067790
time: 0.36 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0124662, 0.7046853, -0.0124662, 0.7046853, -0.7171515, 0.7171515
1: -0.1260490, 0.8739170, -0.1260490, 0.8739170, -0.9999660, 0.9999660
2: -0.0592167, 0.8673251, -0.0592167, 0.8673251, -0.9265418, 0.9265418
3: -0.2830637, 0.8561093, -0.2830637, 0.8561093, -1.1391729, 1.1391729
4: -0.2274244, 0.8807485, -0.2274244, 0.8807485, -1.1081729, 1.1081729

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6769402, upper bound: 0.7092829
time: 0.32 seconds

## Relational analysis of IS_A2_B2_B2
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
time: 0.34 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.44 seconds
IS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 3.44
Output dim: 0, lower bound: -0.6788727, upper bound: 0.6810052
IS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 3.44
Output dim: 0, lower bound: -0.6788728, upper bound: 0.6851157
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.44
Output dim: 0, lower bound: -0.7071055, upper bound: 0.6513010
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.44
Output dim: 0, lower bound: -0.6745204, upper bound: 0.6520481
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.44
Output dim: 0, lower bound: -0.6851157, upper bound: 0.7078883
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.44
Output dim: 0, lower bound: -0.6855405, upper bound: 0.7067790
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.44
Output dim: 0, lower bound: -0.6851157, upper bound: 0.7078883
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.44
Output dim: 0, lower bound: -0.6855405, upper bound: 0.7067790

## BFS IS instance: IS_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0017378, 0.9577341, 0.0197756, 0.9062033, -0.9079411, 0.9379585
1: -0.1117451, 1.2035816, -0.0792882, 1.1450822, -1.2568274, 1.2828698
2: -0.0625789, 1.0695953, -0.0386198, 1.0002036, -1.0627825, 1.1082151
3: -0.2743888, 1.0868173, -0.2423158, 1.0302336, -1.3046224, 1.3291330
4: -0.2543875, 0.9355542, -0.2225227, 0.8589398, -1.1133273, 1.1580770

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6738227, upper bound: 0.6738227
time: 0.28 seconds

## Relational analysis of IS_A1_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6738227, upper bound: 0.6738227
time: 0.30 seconds

## BFS IS instance: IS_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0017378, 0.9577341, 0.0054424, 0.6499398, -0.6516776, 0.9522917
1: -0.1117451, 1.2035816, -0.1022949, 0.8109143, -0.9226594, 1.3058765
2: -0.0625789, 1.0695953, -0.0393627, 0.7986160, -0.8611949, 1.1089580
3: -0.2743888, 1.0868173, -0.2590857, 0.7998861, -1.0742749, 1.3459029
4: -0.2543875, 0.9355542, -0.1939640, 0.8231770, -1.0775645, 1.1295183

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_B2_B1

### Relational analysis result of IS_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6642457, upper bound: 0.6772571
time: 0.32 seconds

## Relational analysis of IS_A1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6738227, upper bound: 0.6779333
time: 0.31 seconds

## Relational analysis of IS_A1_B1_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6738227, upper bound: 0.6851157
time: 0.34 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0001714, 0.9490757, -0.0200248, 0.7691019, -0.7689304, 0.9691005
1: -0.1092672, 1.1938853, -0.1372972, 0.9491606, -1.0584278, 1.3311825
2: -0.0603507, 1.0589557, -0.0651515, 0.9265456, -0.9868963, 1.1241071
3: -0.2721436, 1.0785306, -0.2961719, 0.9082789, -1.1804225, 1.3747025
4: -0.2510053, 0.9276448, -0.2432690, 0.9243544, -1.1753597, 1.1709138

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6860459, upper bound: 0.6513010
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6860459, upper bound: 0.6513010
time: 0.34 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0002806, 0.9836771, -0.0198476, 0.7683322, -0.7680516, 1.0035248
1: -0.1087384, 1.2370522, -0.1370933, 0.9482654, -1.0570039, 1.3741455
2: -0.0640910, 1.0932167, -0.0649593, 0.9256678, -0.9897587, 1.1581759
3: -0.2707281, 1.1106994, -0.2959552, 0.9075292, -1.1782572, 1.4066546
4: -0.2621267, 0.9407284, -0.2429371, 0.9237003, -1.1858270, 1.1836655

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6534607, upper bound: 0.6520481
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6534607, upper bound: 0.6520481
time: 0.34 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0054424, 0.6499398, -0.0017378, 0.9577341, -0.9522917, 0.6516776
1: -0.1022949, 0.8109143, -0.1117451, 1.2035816, -1.3058765, 0.9226594
2: -0.0393627, 0.7986160, -0.0625789, 1.0695953, -1.1089580, 0.8611949
3: -0.2590857, 0.7998861, -0.2743888, 1.0868173, -1.3459029, 1.0742749
4: -0.1939640, 0.8231770, -0.2543875, 0.9355542, -1.1295183, 1.0775645

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6760281, upper bound: 0.7083998
time: 0.33 seconds

## Relational analysis of IS_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_A1

### Relational analysis result of IS_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6772571, upper bound: 0.6937729
time: 0.31 seconds

## Relational analysis of IS_A2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

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

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6513010, upper bound: 0.7071055
time: 0.32 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6520481, upper bound: 0.6745204
time: 0.31 seconds

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
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6926397, upper bound: 0.7078883
time: 0.33 seconds

## Relational analysis of IS_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_A1
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

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6833646, upper bound: 0.7063542
time: 0.31 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6833646, upper bound: 0.7067790
time: 0.33 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.27 seconds
IS_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 0, lower bound: -0.6738227, upper bound: 0.6738227
IS_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 0, lower bound: -0.6738227, upper bound: 0.6738227
IS_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 0, lower bound: -0.6738227, upper bound: 0.6779333
IS_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 0, lower bound: -0.6738227, upper bound: 0.6851157
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 0, lower bound: -0.6860459, upper bound: 0.6513010
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 0, lower bound: -0.6860459, upper bound: 0.6513010
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 3.27
Output dim: 0, lower bound: -0.6534607, upper bound: 0.6520481
IS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 3.27
Output dim: 0, lower bound: -0.6534607, upper bound: 0.6520481
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 0, lower bound: -0.6779333, upper bound: 0.7022436
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 0, lower bound: -0.6779333, upper bound: 0.7072936
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 0, lower bound: -0.6513010, upper bound: 0.7071055
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 0, lower bound: -0.6520481, upper bound: 0.6745204
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 0, lower bound: -0.6833646, upper bound: 0.7063542
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 0, lower bound: -0.6833646, upper bound: 0.7067790
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 0, lower bound: -0.6833646, upper bound: 0.7063542
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 0, lower bound: -0.6833646, upper bound: 0.7067790

## BFS IS instance: IS_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0197756, 0.9062033, 0.0197756, 0.9062033, -0.8864276, 0.8864276
1: -0.0792882, 1.1450822, -0.0792882, 1.1450822, -1.2243705, 1.2243705
2: -0.0386198, 1.0002036, -0.0386198, 1.0002036, -1.0388234, 1.0388234
3: -0.2423158, 1.0302336, -0.2423158, 1.0302336, -1.2725494, 1.2725494
4: -0.2225227, 0.8589398, -0.2225227, 0.8589398, -1.0814625, 1.0814625

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5

Time for candidate selection: 3.08 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_B1_A1_A1

### Relational analysis result of IS_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6735132, upper bound: 0.6341388
time: 0.33 seconds

## Relational analysis of IS_A1_B1_B1_A1_A2

### Relational analysis result of IS_A1_B1_B1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6461180, upper bound: 0.6461180
time: 0.29 seconds

## BFS IS instance: IS_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0046289, 1.0400095, 0.0197756, 0.9062033, -0.9108322, 1.0202339
1: -0.1109180, 1.3149409, -0.0792882, 1.1450822, -1.2560003, 1.3942292
2: -0.0750914, 1.1419265, -0.0386198, 1.0002036, -1.0752950, 1.1805463
3: -0.2722509, 1.1791039, -0.2423158, 1.0302336, -1.3024845, 1.4214196
4: -0.2956257, 0.9621377, -0.2225227, 0.8589398, -1.1545655, 1.1846604

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 5

Time for candidate selection: 3.15 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_B1_A2_A1

### Relational analysis result of IS_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6735132, upper bound: 0.6365467
time: 0.30 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2

### Relational analysis result of IS_A1_B1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6461180, upper bound: 0.6461180
time: 0.33 seconds

## BFS IS instance: IS_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0197756, 0.9062033, 0.0054424, 0.6499398, -0.6301641, 0.9007609
1: -0.0792882, 1.1450822, -0.1022949, 0.8109143, -0.8902025, 1.2473772
2: -0.0386198, 1.0002036, -0.0393627, 0.7986160, -0.8372357, 1.0395663
3: -0.2423158, 1.0302336, -0.2590857, 0.7998861, -1.0422019, 1.2893193
4: -0.2225227, 0.8589398, -0.1939640, 0.8231770, -1.0456997, 1.0529038

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_B2_A1_A1

### Relational analysis result of IS_A1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7033530, upper bound: 0.6734183
time: 0.32 seconds

## Relational analysis of IS_A1_B1_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 5

Time for candidate selection: 3.44 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_B2_A1_A1

### Relational analysis result of IS_A1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7029979, upper bound: 0.6420588
time: 0.35 seconds

## Relational analysis of IS_A1_B1_B2_A1_A2

### Relational analysis result of IS_A1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6756027, upper bound: 0.6540380
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0046289, 1.0400095, 0.0054424, 0.6499398, -0.6545687, 1.0345671
1: -0.1109180, 1.3149409, -0.1022949, 0.8109143, -0.9218323, 1.4172359
2: -0.0750914, 1.1419265, -0.0393627, 0.7986160, -0.8737074, 1.1812892
3: -0.2722509, 1.1791039, -0.2590857, 0.7998861, -1.0721370, 1.4381895
4: -0.2956257, 0.9621377, -0.1939640, 0.8231770, -1.1188027, 1.1561017

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_B2_A2_A1

### Relational analysis result of IS_A1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7033530, upper bound: 0.6760281
time: 0.33 seconds

## Relational analysis of IS_A1_B1_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 5

Time for candidate selection: 3.47 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_B2_A2_B1

### Relational analysis result of IS_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6663589, upper bound: 0.6776237
time: 0.32 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2

### Relational analysis result of IS_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6756027, upper bound: 0.6540380
time: 0.35 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0001714, 0.9490757, -0.0046289, 1.0400095, -1.0398381, 0.9537046
1: -0.1092672, 1.1938853, -0.1109180, 1.3149409, -1.4242082, 1.3048034
2: -0.0603507, 1.0589557, -0.0750914, 1.1419265, -1.2022772, 1.1340470
3: -0.2721436, 1.0785306, -0.2722509, 1.1791039, -1.4512475, 1.3507814
4: -0.2510053, 0.9276448, -0.2956257, 0.9621377, -1.2131430, 1.2232705

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6751655, upper bound: 0.6508189
time: 0.31 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6607286, upper bound: 0.6397003
time: 0.31 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0001714, 0.9490757, 0.0004525, 0.7221838, -0.7220124, 0.9486232
1: -0.1092672, 1.1938853, -0.1048932, 0.8939738, -1.0032411, 1.2987785
2: -0.0603507, 1.0589557, -0.0500441, 0.8598962, -0.9202468, 1.1089997
3: -0.2721436, 1.0785306, -0.2612641, 0.8577256, -1.1298692, 1.3397946
4: -0.2510053, 0.9276448, -0.2176540, 0.8560207, -1.1070261, 1.1452988

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B2_B1

### Relational analysis result of IS_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6714281, upper bound: 0.6399833
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2

### Relational analysis result of IS_A1_B2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6607286, upper bound: 0.6397003
time: 0.30 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0054424, 0.6499398, 0.0197756, 0.9062033, -0.9007609, 0.6301641
1: -0.1022949, 0.8109143, -0.0792882, 1.1450822, -1.2473772, 0.8902025
2: -0.0393627, 0.7986160, -0.0386198, 1.0002036, -1.0395663, 0.8372357
3: -0.2590857, 0.7998861, -0.2423158, 1.0302336, -1.2893193, 1.0422019
4: -0.1939640, 0.8231770, -0.2225227, 0.8589398, -1.0529038, 1.0456997

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_B1

### Relational analysis result of IS_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6734182, upper bound: 0.7033530
time: 0.33 seconds

## Relational analysis of IS_A2_B1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5

Time for candidate selection: 3.45 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_B1

### Relational analysis result of IS_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6420588, upper bound: 0.7029979
time: 0.32 seconds

## Relational analysis of IS_A2_B1_A1_B1_B2

### Relational analysis result of IS_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6540380, upper bound: 0.6756027
time: 0.34 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0054424, 0.6499398, -0.0046289, 1.0400095, -1.0345671, 0.6545687
1: -0.1022949, 0.8109143, -0.1109180, 1.3149409, -1.4172359, 0.9218323
2: -0.0393627, 0.7986160, -0.0750914, 1.1419265, -1.1812892, 0.8737074
3: -0.2590857, 0.7998861, -0.2722509, 1.1791039, -1.4381895, 1.0721370
4: -0.1939640, 0.8231770, -0.2956257, 0.9621377, -1.1561017, 1.1188027

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_B1

### Relational analysis result of IS_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6734182, upper bound: 0.7083998
time: 0.33 seconds

## Relational analysis of IS_A2_B1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 5

Time for candidate selection: 3.51 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6776237, upper bound: 0.6773665
time: 0.32 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6540380, upper bound: 0.6866103
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0004525, 0.7221838, 0.0001714, 0.9490757, -0.9486232, 0.7220124
1: -0.1048932, 0.8939738, -0.1092672, 1.1938853, -1.2987785, 1.0032411
2: -0.0500441, 0.8598962, -0.0603507, 1.0589557, -1.1089997, 0.9202468
3: -0.2612641, 0.8577256, -0.2721436, 1.0785306, -1.3397946, 1.1298692
4: -0.2176540, 0.8560207, -0.2510053, 0.9276448, -1.1452988, 1.1070261

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6399833, upper bound: 0.6924877
time: 0.33 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6397003, upper bound: 0.6607286
time: 0.33 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0006387, 0.7210748, 0.0002806, 0.9836771, -0.9830384, 0.7207942
1: -0.1046588, 0.8927095, -0.1087384, 1.2370522, -1.3417110, 1.0014479
2: -0.0498471, 0.8586218, -0.0640910, 1.0932167, -1.1430638, 0.9227128
3: -0.2610173, 0.8567186, -0.2707281, 1.1106994, -1.3717167, 1.1274467
4: -0.2173126, 0.8551075, -0.2621267, 0.9407284, -1.1580410, 1.1172342

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6441621, upper bound: 0.6744598
time: 0.32 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6438791, upper bound: 0.6427007
time: 0.34 seconds

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
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B1_B1

### Relational analysis result of IS_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6926397, upper bound: 0.7074296
time: 0.33 seconds

## Relational analysis of IS_A2_B2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5

Time for candidate selection: 3.61 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6937080, upper bound: 0.6732796
time: 0.36 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6727303, upper bound: 0.6825233
time: 0.34 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0054424, 0.6499398, 0.0004525, 0.7221838, -0.7167414, 0.6494873
1: -0.1022949, 0.8109143, -0.1048932, 0.8939738, -0.9962687, 0.9158075
2: -0.0393627, 0.7986160, -0.0500441, 0.8598962, -0.8992589, 0.8486601
3: -0.2590857, 0.7998861, -0.2612641, 0.8577256, -1.1168113, 1.0611502
4: -0.1939640, 0.8231770, -0.2176540, 0.8560207, -1.0499847, 1.0408310

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B2_B1

### Relational analysis result of IS_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6926397, upper bound: 0.7078883
time: 0.34 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5

Time for candidate selection: 3.69 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6937080, upper bound: 0.6764579
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6727303, upper bound: 0.6857017
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0004525, 0.7221838, 0.0054424, 0.6499398, -0.6494873, 0.7167414
1: -0.1048932, 0.8939738, -0.1022949, 0.8109143, -0.9158075, 0.9962687
2: -0.0500441, 0.8598962, -0.0393627, 0.7986160, -0.8486601, 0.8992589
3: -0.2612641, 0.8577256, -0.2590857, 0.7998861, -1.0611502, 1.1168113
4: -0.2176540, 0.8560207, -0.1939640, 0.8231770, -1.0408310, 1.0499847

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6482833, upper bound: 0.7061672
time: 0.36 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6443836, upper bound: 0.6701376
time: 0.31 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 5

Time for candidate selection: 3.97 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6529501, upper bound: 0.7057270
time: 0.34 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6615940, upper bound: 0.6811419
time: 0.35 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0004525, 0.7221838, 0.0004525, 0.7221838, -0.7217313, 0.7217313
1: -0.1048932, 0.8939738, -0.1048932, 0.8939738, -0.9988670, 0.9988670
2: -0.0500441, 0.8598962, -0.0500441, 0.8598962, -0.9099402, 0.9099402
3: -0.2612641, 0.8577256, -0.2612641, 0.8577256, -1.1189897, 1.1189897
4: -0.2176540, 0.8560207, -0.2176540, 0.8560207, -1.0736747, 1.0736747

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B2_B1

### Relational analysis result of IS_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6443836, upper bound: 0.6746649
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B2_B1

### Relational analysis result of IS_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6482833, upper bound: 0.7061672
time: 0.36 seconds

## Relational analysis of IS_A2_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5

Time for candidate selection: 4.09 seconds

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
- Time for IS candidates: 6.55 seconds
IS_A1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 6.55
Output dim: 0, lower bound: -0.6735132, upper bound: 0.6341388
IS_A1_B1_B1_A1_A2, status: Status.VERIFIED, split count: 5, time: 6.55
Output dim: 0, lower bound: -0.6461180, upper bound: 0.6461180
IS_A1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 6.55
Output dim: 0, lower bound: -0.6735132, upper bound: 0.6365467
IS_A1_B1_B1_A2_A2, status: Status.VERIFIED, split count: 5, time: 6.55
Output dim: 0, lower bound: -0.6461180, upper bound: 0.6461180
IS_A1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 6.55
Output dim: 0, lower bound: -0.7029979, upper bound: 0.6420588
IS_A1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 6.55
Output dim: 0, lower bound: -0.6756027, upper bound: 0.6540380
IS_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 6.55
Output dim: 0, lower bound: -0.6663589, upper bound: 0.6776237
IS_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 6.55
Output dim: 0, lower bound: -0.6756027, upper bound: 0.6540380
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.55
Output dim: 0, lower bound: -0.6751655, upper bound: 0.6508189
IS_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 6.55
Output dim: 0, lower bound: -0.6607286, upper bound: 0.6397003
IS_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 6.55
Output dim: 0, lower bound: -0.6714281, upper bound: 0.6399833
IS_A1_B2_A1_B2_B2, status: Status.VERIFIED, split count: 5, time: 6.55
Output dim: 0, lower bound: -0.6607286, upper bound: 0.6397003
IS_A2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 6.55
Output dim: 0, lower bound: -0.6420588, upper bound: 0.7029979
IS_A2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 6.55
Output dim: 0, lower bound: -0.6540380, upper bound: 0.6756027
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.55
Output dim: 0, lower bound: -0.6776237, upper bound: 0.6773665
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.55
Output dim: 0, lower bound: -0.6540380, upper bound: 0.6866103
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.55
Output dim: 0, lower bound: -0.6399833, upper bound: 0.6924877
IS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 6.55
Output dim: 0, lower bound: -0.6397003, upper bound: 0.6607286
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.55
Output dim: 0, lower bound: -0.6441621, upper bound: 0.6744598
IS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 6.55
Output dim: 0, lower bound: -0.6438791, upper bound: 0.6427007
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.55
Output dim: 0, lower bound: -0.6937080, upper bound: 0.6732796
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.55
Output dim: 0, lower bound: -0.6727303, upper bound: 0.6825233
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.55
Output dim: 0, lower bound: -0.6937080, upper bound: 0.6764579
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.55
Output dim: 0, lower bound: -0.6727303, upper bound: 0.6857017
IS_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 6.55
Output dim: 0, lower bound: -0.6529501, upper bound: 0.7057270
IS_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 6.55
Output dim: 0, lower bound: -0.6615940, upper bound: 0.6811419
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.55
Output dim: 0, lower bound: -0.6825935, upper bound: 0.6737599
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.55
Output dim: 0, lower bound: -0.6615940, upper bound: 0.6836975

## BFS IS instance: IS_A1_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0248351, 0.8239982, 0.0197756, 0.9062033, -0.8813682, 0.8042226
1: -0.0720310, 1.0479953, -0.0792882, 1.1450822, -1.2171133, 1.1272836
2: -0.0307086, 0.9141164, -0.0386198, 1.0002036, -1.0309122, 0.9527361
3: -0.2340453, 0.9604603, -0.2423158, 1.0302336, -1.2642789, 1.2027761
4: -0.2052969, 0.8165774, -0.2225227, 0.8589398, -1.0642366, 1.0391002

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_B1_A1_A1_B1

### Relational analysis result of IS_A1_B1_B1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6341388, upper bound: 0.6341388
time: 0.31 seconds

## Relational analysis of IS_A1_B1_B1_A1_A1_B2

### Relational analysis result of IS_A1_B1_B1_A1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6341388, upper bound: 0.6341388
time: 0.28 seconds

## BFS IS instance: IS_A1_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0006924, 0.9639854, 0.0197756, 0.9062033, -0.9055109, 0.9442098
1: -0.1038828, 1.2270458, -0.0792882, 1.1450822, -1.2489650, 1.3063340
2: -0.0676236, 1.0606856, -0.0386198, 1.0002036, -1.0678272, 1.0993054
3: -0.2647138, 1.1143060, -0.2423158, 1.0302336, -1.2949474, 1.3566217
4: -0.2805611, 0.9134201, -0.2225227, 0.8589398, -1.1395009, 1.1359428

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_B1_A2_A1_B1

### Relational analysis result of IS_A1_B1_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6451461, upper bound: 0.6365467
time: 0.34 seconds

## Relational analysis of IS_A1_B1_B1_A2_A1_B2

### Relational analysis result of IS_A1_B1_B1_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6451461, upper bound: 0.6365467
time: 0.35 seconds

## BFS IS instance: IS_A1_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: 0.0248351, 0.8239982, 0.0054424, 0.6499398, -0.6251047, 0.8185558
1: -0.0720310, 1.0479953, -0.1022949, 0.8109143, -0.8829453, 1.1502903
2: -0.0307086, 0.9141164, -0.0393627, 0.7986160, -0.8293245, 0.9534791
3: -0.2340453, 0.9604603, -0.2590857, 0.7998861, -1.0339314, 1.2195460
4: -0.2052969, 0.8165774, -0.1939640, 0.8231770, -1.0284739, 1.0105414

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_B2_A1_A1_B1

### Relational analysis result of IS_A1_B1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6663589, upper bound: 0.6420588
time: 0.34 seconds

## Relational analysis of IS_A1_B1_B2_A1_A1_B2

### Relational analysis result of IS_A1_B1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6663589, upper bound: 0.6420589
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0167825, 0.9250188, 0.0054424, 0.6499398, -0.6667223, 0.9195764
1: -0.1252913, 1.1663461, -0.1022949, 0.8109143, -0.9362056, 1.2686410
2: -0.0773900, 1.0296290, -0.0393627, 0.7986160, -0.8760059, 1.0689917
3: -0.2777247, 1.0633786, -0.2590857, 0.7998861, -1.0776109, 1.3224642
4: -0.2678475, 0.8997647, -0.1939640, 0.8231770, -1.0910245, 1.0937288

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_B2_A1_A2_B1

### Relational analysis result of IS_A1_B1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6663589, upper bound: 0.6540380
time: 0.34 seconds

## Relational analysis of IS_A1_B1_B2_A1_A2_B2

### Relational analysis result of IS_A1_B1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6663589, upper bound: 0.6540380
time: 0.36 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0046289, 1.0400095, 0.0111880, 0.5648615, -0.5694904, 1.0288215
1: -0.1109180, 1.3149409, -0.0944340, 0.7130072, -0.8239253, 1.4093750
2: -0.0750914, 1.1419265, -0.0312903, 0.7124612, -0.7875526, 1.1732168
3: -0.2722509, 1.1791039, -0.2500148, 0.7301772, -1.0024281, 1.4291186
4: -0.2956257, 0.9621377, -0.1787794, 0.7720095, -1.0676352, 1.1409171

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6773662, upper bound: 0.6444212
time: 0.33 seconds

## Relational analysis of IS_A1_B1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6773662, upper bound: 0.6608971
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0046289, 1.0400095, -0.0305669, 0.6706582, -0.6752871, 1.0705764
1: -0.1109180, 1.3149409, -0.1440713, 0.8347254, -0.9456434, 1.4590123
2: -0.0750914, 1.1419265, -0.0742025, 0.8280591, -0.9031505, 1.2161291
3: -0.2722509, 1.1791039, -0.2901301, 0.8312570, -1.1035080, 1.4692340
4: -0.2956257, 0.9621377, -0.2451730, 0.8507983, -1.1464241, 1.2073107

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6863411, upper bound: 0.6574645
time: 0.33 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6751765, upper bound: 0.6585654
time: 0.34 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0019321, 0.9436822, -0.0046289, 1.0400095, -1.0380774, 0.9483111
1: -0.1070485, 1.1877589, -0.1109180, 1.3149409, -1.4219894, 1.2986770
2: -0.0584593, 1.0525572, -0.0750914, 1.1419265, -1.2003858, 1.1276486
3: -0.2703054, 1.0732428, -0.2722509, 1.1791039, -1.4494092, 1.3454937
4: -0.2486541, 0.9223583, -0.2956257, 0.9621377, -1.2107918, 1.2179840

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6637412, upper bound: 0.6400006
time: 0.31 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6637412, upper bound: 0.6415345
time: 0.31 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0001714, 0.9490757, 0.0021317, 0.7169888, -0.7168174, 0.9469440
1: -0.1092672, 1.1938853, -0.1028535, 0.8879859, -0.9972532, 1.2967389
2: -0.0603507, 1.0589557, -0.0482047, 0.8538951, -0.9142457, 1.1071603
3: -0.2721436, 1.0785306, -0.2595775, 0.8524666, -1.1246102, 1.3381081
4: -0.2510053, 0.9276448, -0.2148404, 0.8513455, -1.1023508, 1.1424853

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6607286, upper bound: 0.6397003
time: 0.30 seconds

## Relational analysis of IS_A1_B2_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6607286, upper bound: 0.6397003
time: 0.32 seconds

## BFS IS instance: IS_A2_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0054424, 0.6499398, 0.0248351, 0.8239982, -0.8185558, 0.6251047
1: -0.1022949, 0.8109143, -0.0720310, 1.0479953, -1.1502903, 0.8829453
2: -0.0393627, 0.7986160, -0.0307086, 0.9141164, -0.9534791, 0.8293245
3: -0.2590857, 0.7998861, -0.2340453, 0.9604603, -1.2195460, 1.0339314
4: -0.1939640, 0.8231770, -0.2052969, 0.8165774, -1.0105414, 1.0284739

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6663589
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A1_B1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6756027
time: 0.33 seconds

## BFS IS instance: IS_A2_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0054424, 0.6499398, -0.0167825, 0.9250188, -0.9195764, 0.6667223
1: -0.1022949, 0.8109143, -0.1252913, 1.1663461, -1.2686410, 0.9362056
2: -0.0393627, 0.7986160, -0.0773900, 1.0296290, -1.0689917, 0.8760059
3: -0.2590857, 0.7998861, -0.2777247, 1.0633786, -1.3224642, 1.0776109
4: -0.1939640, 0.8231770, -0.2678475, 0.8997647, -1.0937288, 1.0910245

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6540380, upper bound: 0.6663589
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A1_B1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6540380, upper bound: 0.6756027
time: 0.34 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, -0.0046289, 1.0400095, -1.0288215, 0.5694904
1: -0.0944340, 0.7130072, -0.1109180, 1.3149409, -1.4093750, 0.8239253
2: -0.0312903, 0.7124612, -0.0750914, 1.1419265, -1.1732168, 0.7875526
3: -0.2500148, 0.7301772, -0.2722509, 1.1791039, -1.4291186, 1.0024281
4: -0.1787794, 0.7720095, -0.2956257, 0.9621377, -1.1409171, 1.0676352

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6808037, upper bound: 0.6770974
time: 0.30 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6819046, upper bound: 0.6659327
time: 0.33 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, -0.0046289, 1.0400095, -1.0705764, 0.6752871
1: -0.1440713, 0.8347254, -0.1109180, 1.3149409, -1.4590123, 0.9456434
2: -0.0742025, 0.8280591, -0.0750914, 1.1419265, -1.2161291, 0.9031505
3: -0.2901301, 0.8312570, -0.2722509, 1.1791039, -1.4692340, 1.1035080
4: -0.2451730, 0.8507983, -0.2956257, 0.9621377, -1.2073107, 1.1464241

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6574645, upper bound: 0.6863411
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6585654, upper bound: 0.6751765
time: 0.34 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0021317, 0.7169888, 0.0001714, 0.9490757, -0.9469440, 0.7168174
1: -0.1028535, 0.8879859, -0.1092672, 1.1938853, -1.2967389, 0.9972532
2: -0.0482047, 0.8538951, -0.0603507, 1.0589557, -1.1071603, 0.9142457
3: -0.2595775, 0.8524666, -0.2721436, 1.0785306, -1.3381081, 1.1246102
4: -0.2148404, 0.8513455, -0.2510053, 0.9276448, -1.1424853, 1.1023508

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6397003, upper bound: 0.6607286
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6397003, upper bound: 0.6607286
time: 0.32 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0023172, 0.7158774, 0.0002806, 0.9836771, -0.9813600, 0.7155968
1: -0.1026192, 0.8867176, -0.1087384, 1.2370522, -1.3396714, 0.9954560
2: -0.0480075, 0.8526177, -0.0640910, 1.0932167, -1.1412241, 0.9167087
3: -0.2593307, 0.8514552, -0.2707281, 1.1106994, -1.3700302, 1.1221833
4: -0.2145011, 0.8504300, -0.2621267, 0.9407284, -1.1552296, 1.1125567

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A2_B2_A1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6399833, upper bound: 0.6744598
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6399833, upper bound: 0.6744598
time: 0.33 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, 0.0054424, 0.6499398, -0.6387517, 0.5594192
1: -0.0944340, 0.7130072, -0.1022949, 0.8109143, -0.9053483, 0.8153021
2: -0.0312903, 0.7124612, -0.0393627, 0.7986160, -0.8299063, 0.7518239
3: -0.2500148, 0.7301772, -0.2590857, 0.7998861, -1.0499009, 0.9892629
4: -0.1787794, 0.7720095, -0.1939640, 0.8231770, -1.0019563, 0.9659735

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6654188, upper bound: 0.6742789
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6654188, upper bound: 0.6742789
time: 0.34 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, 0.0054424, 0.6499398, -0.6805067, 0.6652158
1: -0.1440713, 0.8347254, -0.1022949, 0.8109143, -0.9549856, 0.9370203
2: -0.0742025, 0.8280591, -0.0393627, 0.7986160, -0.8728185, 0.8674218
3: -0.2901301, 0.8312570, -0.2590857, 0.7998861, -1.0900162, 1.0903428
4: -0.2451730, 0.8507983, -0.1939640, 0.8231770, -1.0683500, 1.0447624

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6654188, upper bound: 0.6835228
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6654188, upper bound: 0.6835228
time: 0.34 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, 0.0004525, 0.7221838, -0.7109958, 0.5644090
1: -0.0944340, 0.7130072, -0.1048932, 0.8939738, -0.9884079, 0.8179004
2: -0.0312903, 0.7124612, -0.0500441, 0.8598962, -0.8911865, 0.7625053
3: -0.2500148, 0.7301772, -0.2612641, 0.8577256, -1.1077404, 0.9914413
4: -0.1787794, 0.7720095, -0.2176540, 0.8560207, -1.0348001, 0.9896635

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6654033, upper bound: 0.6764576
time: 0.36 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6654033, upper bound: 0.6764576
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, 0.0004525, 0.7221838, -0.7527508, 0.6702057
1: -0.1440713, 0.8347254, -0.1048932, 0.8939738, -1.0380452, 0.9396186
2: -0.0742025, 0.8280591, -0.0500441, 0.8598962, -0.9340987, 0.8781032
3: -0.2901301, 0.8312570, -0.2612641, 0.8577256, -1.1478558, 1.0925212
4: -0.2451730, 0.8507983, -0.2176540, 0.8560207, -1.1011937, 1.0684524

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6654033, upper bound: 0.6857014
time: 0.34 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6654033, upper bound: 0.6857014
time: 0.34 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0004525, 0.7221838, 0.0111880, 0.5648615, -0.5644090, 0.7109958
1: -0.1048932, 0.8939738, -0.0944340, 0.7130072, -0.8179004, 0.9884079
2: -0.0500441, 0.8598962, -0.0312903, 0.7124612, -0.7625053, 0.8911865
3: -0.2612641, 0.8577256, -0.2500148, 0.7301772, -0.9914413, 1.1077404
4: -0.2176540, 0.8560207, -0.1787794, 0.7720095, -0.9896635, 1.0348001

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A2_B1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6529501, upper bound: 0.6742691
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6529501, upper bound: 0.6821413
time: 0.38 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0004525, 0.7221838, -0.0305669, 0.6706582, -0.6702057, 0.7527508
1: -0.1048932, 0.8939738, -0.1440713, 0.8347254, -0.9396186, 1.0380452
2: -0.0500441, 0.8598962, -0.0742025, 0.8280591, -0.8781032, 0.9340987
3: -0.2612641, 0.8577256, -0.2901301, 0.8312570, -1.0925212, 1.1478558
4: -0.2176540, 0.8560207, -0.2451730, 0.8507983, -1.0684524, 1.1011937

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A2_B1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6615940, upper bound: 0.6742691
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6615940, upper bound: 0.6821412
time: 0.37 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0061345, 0.6507710, 0.0004525, 0.7221838, -0.7160493, 0.6503185
1: -0.0976210, 0.8106803, -0.1048932, 0.8939738, -0.9915948, 0.9155735
2: -0.0417109, 0.7880348, -0.0500441, 0.8598962, -0.9016070, 0.8380789
3: -0.2534082, 0.7965333, -0.2612641, 0.8577256, -1.1111338, 1.0577974
4: -0.2023642, 0.8128391, -0.2176540, 0.8560207, -1.0583849, 1.0304930

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6530561, upper bound: 0.6737598
time: 0.36 seconds

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

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6530565, upper bound: 0.6836972
time: 0.34 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6530565, upper bound: 0.6836972
time: 0.35 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.64 seconds
IS_A1_B1_B1_A1_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.6341388, upper bound: 0.6341388
IS_A1_B1_B1_A1_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.6341388, upper bound: 0.6341388
IS_A1_B1_B1_A2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.6451461, upper bound: 0.6365467
IS_A1_B1_B1_A2_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.6451461, upper bound: 0.6365467
IS_A1_B1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.6663589, upper bound: 0.6420588
IS_A1_B1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.6663589, upper bound: 0.6420589
IS_A1_B1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.6663589, upper bound: 0.6540380
IS_A1_B1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.6663589, upper bound: 0.6540380
IS_A1_B1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.6773662, upper bound: 0.6444212
IS_A1_B1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.6773662, upper bound: 0.6608971
IS_A1_B1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.6863411, upper bound: 0.6574645
IS_A1_B1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.6751765, upper bound: 0.6585654
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.6637412, upper bound: 0.6400006
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.6637412, upper bound: 0.6415345
IS_A1_B2_A1_B2_B1_A1, status: Status.VERIFIED, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.6607286, upper bound: 0.6397003
IS_A1_B2_A1_B2_B1_A2, status: Status.VERIFIED, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.6607286, upper bound: 0.6397003
IS_A2_B1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6663589
IS_A2_B1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6756027
IS_A2_B1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.6540380, upper bound: 0.6663589
IS_A2_B1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.6540380, upper bound: 0.6756027
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.6808037, upper bound: 0.6770974
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.6819046, upper bound: 0.6659327
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.6574645, upper bound: 0.6863411
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.6585654, upper bound: 0.6751765
IS_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.6397003, upper bound: 0.6607286
IS_A2_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.6397003, upper bound: 0.6607286
IS_A2_B1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.6399833, upper bound: 0.6744598
IS_A2_B1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.6399833, upper bound: 0.6744598
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.6654188, upper bound: 0.6742789
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.6654188, upper bound: 0.6742789
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.6654188, upper bound: 0.6835228
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.6654188, upper bound: 0.6835228
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.6654033, upper bound: 0.6764576
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.6654033, upper bound: 0.6764576
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.6654033, upper bound: 0.6857014
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.6654033, upper bound: 0.6857014
IS_A2_B2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.6529501, upper bound: 0.6742691
IS_A2_B2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.6529501, upper bound: 0.6821413
IS_A2_B2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.6615940, upper bound: 0.6742691
IS_A2_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.6615940, upper bound: 0.6821412
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.6530561, upper bound: 0.6737598
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.6530561, upper bound: 0.6737598
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.6530565, upper bound: 0.6836972
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.6530565, upper bound: 0.6836972

## BFS IS instance: IS_A1_B1_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0248351, 0.8239982, 0.0111880, 0.5648615, -0.5400264, 0.8128102
1: -0.0720310, 1.0479953, -0.0944340, 0.7130072, -0.7850382, 1.1424294
2: -0.0307086, 0.9141164, -0.0312903, 0.7124612, -0.7431698, 0.9454067
3: -0.2340453, 0.9604603, -0.2500148, 0.7301772, -0.9642225, 1.2104751
4: -0.2052969, 0.8165774, -0.1787794, 0.7720095, -0.9773064, 0.9953567

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_B2_A1_A1_B1_A1

### Relational analysis result of IS_A1_B1_B2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6937541, upper bound: 0.6420589
time: 0.34 seconds

## Relational analysis of IS_A1_B1_B2_A1_A1_B1_A2

### Relational analysis result of IS_A1_B1_B2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6663589, upper bound: 0.6420588
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0248351, 0.8239982, -0.0305669, 0.6706582, -0.6458231, 0.8545651
1: -0.0720310, 1.0479953, -0.1440713, 0.8347254, -0.9067564, 1.1920667
2: -0.0307086, 0.9141164, -0.0742025, 0.8280591, -0.8587677, 0.9883189
3: -0.2340453, 0.9604603, -0.2901301, 0.8312570, -1.0653024, 1.2505904
4: -0.2052969, 0.8165774, -0.2451730, 0.8507983, -1.0560951, 1.0617504

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_B2_A1_A1_B2_A1

### Relational analysis result of IS_A1_B1_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7029979, upper bound: 0.6420589
time: 0.35 seconds

## Relational analysis of IS_A1_B1_B2_A1_A1_B2_A2

### Relational analysis result of IS_A1_B1_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6756027, upper bound: 0.6420589
time: 0.33 seconds

## BFS IS instance: IS_A1_B1_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0167825, 0.9250188, 0.0111880, 0.5648615, -0.5816441, 0.9138308
1: -0.1252913, 1.1663461, -0.0944340, 0.7130072, -0.8382986, 1.2607801
2: -0.0773900, 1.0296290, -0.0312903, 0.7124612, -0.7898512, 1.0609193
3: -0.2777247, 1.0633786, -0.2500148, 0.7301772, -1.0079019, 1.3133934
4: -0.2678475, 0.8997647, -0.1787794, 0.7720095, -1.0398570, 1.0785441

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_B2_A1_A2_B1_A1

### Relational analysis result of IS_A1_B1_B2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6663589, upper bound: 0.6420589
time: 0.34 seconds

## Relational analysis of IS_A1_B1_B2_A1_A2_B1_A2

### Relational analysis result of IS_A1_B1_B2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6663589, upper bound: 0.6540380
time: 0.35 seconds

## BFS IS instance: IS_A1_B1_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0167825, 0.9250188, -0.0305669, 0.6706582, -0.6874408, 0.9555857
1: -0.1252913, 1.1663461, -0.1440713, 0.8347254, -0.9600167, 1.3104174
2: -0.0773900, 1.0296290, -0.0742025, 0.8280591, -0.9054491, 1.1038315
3: -0.2777247, 1.0633786, -0.2901301, 0.8312570, -1.1089818, 1.3535087
4: -0.2678475, 0.8997647, -0.2451730, 0.8507983, -1.1186459, 1.1449378

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_B2_A1_A2_B2_A1

### Relational analysis result of IS_A1_B1_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6756027, upper bound: 0.6420588
time: 0.33 seconds

## Relational analysis of IS_A1_B1_B2_A1_A2_B2_A2

### Relational analysis result of IS_A1_B1_B2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6756027, upper bound: 0.6540380
time: 0.36 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0006924, 0.9639854, 0.0111880, 0.5648615, -0.5641692, 0.9527974
1: -0.1038828, 1.2270458, -0.0944340, 0.7130072, -0.8168900, 1.3214798
2: -0.0676236, 1.0606856, -0.0312903, 0.7124612, -0.7800848, 1.0919759
3: -0.2647138, 1.1143060, -0.2500148, 0.7301772, -0.9948910, 1.3643208
4: -0.2805611, 0.9134201, -0.1787794, 0.7720095, -1.0525706, 1.0921994

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_B2_A2_B1_A1_A1

### Relational analysis result of IS_A1_B1_B2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6770974, upper bound: 0.6629717
time: 0.33 seconds

## Relational analysis of IS_A1_B1_B2_A2_B1_A1_A2

### Relational analysis result of IS_A1_B1_B2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6659327, upper bound: 0.6654287
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0360885, 1.0472777, 0.0111880, 0.5648615, -0.6009500, 1.0360897
1: -0.1492593, 1.3218184, -0.0944340, 0.7130072, -0.8622665, 1.4162524
2: -0.1066136, 1.1590223, -0.0312903, 0.7124612, -0.8190749, 1.1903126
3: -0.3045232, 1.1983278, -0.2500148, 0.7301772, -1.0347004, 1.4483426
4: -0.3275616, 0.9970202, -0.1787794, 0.7720095, -1.0995711, 1.1757996

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6773662, upper bound: 0.6845283
time: 0.35 seconds

## Relational analysis of IS_A1_B1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6773662, upper bound: 0.6608971
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0037274, 1.2130017, -0.0305669, 0.6706582, -0.6669308, 1.2435687
1: -0.1025641, 1.5101075, -0.1440713, 0.8347254, -0.9372895, 1.6541789
2: -0.0643606, 1.3270278, -0.0742025, 0.8280591, -0.8924198, 1.4012303
3: -0.2630773, 1.2870049, -0.2901301, 0.8312570, -1.0943344, 1.5771351
4: -0.2849326, 1.0319966, -0.2451730, 0.8507983, -1.1357310, 1.2771696

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_B2_A2_B2_A1_A1

### Relational analysis result of IS_A1_B1_B2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6800659, upper bound: 0.6116068
time: 0.33 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6770974, upper bound: 0.6574645
time: 0.36 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6863411, upper bound: 0.6574645
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0029831, 1.0288341, -0.0305669, 0.6706582, -0.6676751, 1.0594010
1: -0.1023865, 1.3024969, -0.1440713, 0.8347254, -0.9371119, 1.4465683
2: -0.0674717, 1.1274939, -0.0742025, 0.8280591, -0.8955309, 1.2016964
3: -0.2657092, 1.1673520, -0.2901301, 0.8312570, -1.0969663, 1.4574821
4: -0.2901115, 0.9485722, -0.2451730, 0.8507983, -1.1409099, 1.1937451

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6659327, upper bound: 0.6585654
time: 0.34 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6751765, upper bound: 0.6585654
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0019321, 0.9436822, -0.0028670, 1.0353181, -1.0333860, 0.9465492
1: -0.1070485, 1.1877589, -0.1088188, 1.3095722, -1.4166207, 1.2965777
2: -0.0584593, 1.0525572, -0.0732558, 1.1363764, -1.1948357, 1.1258130
3: -0.2703054, 1.0732428, -0.2704966, 1.1744311, -1.4447365, 1.3437394
4: -0.2486541, 0.9223583, -0.2937192, 0.9576517, -1.2063059, 1.2160774

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6739993, upper bound: 0.6511192
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6739993, upper bound: 0.6511192
time: 0.34 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0019321, 0.9436822, -0.0371032, 1.8955507, -1.8936186, 0.9807854
1: -0.1070485, 1.1877589, -0.1546662, 2.3333716, -2.4404202, 1.3424251
2: -0.0584593, 1.0525572, -0.1336410, 2.0746813, -2.1331406, 1.1861982
3: -0.2703054, 1.0732428, -0.3200390, 1.8966720, -2.1669774, 1.3932818
4: -0.2486541, 0.9223583, -0.5061331, 1.4818683, -1.7305224, 1.4284914

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6739993, upper bound: 0.6526531
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6739993, upper bound: 0.6526531
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, 0.0248351, 0.8239982, -0.8128102, 0.5400264
1: -0.0944340, 0.7130072, -0.0720310, 1.0479953, -1.1424294, 0.7850382
2: -0.0312903, 0.7124612, -0.0307086, 0.9141164, -0.9454067, 0.7431698
3: -0.2500148, 0.7301772, -0.2340453, 0.9604603, -1.2104751, 0.9642225
4: -0.1787794, 0.7720095, -0.2052969, 0.8165774, -0.9953567, 0.9773064

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6937541
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6663589
time: 0.34 seconds

## BFS IS instance: IS_A2_B1_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, 0.0248351, 0.8239982, -0.8545651, 0.6458231
1: -0.1440713, 0.8347254, -0.0720310, 1.0479953, -1.1920667, 0.9067564
2: -0.0742025, 0.8280591, -0.0307086, 0.9141164, -0.9883189, 0.8587677
3: -0.2901301, 0.8312570, -0.2340453, 0.9604603, -1.2505904, 1.0653024
4: -0.2451730, 0.8507983, -0.2052969, 0.8165774, -1.0617504, 1.0560951

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6420588, upper bound: 0.7029979
time: 0.33 seconds

## Relational analysis of IS_A2_B1_A1_B1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6756027
time: 0.31 seconds

## BFS IS instance: IS_A2_B1_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, -0.0167825, 0.9250188, -0.9138308, 0.5816441
1: -0.0944340, 0.7130072, -0.1252913, 1.1663461, -1.2607801, 0.8382986
2: -0.0312903, 0.7124612, -0.0773900, 1.0296290, -1.0609193, 0.7898512
3: -0.2500148, 0.7301772, -0.2777247, 1.0633786, -1.3133934, 1.0079019
4: -0.1787794, 0.7720095, -0.2678475, 0.8997647, -1.0785441, 1.0398570

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6663589
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A1_B1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6663589
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, -0.0167825, 0.9250188, -0.9555857, 0.6874408
1: -0.1440713, 0.8347254, -0.1252913, 1.1663461, -1.3104174, 0.9600167
2: -0.0742025, 0.8280591, -0.0773900, 1.0296290, -1.1038315, 0.9054491
3: -0.2901301, 0.8312570, -0.2777247, 1.0633786, -1.3535087, 1.1089818
4: -0.2451730, 0.8507983, -0.2678475, 0.8997647, -1.1449378, 1.1186459

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6756027
time: 0.32 seconds

## Relational analysis of IS_A2_B1_A1_B1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6540380, upper bound: 0.6756027
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, 0.0037274, 1.2130017, -1.2018137, 0.5611341
1: -0.0944340, 0.7130072, -0.1025641, 1.5101075, -1.6045415, 0.8155713
2: -0.0312903, 0.7124612, -0.0643606, 1.3270278, -1.3583181, 0.7768219
3: -0.2500148, 0.7301772, -0.2630773, 1.2870049, -1.5370197, 0.9932545
4: -0.1787794, 0.7720095, -0.2849326, 1.0319966, -1.2107760, 1.0569421

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6808037, upper bound: 0.6770974
time: 0.32 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6574645, upper bound: 0.6770974
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, 0.0029831, 1.0288341, -1.0176461, 0.5618784
1: -0.0944340, 0.7130072, -0.1023865, 1.3024969, -1.3969309, 0.8153937
2: -0.0312903, 0.7124612, -0.0674717, 1.1274939, -1.1587842, 0.7799330
3: -0.2500148, 0.7301772, -0.2657092, 1.1673520, -1.4173667, 0.9958864
4: -0.1787794, 0.7720095, -0.2901115, 0.9485722, -1.1273515, 1.0621210

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6819046, upper bound: 0.6659327
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6585654, upper bound: 0.6659327
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, 0.0037274, 1.2130017, -1.2435687, 0.6669308
1: -0.1440713, 0.8347254, -0.1025641, 1.5101075, -1.6541789, 0.9372895
2: -0.0742025, 0.8280591, -0.0643606, 1.3270278, -1.4012303, 0.8924198
3: -0.2901301, 0.8312570, -0.2630773, 1.2870049, -1.5771351, 1.0943344
4: -0.2451730, 0.8507983, -0.2849326, 1.0319966, -1.2771696, 1.1357310

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6116068, upper bound: 0.6800659
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6574645, upper bound: 0.6770974
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6574645, upper bound: 0.6863411
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, 0.0029831, 1.0288341, -1.0594010, 0.6676751
1: -0.1440713, 0.8347254, -0.1023865, 1.3024969, -1.4465683, 0.9371119
2: -0.0742025, 0.8280591, -0.0674717, 1.1274939, -1.2016964, 0.8955309
3: -0.2901301, 0.8312570, -0.2657092, 1.1673520, -1.4574821, 1.0969663
4: -0.2451730, 0.8507983, -0.2901115, 0.9485722, -1.1937451, 1.1409099

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6585654, upper bound: 0.6659327
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6585654, upper bound: 0.6751765
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_A1

### Backsubstitution after applying IS history:
0: 0.0034440, 0.7121074, 0.0002806, 0.9836771, -0.9802332, 0.7118268
1: -0.1012318, 0.8823657, -0.1087384, 1.2370522, -1.3382840, 0.9911041
2: -0.0465870, 0.8481500, -0.0640910, 1.0932167, -1.1398036, 0.9122410
3: -0.2582860, 0.8474270, -0.2707281, 1.1106994, -1.3689854, 1.1181550
4: -0.2120385, 0.8468478, -0.2621267, 0.9407284, -1.1527669, 1.1089745

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A2_B2_A1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6399833, upper bound: 0.6729259
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6399833, upper bound: 0.6744598
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: 0.0019789, 0.7616005, 0.0002806, 0.9836771, -0.9816983, 0.7613199
1: -0.1036451, 0.9385105, -0.1087384, 1.2370522, -1.3406973, 1.0472490
2: -0.0519636, 0.8987544, -0.0640910, 1.0932167, -1.1451802, 0.9628453
3: -0.2597382, 0.8868446, -0.2707281, 1.1106994, -1.3704376, 1.1575727
4: -0.2227259, 0.8787709, -0.2621267, 0.9407284, -1.1634543, 1.1408976

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A2_B2_A1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6399833, upper bound: 0.6729259
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6399833, upper bound: 0.6744598
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, 0.0111880, 0.5648615, -0.5536735, 0.5536735
1: -0.0944340, 0.7130072, -0.0944340, 0.7130072, -0.8074412, 0.8074412
2: -0.0312903, 0.7124612, -0.0312903, 0.7124612, -0.7437515, 0.7437515
3: -0.2500148, 0.7301772, -0.2500148, 0.7301772, -0.9801920, 0.9801920
4: -0.1787794, 0.7720095, -0.1787794, 0.7720095, -0.9507889, 0.9507889

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6866208, upper bound: 0.6742789
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6654188, upper bound: 0.6742789
time: 0.40 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, -0.0305669, 0.6706582, -0.6594702, 0.5954285
1: -0.0944340, 0.7130072, -0.1440713, 0.8347254, -0.9291594, 0.8570786
2: -0.0312903, 0.7124612, -0.0742025, 0.8280591, -0.8593494, 0.7866638
3: -0.2500148, 0.7301772, -0.2901301, 0.8312570, -1.0812719, 1.0203073
4: -0.1787794, 0.7720095, -0.2451730, 0.8507983, -1.0295777, 1.0171825

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6937080, upper bound: 0.6742789
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6734482, upper bound: 0.6742789
time: 0.35 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, 0.0111880, 0.5648615, -0.5954285, 0.6594702
1: -0.1440713, 0.8347254, -0.0944340, 0.7130072, -0.8570786, 0.9291594
2: -0.0742025, 0.8280591, -0.0312903, 0.7124612, -0.7866638, 0.8593494
3: -0.2901301, 0.8312570, -0.2500148, 0.7301772, -1.0203073, 1.0812719
4: -0.2451730, 0.8507983, -0.1787794, 0.7720095, -1.0171825, 1.0295777

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6654188, upper bound: 0.6742789
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6654188, upper bound: 0.6835227
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, -0.0305669, 0.6706582, -0.7012252, 0.7012252
1: -0.1440713, 0.8347254, -0.1440713, 0.8347254, -0.9787967, 0.9787967
2: -0.0742025, 0.8280591, -0.0742025, 0.8280591, -0.9022617, 0.9022617
3: -0.2901301, 0.8312570, -0.2901301, 0.8312570, -1.1213872, 1.1213872
4: -0.2451730, 0.8507983, -0.2451730, 0.8507983, -1.0959713, 1.0959713

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6734482, upper bound: 0.6742789
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6734482, upper bound: 0.6835227
time: 0.38 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, 0.0061345, 0.6507710, -0.6395830, 0.5587270
1: -0.0944340, 0.7130072, -0.0976210, 0.8106803, -0.9051143, 0.8106282
2: -0.0312903, 0.7124612, -0.0417109, 0.7880348, -0.8193251, 0.7541721
3: -0.2500148, 0.7301772, -0.2534082, 0.7965333, -1.0465481, 0.9835854
4: -0.1787794, 0.7720095, -0.2023642, 0.8128391, -0.9916185, 0.9743737

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6866208, upper bound: 0.6764576
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6654033, upper bound: 0.6764576
time: 0.34 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, -0.0329778, 0.7304660, -0.7192780, 0.5978394
1: -0.0944340, 0.7130072, -0.1447692, 0.9029138, -0.9973478, 0.8577764
2: -0.0312903, 0.7124612, -0.0813575, 0.8816222, -0.9129125, 0.7938187
3: -0.2500148, 0.7301772, -0.2928538, 0.8801655, -1.1301802, 1.0230310
4: -0.1787794, 0.7720095, -0.2609611, 0.8848863, -1.0636656, 1.0329705

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6953452, upper bound: 0.6764576
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6732061, upper bound: 0.6764576
time: 0.37 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, 0.0061345, 0.6507710, -0.6813380, 0.6645237
1: -0.1440713, 0.8347254, -0.0976210, 0.8106803, -0.9547516, 0.9323463
2: -0.0742025, 0.8280591, -0.0417109, 0.7880348, -0.8622373, 0.8697700
3: -0.2901301, 0.8312570, -0.2534082, 0.7965333, -1.0866635, 1.0846653
4: -0.2451730, 0.8507983, -0.2023642, 0.8128391, -1.0580120, 1.0531626

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6654033, upper bound: 0.6764576
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6654033, upper bound: 0.6857014
time: 0.35 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, -0.0329778, 0.7304660, -0.7610329, 0.7036361
1: -0.1440713, 0.8347254, -0.1447692, 0.9029138, -1.0469851, 0.9794946
2: -0.0742025, 0.8280591, -0.0813575, 0.8816222, -0.9558247, 0.9094166
3: -0.2901301, 0.8312570, -0.2928538, 0.8801655, -1.1702956, 1.1241109
4: -0.2451730, 0.8507983, -0.2609611, 0.8848863, -1.1300592, 1.1117594

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6732061, upper bound: 0.6764576
time: 0.40 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6732061, upper bound: 0.6857014
time: 0.37 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0061345, 0.6507710, 0.0111880, 0.5648615, -0.5587270, 0.6395830
1: -0.0976210, 0.8106803, -0.0944340, 0.7130072, -0.8106282, 0.9051143
2: -0.0417109, 0.7880348, -0.0312903, 0.7124612, -0.7541721, 0.8193251
3: -0.2534082, 0.7965333, -0.2500148, 0.7301772, -0.9835854, 1.0465481
4: -0.2023642, 0.8128391, -0.1787794, 0.7720095, -0.9743737, 0.9916185

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A2_B1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6529501, upper bound: 0.6978548
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6529501, upper bound: 0.6742691
time: 0.37 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0329778, 0.7304660, 0.0111880, 0.5648615, -0.5978394, 0.7192780
1: -0.1447692, 0.9029138, -0.0944340, 0.7130072, -0.8577764, 0.9973478
2: -0.0813575, 0.8816222, -0.0312903, 0.7124612, -0.7938187, 0.9129125
3: -0.2928538, 0.8801655, -0.2500148, 0.7301772, -1.0230310, 1.1301802
4: -0.2609611, 0.8848863, -0.1787794, 0.7720095, -1.0329705, 1.0636656

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6529501, upper bound: 0.7057270
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6529501, upper bound: 0.6821413
time: 0.37 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0061345, 0.6507710, -0.0305669, 0.6706582, -0.6645237, 0.6813380
1: -0.0976210, 0.8106803, -0.1440713, 0.8347254, -0.9323463, 0.9547516
2: -0.0417109, 0.7880348, -0.0742025, 0.8280591, -0.8697700, 0.8622373
3: -0.2534082, 0.7965333, -0.2901301, 0.8312570, -1.0846653, 1.0866635
4: -0.2023642, 0.8128391, -0.2451730, 0.8507983, -1.0531626, 1.0580120

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6529501, upper bound: 0.6742691
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6529501, upper bound: 0.6742691
time: 0.37 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0329778, 0.7304660, -0.0305669, 0.6706582, -0.7036361, 0.7610329
1: -0.1447692, 0.9029138, -0.1440713, 0.8347254, -0.9794946, 1.0469851
2: -0.0813575, 0.8816222, -0.0742025, 0.8280591, -0.9094166, 0.9558247
3: -0.2928538, 0.8801655, -0.2901301, 0.8312570, -1.1241109, 1.1702956
4: -0.2609611, 0.8848863, -0.2451730, 0.8507983, -1.1117594, 1.1300592

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6529501, upper bound: 0.6821413
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6615939, upper bound: 0.6821413
time: 0.38 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0061345, 0.6507710, 0.0061345, 0.6507710, -0.6446365, 0.6446365
1: -0.0976210, 0.8106803, -0.0976210, 0.8106803, -0.9083012, 0.9083012
2: -0.0417109, 0.7880348, -0.0417109, 0.7880348, -0.8297457, 0.8297457
3: -0.2534082, 0.7965333, -0.2534082, 0.7965333, -1.0499415, 1.0499415
4: -0.2023642, 0.8128391, -0.2023642, 0.8128391, -1.0152032, 1.0152032

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6736765, upper bound: 0.6737598
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6530561, upper bound: 0.6737598
time: 0.35 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0061345, 0.6507710, -0.0329778, 0.7304660, -0.7243315, 0.6837488
1: -0.0976210, 0.8106803, -0.1447692, 0.9029138, -1.0005348, 0.9554495
2: -0.0417109, 0.7880348, -0.0813575, 0.8816222, -0.9233330, 0.8693923
3: -0.2534082, 0.7965333, -0.2928538, 0.8801655, -1.1335737, 1.0893872
4: -0.2023642, 0.8128391, -0.2609611, 0.8848863, -1.0872505, 1.0738001

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6825934, upper bound: 0.6737598
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6631956, upper bound: 0.6737598
time: 0.39 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0329778, 0.7304660, 0.0061345, 0.6507710, -0.6837488, 0.7243315
1: -0.1447692, 0.9029138, -0.0976210, 0.8106803, -0.9554495, 1.0005348
2: -0.0813575, 0.8816222, -0.0417109, 0.7880348, -0.8693923, 0.9233330
3: -0.2928538, 0.8801655, -0.2534082, 0.7965333, -1.0893872, 1.1335737
4: -0.2609611, 0.8848863, -0.2023642, 0.8128391, -1.0738001, 1.0872505

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6530561, upper bound: 0.6737598
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6530562, upper bound: 0.6836972
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0329778, 0.7304660, -0.0329778, 0.7304660, -0.7634438, 0.7634438
1: -0.1447692, 0.9029138, -0.1447692, 0.9029138, -1.0476830, 1.0476830
2: -0.0813575, 0.8816222, -0.0813575, 0.8816222, -0.9629797, 0.9629797
3: -0.2928538, 0.8801655, -0.2928538, 0.8801655, -1.1730193, 1.1730193
4: -0.2609611, 0.8848863, -0.2609611, 0.8848863, -1.1458473, 1.1458473

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6631953, upper bound: 0.6737598
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6631956, upper bound: 0.6836972
time: 0.40 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 2.73 seconds
IS_A1_B1_B2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6937541, upper bound: 0.6420589
IS_A1_B1_B2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6663589, upper bound: 0.6420588
IS_A1_B1_B2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.7029979, upper bound: 0.6420589
IS_A1_B1_B2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6756027, upper bound: 0.6420589
IS_A1_B1_B2_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6663589, upper bound: 0.6420589
IS_A1_B1_B2_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6663589, upper bound: 0.6540380
IS_A1_B1_B2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6756027, upper bound: 0.6420588
IS_A1_B1_B2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6756027, upper bound: 0.6540380
IS_A1_B1_B2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6770974, upper bound: 0.6629717
IS_A1_B1_B2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6659327, upper bound: 0.6654287
IS_A1_B1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6773662, upper bound: 0.6845283
IS_A1_B1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6773662, upper bound: 0.6608971
IS_A1_B1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6770974, upper bound: 0.6574645
IS_A1_B1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6863411, upper bound: 0.6574645
IS_A1_B1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6659327, upper bound: 0.6585654
IS_A1_B1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6751765, upper bound: 0.6585654
IS_A1_B2_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6739993, upper bound: 0.6511192
IS_A1_B2_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6739993, upper bound: 0.6511192
IS_A1_B2_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6739993, upper bound: 0.6526531
IS_A1_B2_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6739993, upper bound: 0.6526531
IS_A2_B1_A1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6937541
IS_A2_B1_A1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6663589
IS_A2_B1_A1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6420588, upper bound: 0.7029979
IS_A2_B1_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6756027
IS_A2_B1_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6663589
IS_A2_B1_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6663589
IS_A2_B1_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6756027
IS_A2_B1_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6540380, upper bound: 0.6756027
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6808037, upper bound: 0.6770974
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6574645, upper bound: 0.6770974
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6819046, upper bound: 0.6659327
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6585654, upper bound: 0.6659327
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6574645, upper bound: 0.6770974
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6574645, upper bound: 0.6863411
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6585654, upper bound: 0.6659327
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6585654, upper bound: 0.6751765
IS_A2_B1_A2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6399833, upper bound: 0.6729259
IS_A2_B1_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6399833, upper bound: 0.6744598
IS_A2_B1_A2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6399833, upper bound: 0.6729259
IS_A2_B1_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6399833, upper bound: 0.6744598
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6866208, upper bound: 0.6742789
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6654188, upper bound: 0.6742789
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6937080, upper bound: 0.6742789
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6734482, upper bound: 0.6742789
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6654188, upper bound: 0.6742789
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6654188, upper bound: 0.6835227
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6734482, upper bound: 0.6742789
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6734482, upper bound: 0.6835227
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6866208, upper bound: 0.6764576
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6654033, upper bound: 0.6764576
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6953452, upper bound: 0.6764576
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6732061, upper bound: 0.6764576
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6654033, upper bound: 0.6764576
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6654033, upper bound: 0.6857014
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6732061, upper bound: 0.6764576
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6732061, upper bound: 0.6857014
IS_A2_B2_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6529501, upper bound: 0.6978548
IS_A2_B2_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6529501, upper bound: 0.6742691
IS_A2_B2_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6529501, upper bound: 0.7057270
IS_A2_B2_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6529501, upper bound: 0.6821413
IS_A2_B2_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6529501, upper bound: 0.6742691
IS_A2_B2_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6529501, upper bound: 0.6742691
IS_A2_B2_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6529501, upper bound: 0.6821413
IS_A2_B2_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6615939, upper bound: 0.6821413
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6736765, upper bound: 0.6737598
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6530561, upper bound: 0.6737598
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6825934, upper bound: 0.6737598
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6631956, upper bound: 0.6737598
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6530561, upper bound: 0.6737598
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6530562, upper bound: 0.6836972
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6631953, upper bound: 0.6737598
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -0.6631956, upper bound: 0.6836972

## BFS IS instance: IS_A1_B1_B2_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0248351, 0.8239982, 0.0111880, 0.5648615, -0.5400264, 0.8128102
1: -0.0720310, 1.0479953, -0.0944340, 0.7130072, -0.7850382, 1.1424294
2: -0.0307086, 0.9141164, -0.0312903, 0.7124612, -0.7431698, 0.9454067
3: -0.2340453, 0.9604603, -0.2500148, 0.7301772, -0.9642225, 1.2104751
4: -0.2052969, 0.8165774, -0.1787794, 0.7720095, -0.9773064, 0.9953567

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_B2_A1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_B2_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6663589, upper bound: 0.6656445
time: 0.35 seconds

## Relational analysis of IS_A1_B1_B2_A1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_B2_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6663589, upper bound: 0.6420589
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_B2_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0167825, 0.9152751, 0.0111880, 0.5648615, -0.5816441, 0.9040871
1: -0.1252913, 1.1558418, -0.0944340, 0.7130072, -0.8382986, 1.2502759
2: -0.0773900, 1.0191219, -0.0312903, 0.7124612, -0.7898512, 1.0504122
3: -0.2777247, 1.0548213, -0.2500148, 0.7301772, -1.0079019, 1.3048360
4: -0.2678475, 0.8919395, -0.1787794, 0.7720095, -1.0398570, 1.0707188

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_B2_A1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_B2_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6663589, upper bound: 0.6656445
time: 0.34 seconds

## Relational analysis of IS_A1_B1_B2_A1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_B2_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6663589, upper bound: 0.6420588
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_B2_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0248351, 0.8239982, -0.0305669, 0.6706582, -0.6458231, 0.8545651
1: -0.0720310, 1.0479953, -0.1440713, 0.8347254, -0.9067564, 1.1920667
2: -0.0307086, 0.9141164, -0.0742025, 0.8280591, -0.8587677, 0.9883189
3: -0.2340453, 0.9604603, -0.2901301, 0.8312570, -1.0653024, 1.2505904
4: -0.2052969, 0.8165774, -0.2451730, 0.8507983, -1.0560951, 1.0617504

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_B2_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_B2_A1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_B2_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6663589, upper bound: 0.6420589
time: 0.35 seconds

## Relational analysis of IS_A1_B1_B2_A1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_B2_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6663589, upper bound: 0.6420589
time: 0.32 seconds

## BFS IS instance: IS_A1_B1_B2_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0167825, 0.9152751, -0.0305669, 0.6706582, -0.6874408, 0.9458420
1: -0.1252913, 1.1558418, -0.1440713, 0.8347254, -0.9600167, 1.2999132
2: -0.0773900, 1.0191219, -0.0742025, 0.8280591, -0.9054491, 1.0933244
3: -0.2777247, 1.0548213, -0.2901301, 0.8312570, -1.1089818, 1.3449514
4: -0.2678475, 0.8919395, -0.2451730, 0.8507983, -1.1186459, 1.1371124

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_B2_A1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_B2_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6663589, upper bound: 0.6420588
time: 0.36 seconds

## Relational analysis of IS_A1_B1_B2_A1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_B2_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6663589, upper bound: 0.6420589
time: 0.32 seconds

## BFS IS instance: IS_A1_B1_B2_A1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0248351, 0.8239982, 0.0111880, 0.5648615, -0.5400264, 0.8128102
1: -0.0720310, 1.0479953, -0.0944340, 0.7130072, -0.7850382, 1.1424294
2: -0.0307086, 0.9141164, -0.0312903, 0.7124612, -0.7431698, 0.9454067
3: -0.2340453, 0.9604603, -0.2500148, 0.7301772, -0.9642225, 1.2104751
4: -0.2052969, 0.8165774, -0.1787794, 0.7720095, -0.9773064, 0.9953567

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_B2_A1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_B2_A1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6663589, upper bound: 0.6656445
time: 0.34 seconds

## Relational analysis of IS_A1_B1_B2_A1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_B2_A1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6663589, upper bound: 0.6420588
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_B2_A1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0167825, 0.9250188, 0.0111880, 0.5648615, -0.5816441, 0.9138308
1: -0.1252913, 1.1663461, -0.0944340, 0.7130072, -0.8382986, 1.2607801
2: -0.0773900, 1.0296290, -0.0312903, 0.7124612, -0.7898512, 1.0609193
3: -0.2777247, 1.0633786, -0.2500148, 0.7301772, -1.0079019, 1.3133934
4: -0.2678475, 0.8997647, -0.1787794, 0.7720095, -1.0398570, 1.0785441

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_B2_A1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_B2_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6663589, upper bound: 0.6776237
time: 0.36 seconds

## Relational analysis of IS_A1_B1_B2_A1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_B2_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6663589, upper bound: 0.6540380
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_B2_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0248351, 0.8239982, -0.0305669, 0.6706582, -0.6458231, 0.8545651
1: -0.0720310, 1.0479953, -0.1440713, 0.8347254, -0.9067564, 1.1920667
2: -0.0307086, 0.9141164, -0.0742025, 0.8280591, -0.8587677, 0.9883189
3: -0.2340453, 0.9604603, -0.2901301, 0.8312570, -1.0653024, 1.2505904
4: -0.2052969, 0.8165774, -0.2451730, 0.8507983, -1.0560951, 1.0617504

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_B2_A1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_B2_A1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_B2_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6663589, upper bound: 0.6420588
time: 0.36 seconds

## Relational analysis of IS_A1_B1_B2_A1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_B2_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6663589, upper bound: 0.6420589
time: 0.32 seconds

## BFS IS instance: IS_A1_B1_B2_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0167825, 0.9250188, -0.0305669, 0.6706582, -0.6874408, 0.9555857
1: -0.1252913, 1.1663461, -0.1440713, 0.8347254, -0.9600167, 1.3104174
2: -0.0773900, 1.0296290, -0.0742025, 0.8280591, -0.9054491, 1.1038315
3: -0.2777247, 1.0633786, -0.2901301, 0.8312570, -1.1089818, 1.3535087
4: -0.2678475, 0.8997647, -0.2451730, 0.8507983, -1.1186459, 1.1449378

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_B2_A1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_B2_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6663589, upper bound: 0.6540380
time: 0.34 seconds

## Relational analysis of IS_A1_B1_B2_A1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_B2_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6756027, upper bound: 0.6540380
time: 0.36 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0080442, 1.1417933, 0.0111880, 0.5648615, -0.5568173, 1.1306052
1: -0.0962224, 1.4277072, -0.0944340, 0.7130072, -0.8092296, 1.5221412
2: -0.0576441, 1.2487886, -0.0312903, 0.7124612, -0.7701054, 1.2800789
3: -0.2561867, 1.2262950, -0.2500148, 0.7301772, -0.9863639, 1.4763098
4: -0.2707347, 0.9847741, -0.1787794, 0.7720095, -1.0427442, 1.1635535

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 29

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_B2_A2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_B2_A2_B1_A1_A1_B1

### Relational analysis result of IS_A1_B1_B2_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6986278, upper bound: 0.6396325
time: 0.38 seconds

## Relational analysis of IS_A1_B1_B2_A2_B1_A1_A1_B2

### Relational analysis result of IS_A1_B1_B2_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6986278, upper bound: 0.6396325
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0078752, 0.9549458, 0.0111880, 0.5648615, -0.5569863, 0.9437578
1: -0.0954776, 1.2169664, -0.0944340, 0.7130072, -0.8084848, 1.3114004
2: -0.0602107, 1.0485539, -0.0312903, 0.7124612, -0.7726719, 1.0798442
3: -0.2583208, 1.1043172, -0.2500148, 0.7301772, -0.9884980, 1.3543320
4: -0.2754943, 0.8998972, -0.1787794, 0.7720095, -1.0475038, 1.0786766

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_B2_A2_B1_A1_A2_B1

### Relational analysis result of IS_A1_B1_B2_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6880626, upper bound: 0.6420895
time: 0.35 seconds

## Relational analysis of IS_A1_B1_B2_A2_B1_A1_A2_B2

### Relational analysis result of IS_A1_B1_B2_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6880626, upper bound: 0.6420895
time: 0.35 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0360885, 1.0472777, 0.0111880, 0.5648615, -0.6009500, 1.0360897
1: -0.1492593, 1.3218184, -0.0944340, 0.7130072, -0.8622665, 1.4162524
2: -0.1066136, 1.1590223, -0.0312903, 0.7124612, -0.8190749, 1.1903126
3: -0.3045232, 1.1983278, -0.2500148, 0.7301772, -1.0347004, 1.4483426
4: -0.3275616, 0.9970202, -0.1787794, 0.7720095, -1.0995711, 1.1757996

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6773662, upper bound: 0.6444212
time: 0.36 seconds

## Relational analysis of IS_A1_B1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6773662, upper bound: 0.6608971
time: 0.37 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0360885, 1.0472777, -0.0305669, 0.6706582, -0.7067467, 1.0778446
1: -0.1492593, 1.3218184, -0.1440713, 0.8347254, -0.9839847, 1.4658897
2: -0.1066136, 1.1590223, -0.0742025, 0.8280591, -0.9346728, 1.2332249
3: -0.3045232, 1.1983278, -0.2901301, 0.8312570, -1.1357803, 1.4884579
4: -0.3275616, 0.9970202, -0.2451730, 0.8507983, -1.1783600, 1.2421932

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6773662, upper bound: 0.6444212
time: 0.36 seconds

## Relational analysis of IS_A1_B1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6773662, upper bound: 0.6608971
time: 0.37 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0037274, 1.2130017, 0.0111880, 0.5648615, -0.5611341, 1.2018137
1: -0.1025641, 1.5101075, -0.0944340, 0.7130072, -0.8155713, 1.6045415
2: -0.0643606, 1.3270278, -0.0312903, 0.7124612, -0.7768219, 1.3583181
3: -0.2630773, 1.2870049, -0.2500148, 0.7301772, -0.9932545, 1.5370197
4: -0.2849326, 1.0319966, -0.1787794, 0.7720095, -1.0569421, 1.2107760

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6770974, upper bound: 0.6808037
time: 0.37 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6659327, upper bound: 0.6808037
time: 0.35 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0037274, 1.2130017, -0.0305669, 0.6706582, -0.6669308, 1.2435687
1: -0.1025641, 1.5101075, -0.1440713, 0.8347254, -0.9372895, 1.6541789
2: -0.0643606, 1.3270278, -0.0742025, 0.8280591, -0.8924198, 1.4012303
3: -0.2630773, 1.2870049, -0.2901301, 0.8312570, -1.0943344, 1.5771351
4: -0.2849326, 1.0319966, -0.2451730, 0.8507983, -1.1357310, 1.2771696

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6863411, upper bound: 0.6574645
time: 0.35 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6751765, upper bound: 0.6574645
time: 0.35 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0029831, 1.0288341, 0.0111880, 0.5648615, -0.5618784, 1.0176461
1: -0.1023865, 1.3024969, -0.0944340, 0.7130072, -0.8153937, 1.3969309
2: -0.0674717, 1.1274939, -0.0312903, 0.7124612, -0.7799330, 1.1587842
3: -0.2657092, 1.1673520, -0.2500148, 0.7301772, -0.9958864, 1.4173667
4: -0.2901115, 0.9485722, -0.1787794, 0.7720095, -1.0621210, 1.1273515

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6659327, upper bound: 0.6808037
time: 0.36 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6659327, upper bound: 0.6819046
time: 0.36 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0029831, 1.0288341, -0.0305669, 0.6706582, -0.6676751, 1.0594010
1: -0.1023865, 1.3024969, -0.1440713, 0.8347254, -0.9371119, 1.4465683
2: -0.0674717, 1.1274939, -0.0742025, 0.8280591, -0.8955309, 1.2016964
3: -0.2657092, 1.1673520, -0.2901301, 0.8312570, -1.0969663, 1.4574821
4: -0.2901115, 0.9485722, -0.2451730, 0.8507983, -1.1409099, 1.1937451

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6751765, upper bound: 0.6574645
time: 0.35 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6751765, upper bound: 0.6585654
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0019321, 0.9436822, -0.0014365, 1.0292757, -1.0273435, 0.9451187
1: -0.1070485, 1.1877589, -0.1070645, 1.3027349, -1.4097834, 1.2948234
2: -0.0584593, 1.0525572, -0.0715175, 1.1294107, -1.1878700, 1.1240747
3: -0.2703054, 1.0732428, -0.2690842, 1.1685212, -1.4388266, 1.3423270
4: -0.2486541, 0.9223583, -0.2915230, 0.9521359, -1.2007900, 1.2138813

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 5

Time for candidate selection: 2.33 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6844666, upper bound: 0.6348030
time: 0.32 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6626330, upper bound: 0.6504680
time: 0.33 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0019321, 0.9436822, -0.0034957, 1.0884125, -1.0864804, 0.9471779
1: -0.1070485, 1.1877589, -0.1095273, 1.3744173, -1.4814658, 1.2972863
2: -0.0584593, 1.0525572, -0.0774741, 1.1921680, -1.2506273, 1.1300313
3: -0.2703054, 1.0732428, -0.2709143, 1.2222157, -1.4925210, 1.3441571
4: -0.2486541, 0.9223583, -0.3122784, 0.9898332, -1.2384874, 1.2346367

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 5

Time for candidate selection: 2.27 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6844666, upper bound: 0.6348030
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6626330, upper bound: 0.6504680
time: 0.32 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0019321, 0.9436822, -0.0357146, 1.8917470, -1.8898149, 0.9793968
1: -0.1070485, 1.1877589, -0.1529815, 2.3290219, -2.4360704, 1.3407404
2: -0.0584593, 1.0525572, -0.1319740, 2.0703101, -2.1287694, 1.1845312
3: -0.2703054, 1.0732428, -0.3186724, 1.8925552, -2.1628606, 1.3919152
4: -0.2486541, 0.9223583, -0.5038834, 1.4782946, -1.7269487, 1.4262416

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 29

Time for candidate selection: 2.29 seconds

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_B1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6633969, upper bound: 0.6521729
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6726109, upper bound: 0.6363368
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6507773, upper bound: 0.6520017
time: 0.34 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0019321, 0.9436822, -0.0353863, 1.9274912, -1.9255590, 0.9790685
1: -0.1070485, 1.1877589, -0.1541932, 2.3743386, -2.4813871, 1.3419521
2: -0.0584593, 1.0525572, -0.1364565, 2.1104116, -2.1688709, 1.1890137
3: -0.2703054, 1.0732428, -0.3205948, 1.9242938, -2.1945992, 1.3938376
4: -0.2486541, 0.9223583, -0.5220038, 1.5027575, -1.7514117, 1.4443620

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 29

Time for candidate selection: 2.29 seconds

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_B2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6633969, upper bound: 0.6521729
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6726109, upper bound: 0.6363368
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6507773, upper bound: 0.6520017
time: 0.34 seconds

## BFS IS instance: IS_A2_B1_A1_B1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, 0.0248351, 0.8239982, -0.8128102, 0.5400264
1: -0.0944340, 0.7130072, -0.0720310, 1.0479953, -1.1424294, 0.7850382
2: -0.0312903, 0.7124612, -0.0307086, 0.9141164, -0.9454067, 0.7431698
3: -0.2500148, 0.7301772, -0.2340453, 0.9604603, -1.2104751, 0.9642225
4: -0.1787794, 0.7720095, -0.2052969, 0.8165774, -0.9953567, 0.9773064

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6656445, upper bound: 0.6663589
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A1_B1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6663589
time: 0.34 seconds

## BFS IS instance: IS_A2_B1_A1_B1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, -0.0167825, 0.9152751, -0.9040871, 0.5816441
1: -0.0944340, 0.7130072, -0.1252913, 1.1558418, -1.2502759, 0.8382986
2: -0.0312903, 0.7124612, -0.0773900, 1.0191219, -1.0504122, 0.7898512
3: -0.2500148, 0.7301772, -0.2777247, 1.0548213, -1.3048360, 1.0079019
4: -0.1787794, 0.7720095, -0.2678475, 0.8919395, -1.0707188, 1.0398570

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6656445, upper bound: 0.6663589
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A1_B1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6663589
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_A1_B1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, 0.0248351, 0.8239982, -0.8545651, 0.6458231
1: -0.1440713, 0.8347254, -0.0720310, 1.0479953, -1.1920667, 0.9067564
2: -0.0742025, 0.8280591, -0.0307086, 0.9141164, -0.9883189, 0.8587677
3: -0.2901301, 0.8312570, -0.2340453, 0.9604603, -1.2505904, 1.0653024
4: -0.2451730, 0.8507983, -0.2052969, 0.8165774, -1.0617504, 1.0560951

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6663589
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A1_B1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6756027
time: 0.34 seconds

## BFS IS instance: IS_A2_B1_A1_B1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, -0.0167825, 0.9152751, -0.9458420, 0.6874408
1: -0.1440713, 0.8347254, -0.1252913, 1.1558418, -1.2999132, 0.9600167
2: -0.0742025, 0.8280591, -0.0773900, 1.0191219, -1.0933244, 0.9054491
3: -0.2901301, 0.8312570, -0.2777247, 1.0548213, -1.3449514, 1.1089818
4: -0.2451730, 0.8507983, -0.2678475, 0.8919395, -1.1371124, 1.1186459

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6663589
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A1_B1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6756027
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_A1_B1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, 0.0248351, 0.8239982, -0.8128102, 0.5400264
1: -0.0944340, 0.7130072, -0.0720310, 1.0479953, -1.1424294, 0.7850382
2: -0.0312903, 0.7124612, -0.0307086, 0.9141164, -0.9454067, 0.7431698
3: -0.2500148, 0.7301772, -0.2340453, 0.9604603, -1.2104751, 0.9642225
4: -0.1787794, 0.7720095, -0.2052969, 0.8165774, -0.9953567, 0.9773064

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6656445, upper bound: 0.6663589
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A1_B1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6663589
time: 0.34 seconds

## BFS IS instance: IS_A2_B1_A1_B1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, -0.0167825, 0.9250188, -0.9138308, 0.5816441
1: -0.0944340, 0.7130072, -0.1252913, 1.1663461, -1.2607801, 0.8382986
2: -0.0312903, 0.7124612, -0.0773900, 1.0296290, -1.0609193, 0.7898512
3: -0.2500148, 0.7301772, -0.2777247, 1.0633786, -1.3133934, 1.0079019
4: -0.1787794, 0.7720095, -0.2678475, 0.8997647, -1.0785441, 1.0398570

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6776237, upper bound: 0.6663589
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A1_B1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6540380, upper bound: 0.6663589
time: 0.34 seconds

## BFS IS instance: IS_A2_B1_A1_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, 0.0248351, 0.8239982, -0.8545651, 0.6458231
1: -0.1440713, 0.8347254, -0.0720310, 1.0479953, -1.1920667, 0.9067564
2: -0.0742025, 0.8280591, -0.0307086, 0.9141164, -0.9883189, 0.8587677
3: -0.2901301, 0.8312570, -0.2340453, 0.9604603, -1.2505904, 1.0653024
4: -0.2451730, 0.8507983, -0.2052969, 0.8165774, -1.0617504, 1.0560951

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6663589
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A1_B1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6420588, upper bound: 0.6756027
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_A1_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, -0.0167825, 0.9250188, -0.9555857, 0.6874408
1: -0.1440713, 0.8347254, -0.1252913, 1.1663461, -1.3104174, 0.9600167
2: -0.0742025, 0.8280591, -0.0773900, 1.0296290, -1.1038315, 0.9054491
3: -0.2901301, 0.8312570, -0.2777247, 1.0633786, -1.3535087, 1.1089818
4: -0.2451730, 0.8507983, -0.2678475, 0.8997647, -1.1449378, 1.1186459

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6540380, upper bound: 0.6663589
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A1_B1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6540380, upper bound: 0.6756027
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, 0.0037274, 1.2130017, -1.2018137, 0.5611341
1: -0.0944340, 0.7130072, -0.1025641, 1.5101075, -1.6045415, 0.8155713
2: -0.0312903, 0.7124612, -0.0643606, 1.3270278, -1.3583181, 0.7768219
3: -0.2500148, 0.7301772, -0.2630773, 1.2870049, -1.5370197, 0.9932545
4: -0.1787794, 0.7720095, -0.2849326, 1.0319966, -1.2107760, 1.0569421

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6808037, upper bound: 0.6770974
time: 0.32 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6808037, upper bound: 0.6659327
time: 0.34 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, 0.0037274, 1.2130017, -1.2435687, 0.6669308
1: -0.1440713, 0.8347254, -0.1025641, 1.5101075, -1.6541789, 0.9372895
2: -0.0742025, 0.8280591, -0.0643606, 1.3270278, -1.4012303, 0.8924198
3: -0.2901301, 0.8312570, -0.2630773, 1.2870049, -1.5771351, 1.0943344
4: -0.2451730, 0.8507983, -0.2849326, 1.0319966, -1.2771696, 1.1357310

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6574645, upper bound: 0.6770974
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6574645, upper bound: 0.6659327
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, 0.0029831, 1.0288341, -1.0176461, 0.5618784
1: -0.0944340, 0.7130072, -0.1023865, 1.3024969, -1.3969309, 0.8153937
2: -0.0312903, 0.7124612, -0.0674717, 1.1274939, -1.1587842, 0.7799330
3: -0.2500148, 0.7301772, -0.2657092, 1.1673520, -1.4173667, 0.9958864
4: -0.1787794, 0.7720095, -0.2901115, 0.9485722, -1.1273515, 1.0621210

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6808037, upper bound: 0.6659327
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6819046, upper bound: 0.6659327
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, 0.0029831, 1.0288341, -1.0594010, 0.6676751
1: -0.1440713, 0.8347254, -0.1023865, 1.3024969, -1.4465683, 0.9371119
2: -0.0742025, 0.8280591, -0.0674717, 1.1274939, -1.2016964, 0.8955309
3: -0.2901301, 0.8312570, -0.2657092, 1.1673520, -1.4574821, 1.0969663
4: -0.2451730, 0.8507983, -0.2901115, 0.9485722, -1.1937451, 1.1409099

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6574645, upper bound: 0.6659327
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6585654, upper bound: 0.6659327
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, 0.0037274, 1.2130017, -1.2018137, 0.5611341
1: -0.0944340, 0.7130072, -0.1025641, 1.5101075, -1.6045415, 0.8155713
2: -0.0312903, 0.7124612, -0.0643606, 1.3270278, -1.3583181, 0.7768219
3: -0.2500148, 0.7301772, -0.2630773, 1.2870049, -1.5370197, 0.9932545
4: -0.1787794, 0.7720095, -0.2849326, 1.0319966, -1.2107760, 1.0569421

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6808037, upper bound: 0.6770974
time: 0.32 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6808037, upper bound: 0.6659327
time: 0.34 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, 0.0037274, 1.2130017, -1.2435687, 0.6669308
1: -0.1440713, 0.8347254, -0.1025641, 1.5101075, -1.6541789, 0.9372895
2: -0.0742025, 0.8280591, -0.0643606, 1.3270278, -1.4012303, 0.8924198
3: -0.2901301, 0.8312570, -0.2630773, 1.2870049, -1.5771351, 1.0943344
4: -0.2451730, 0.8507983, -0.2849326, 1.0319966, -1.2771696, 1.1357310

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6574645, upper bound: 0.6863411
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6574645, upper bound: 0.6751765
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, 0.0029831, 1.0288341, -1.0176461, 0.5618784
1: -0.0944340, 0.7130072, -0.1023865, 1.3024969, -1.3969309, 0.8153937
2: -0.0312903, 0.7124612, -0.0674717, 1.1274939, -1.1587842, 0.7799330
3: -0.2500148, 0.7301772, -0.2657092, 1.1673520, -1.4173667, 0.9958864
4: -0.1787794, 0.7720095, -0.2901115, 0.9485722, -1.1273515, 1.0621210

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6808037, upper bound: 0.6659327
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6819046, upper bound: 0.6659327
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, 0.0029831, 1.0288341, -1.0594010, 0.6676751
1: -0.1440713, 0.8347254, -0.1023865, 1.3024969, -1.4465683, 0.9371119
2: -0.0742025, 0.8280591, -0.0674717, 1.1274939, -1.2016964, 0.8955309
3: -0.2901301, 0.8312570, -0.2657092, 1.1673520, -1.4574821, 1.0969663
4: -0.2451730, 0.8507983, -0.2901115, 0.9485722, -1.1937451, 1.1409099

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6574645, upper bound: 0.6751765
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6585654, upper bound: 0.6751765
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0034440, 0.7121074, 0.0019100, 0.9781148, -0.9746709, 0.7101974
1: -0.1012318, 0.8823657, -0.1064947, 1.2307270, -1.3319588, 0.9888604
2: -0.0465870, 0.8481500, -0.0622022, 1.0866206, -1.1332076, 0.9103522
3: -0.2582860, 0.8474270, -0.2688622, 1.1052685, -1.3635545, 1.1162891
4: -0.2120385, 0.8468478, -0.2600321, 0.9351501, -1.1471887, 1.1068798

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A2_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5

Time for candidate selection: 2.41 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A2_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A2_B2_A1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6766341, upper bound: 0.6645525
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6545333, upper bound: 0.6724505
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0034440, 0.7121074, -0.0409024, 1.8852544, -1.8818104, 0.7530098
1: -0.1012318, 0.8823657, -0.1676061, 2.3075399, -2.4087718, 1.0499718
2: -0.0465870, 0.8481500, -0.1378353, 2.0718379, -2.1184249, 0.9859853
3: -0.2582860, 0.8474270, -0.3307943, 1.8703117, -2.1285977, 1.1782212
4: -0.2120385, 0.8468478, -0.4962190, 1.5079451, -1.7199836, 1.3430668

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A2_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 29

Time for candidate selection: 2.39 seconds

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A2_B1_A2_B2_A1_A1_B2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6205237, upper bound: 0.6721715
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A2_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A2_B2_A1_A1_B2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6715263, upper bound: 0.6741586
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A2_B2_A1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6766341, upper bound: 0.6660864
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6545333, upper bound: 0.6739844
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0019789, 0.7616005, 0.0019100, 0.9781148, -0.9761360, 0.7596905
1: -0.1036451, 0.9385105, -0.1064947, 1.2307270, -1.3343720, 1.0450052
2: -0.0519636, 0.8987544, -0.0622022, 1.0866206, -1.1385841, 0.9609566
3: -0.2597382, 0.8868446, -0.2688622, 1.1052685, -1.3650067, 1.1557069
4: -0.2227259, 0.8787709, -0.2600321, 0.9351501, -1.1578760, 1.1388030

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A2_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5

Time for candidate selection: 2.42 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A2_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_A2_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 0): status=Status.UNKNOWN, low=0.0017312, high=0.0917747, mid=0.0917747, abs_max=0.7819017171859741
rel_dist={0: [-0.7126414684186158, 0.7126414684186155]}

## Binary search (step 1) starts
Candidate diff: 0.0467529


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6862236, upper bound: 0.6999231
time: 0.31 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7000153, upper bound: 0.7000153
time: 0.32 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.78 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 0.78
Output dim: 0, lower bound: -0.6862236, upper bound: 0.6999231
IS_B2, status: Status.UNKNOWN, split count: 1, time: 0.78
Output dim: 0, lower bound: -0.7000153, upper bound: 0.7000153

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -0.0274801, 0.7245864, -0.0017378, 0.9577341, -0.9852142, 0.7263242
1: -0.1478488, 0.8963833, -0.1117451, 1.2035816, -1.3514304, 1.0081284
2: -0.0690396, 0.8955200, -0.0625789, 1.0695953, -1.1386349, 0.9580989
3: -0.3069696, 0.8793733, -0.2743888, 1.0868173, -1.3937869, 1.1537621
4: -0.2427759, 0.9202661, -0.2543875, 0.9355542, -1.1783302, 1.1746535

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_B1

### Relational analysis result of IS_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6514900, upper bound: 0.6989062
time: 0.32 seconds

## Relational analysis of IS_B1_B2

### Relational analysis result of IS_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6514900, upper bound: 0.6658316
time: 0.30 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -0.0323386, 0.7495631, -0.0124662, 0.7046853, -0.7370239, 0.7620293
1: -0.1552227, 0.9255341, -0.1260490, 0.8739170, -1.0291396, 1.0515832
2: -0.0741289, 0.9267865, -0.0592167, 0.8673251, -0.9414539, 0.9860032
3: -0.3143692, 0.9027434, -0.2830637, 0.8561093, -1.1704785, 1.1858070
4: -0.2515757, 0.9410419, -0.2274244, 0.8807485, -1.1323242, 1.1684663

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6996897, upper bound: 0.6862236
time: 0.29 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6996897, upper bound: 0.6862236
time: 0.32 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.31 seconds
IS_B1_B1, status: Status.UNKNOWN, split count: 2, time: 2.31
Output dim: 0, lower bound: -0.6514900, upper bound: 0.6989062
IS_B1_B2, status: Status.UNKNOWN, split count: 2, time: 2.31
Output dim: 0, lower bound: -0.6514900, upper bound: 0.6658316
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 2.31
Output dim: 0, lower bound: -0.6996897, upper bound: 0.6862236
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 2.31
Output dim: 0, lower bound: -0.6996897, upper bound: 0.6862236

## BFS IS instance: IS_B1_B1

### Backsubstitution after applying IS history:
0: -0.0274801, 0.7245864, -0.0004439, 0.9531150, -0.9805951, 0.7250303
1: -0.1478488, 0.8963833, -0.1101191, 1.1983986, -1.3462474, 1.0065024
2: -0.0690396, 0.8955200, -0.0610516, 1.0641370, -1.1331766, 0.9565716
3: -0.3069696, 0.8793733, -0.2731256, 1.0821323, -1.3891020, 1.1524990
4: -0.2427759, 0.9202661, -0.2521458, 0.9308868, -1.1736627, 1.1724119

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_B1_A1

### Relational analysis result of IS_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6514900, upper bound: 0.6866269
time: 0.30 seconds

## Relational analysis of IS_B1_B1_A2

### Relational analysis result of IS_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6514900, upper bound: 0.6989062
time: 0.30 seconds

## BFS IS instance: IS_B1_B2

### Backsubstitution after applying IS history:
0: -0.0268619, 0.7212657, -0.0001364, 0.9854338, -1.0122957, 0.7214020
1: -0.1471319, 0.8925924, -0.1093853, 1.2390206, -1.3861525, 1.0019777
2: -0.0683510, 0.8916904, -0.0645683, 1.0958974, -1.1642485, 0.9562587
3: -0.3063376, 0.8762900, -0.2715011, 1.1123974, -1.4187350, 1.1477911
4: -0.2415609, 0.9177195, -0.2628224, 0.9425689, -1.1841298, 1.1805419

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_B2_A1

### Relational analysis result of IS_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6514900, upper bound: 0.6535523
time: 0.30 seconds

## Relational analysis of IS_B1_B2_A2

### Relational analysis result of IS_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6514900, upper bound: 0.6658316
time: 0.30 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -0.0017378, 0.9577341, -0.0124662, 0.7046853, -0.7064232, 0.9702003
1: -0.1117451, 1.2035816, -0.1260490, 0.8739170, -0.9856621, 1.3296306
2: -0.0625789, 1.0695953, -0.0592167, 0.8673251, -0.9299040, 1.1288121
3: -0.2743888, 1.0868173, -0.2830637, 0.8561093, -1.1304981, 1.3698809
4: -0.2543875, 0.9355542, -0.2274244, 0.8807485, -1.1351360, 1.1629786

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_A1

### Relational analysis result of IS_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6862148, upper bound: 0.6514900
time: 0.30 seconds

## Relational analysis of IS_B2_A1_A2

### Relational analysis result of IS_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6521396, upper bound: 0.6521396
time: 0.30 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -0.0124662, 0.7046853, -0.0124662, 0.7046853, -0.7171515, 0.7171515
1: -0.1260490, 0.8739170, -0.1260490, 0.8739170, -0.9999660, 0.9999660
2: -0.0592167, 0.8673251, -0.0592167, 0.8673251, -0.9265418, 0.9265418
3: -0.2830637, 0.8561093, -0.2830637, 0.8561093, -1.1391729, 1.1391729
4: -0.2274244, 0.8807485, -0.2274244, 0.8807485, -1.1081729, 1.1081729

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_A2_B1

### Relational analysis result of IS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6779584, upper bound: 0.6973597
time: 0.30 seconds

## Relational analysis of IS_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6779584, upper bound: 0.6977845
time: 0.33 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.02 seconds
IS_B1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.02
Output dim: 0, lower bound: -0.6514900, upper bound: 0.6866269
IS_B1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.02
Output dim: 0, lower bound: -0.6514900, upper bound: 0.6989062
IS_B1_B2_A1, status: Status.VERIFIED, split count: 3, time: 3.02
Output dim: 0, lower bound: -0.6514900, upper bound: 0.6535523
IS_B1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.02
Output dim: 0, lower bound: -0.6514900, upper bound: 0.6658316
IS_B2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 3.02
Output dim: 0, lower bound: -0.6862148, upper bound: 0.6514900
IS_B2_A1_A2, status: Status.VERIFIED, split count: 3, time: 3.02
Output dim: 0, lower bound: -0.6521396, upper bound: 0.6521396
IS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 3.02
Output dim: 0, lower bound: -0.6779584, upper bound: 0.6973597
IS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 3.02
Output dim: 0, lower bound: -0.6779584, upper bound: 0.6977845

## BFS IS instance: IS_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0017378, 0.9577341, -0.0004439, 0.9531150, -0.9548528, 0.9581780
1: -0.1117451, 1.2035816, -0.1101191, 1.1983986, -1.3101437, 1.3137007
2: -0.0625789, 1.0695953, -0.0610516, 1.0641370, -1.1267159, 1.1306469
3: -0.2743888, 1.0868173, -0.2731256, 1.0821323, -1.3565211, 1.3599429
4: -0.2543875, 0.9355542, -0.2521458, 0.9308868, -1.1852744, 1.1877000

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B1_B1_A1_B1

### Relational analysis result of IS_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6403625, upper bound: 0.6809969
time: 0.31 seconds

## Relational analysis of IS_B1_B1_A1_B2

### Relational analysis result of IS_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6513009, upper bound: 0.6860459
time: 0.32 seconds

## BFS IS instance: IS_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0124662, 0.7046853, -0.0004439, 0.9531150, -0.9655812, 0.7051293
1: -0.1260490, 0.8739170, -0.1101191, 1.1983986, -1.3244476, 0.9840361
2: -0.0592167, 0.8673251, -0.0610516, 1.0641370, -1.1233537, 0.9283767
3: -0.2830637, 0.8561093, -0.2731256, 1.0821323, -1.3651960, 1.1292349
4: -0.2274244, 0.8807485, -0.2521458, 0.9308868, -1.1583111, 1.1328943

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_B1_A2_A1

### Relational analysis result of IS_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6399833, upper bound: 0.6840421
time: 0.31 seconds

## Relational analysis of IS_B1_B1_A2_A2

### Relational analysis result of IS_B1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6397003, upper bound: 0.6609718
time: 0.28 seconds

## BFS IS instance: IS_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0118110, 0.7013276, -0.0001364, 0.9854338, -0.9972448, 0.7014639
1: -0.1252403, 0.8701496, -0.1093853, 1.2390206, -1.3642609, 0.9795349
2: -0.0585194, 0.8634878, -0.0645683, 1.0958974, -1.1544168, 0.9280561
3: -0.2823272, 0.8530076, -0.2715011, 1.1123974, -1.3947246, 1.1245086
4: -0.2261667, 0.8779168, -0.2628224, 0.9425689, -1.1687356, 1.1407392

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_B2_A2_A1

### Relational analysis result of IS_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6399833, upper bound: 0.6657710
time: 0.33 seconds

## Relational analysis of IS_B1_B2_A2_A2

### Relational analysis result of IS_B1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6438791, upper bound: 0.6427007
time: 0.30 seconds

## BFS IS instance: IS_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0004439, 0.9531150, -0.0124662, 0.7046853, -0.7051293, 0.9655812
1: -0.1101191, 1.1983986, -0.1260490, 0.8739170, -0.9840361, 1.3244476
2: -0.0610516, 1.0641370, -0.0592167, 0.8673251, -0.9283767, 1.1233537
3: -0.2731256, 1.0821323, -0.2830637, 0.8561093, -1.1292349, 1.3651960
4: -0.2521458, 0.9308868, -0.2274244, 0.8807485, -1.1328943, 1.1583111

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_A1_B1

### Relational analysis result of IS_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6840421, upper bound: 0.6399833
time: 0.32 seconds

## Relational analysis of IS_B2_A1_A1_B2

### Relational analysis result of IS_B2_A1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6609718, upper bound: 0.6397003
time: 0.31 seconds

## BFS IS instance: IS_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0124662, 0.7046853, 0.0054424, 0.6499398, -0.6624060, 0.6992429
1: -0.1260490, 0.8739170, -0.1022949, 0.8109143, -0.9369633, 0.9762119
2: -0.0592167, 0.8673251, -0.0393627, 0.7986160, -0.8578327, 0.9066877
3: -0.2830637, 0.8561093, -0.2590857, 0.7998861, -1.0829498, 1.1151949
4: -0.2274244, 0.8807485, -0.1939640, 0.8231770, -1.0506014, 1.0747125

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_B1_A1

### Relational analysis result of IS_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6648599, upper bound: 0.6813945
time: 0.35 seconds

## Relational analysis of IS_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_A2_B1_A1

### Relational analysis result of IS_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6779333, upper bound: 0.6973597
time: 0.43 seconds

## Relational analysis of IS_B2_A2_B1_A2

### Relational analysis result of IS_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6779333, upper bound: 0.6973597
time: 0.35 seconds

## BFS IS instance: IS_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0070884, 0.6870102, 0.0004525, 0.7221838, -0.7292722, 0.6865577
1: -0.1182010, 0.8538061, -0.1048932, 0.8939738, -1.0121748, 0.9586993
2: -0.0546446, 0.8421040, -0.0500441, 0.8598962, -0.9145408, 0.8921480
3: -0.2754803, 0.8384964, -0.2612641, 0.8577256, -1.1332059, 1.0997605
4: -0.2195926, 0.8588813, -0.2176540, 0.8560207, -1.0756133, 1.0765352

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_A2_B2_A1

### Relational analysis result of IS_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6851157, upper bound: 0.6977845
time: 0.34 seconds

## Relational analysis of IS_B2_A2_B2_A2

### Relational analysis result of IS_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6851157, upper bound: 0.6977845
time: 0.32 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.23 seconds
IS_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.23
Output dim: 0, lower bound: -0.6403625, upper bound: 0.6809969
IS_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.23
Output dim: 0, lower bound: -0.6513009, upper bound: 0.6860459
IS_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 3.23
Output dim: 0, lower bound: -0.6399833, upper bound: 0.6840421
IS_B1_B1_A2_A2, status: Status.VERIFIED, split count: 4, time: 3.23
Output dim: 0, lower bound: -0.6397003, upper bound: 0.6609718
IS_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 3.23
Output dim: 0, lower bound: -0.6399833, upper bound: 0.6657710
IS_B1_B2_A2_A2, status: Status.VERIFIED, split count: 4, time: 3.23
Output dim: 0, lower bound: -0.6438791, upper bound: 0.6427007
IS_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.23
Output dim: 0, lower bound: -0.6840421, upper bound: 0.6399833
IS_B2_A1_A1_B2, status: Status.VERIFIED, split count: 4, time: 3.23
Output dim: 0, lower bound: -0.6609718, upper bound: 0.6397003
IS_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.23
Output dim: 0, lower bound: -0.6779333, upper bound: 0.6973597
IS_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.23
Output dim: 0, lower bound: -0.6779333, upper bound: 0.6973597
IS_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.23
Output dim: 0, lower bound: -0.6851157, upper bound: 0.6977845
IS_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.23
Output dim: 0, lower bound: -0.6851157, upper bound: 0.6977845

## BFS IS instance: IS_B1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0017378, 0.9577341, 0.0209117, 0.9025295, -0.9042673, 0.9368224
1: -0.1117451, 1.2035816, -0.0776985, 1.1409044, -1.2526495, 1.2812800
2: -0.0625789, 1.0695953, -0.0370891, 0.9958014, -1.0583804, 1.1066844
3: -0.2743888, 1.0868173, -0.2410901, 1.0262597, -1.3006485, 1.3279073
4: -0.2543875, 0.9355542, -0.2204622, 0.8550285, -1.1094160, 1.1560163

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_B1_A1_B1_A1

### Relational analysis result of IS_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6404941, upper bound: 0.6802498
time: 0.32 seconds

## Relational analysis of IS_B1_B1_A1_B1_A2

### Relational analysis result of IS_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6404941, upper bound: 0.6809969
time: 0.32 seconds

## BFS IS instance: IS_B1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0004685, 0.9435406, -0.0032020, 1.0339718, -1.0335033, 0.9467425
1: -0.1086941, 1.1876776, -0.1091664, 1.3081079, -1.4168019, 1.2968440
2: -0.0600665, 1.0513515, -0.0733573, 1.1349654, -1.1950319, 1.1247089
3: -0.2708709, 1.0741191, -0.2708402, 1.1731939, -1.4440649, 1.3449593
4: -0.2502931, 0.9241737, -0.2933947, 0.9566256, -1.2069187, 1.2175684

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_B1_A1_B2_A1

### Relational analysis result of IS_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6400006, upper bound: 0.6714281
time: 0.36 seconds

## Relational analysis of IS_B1_B1_A1_B2_A2

### Relational analysis result of IS_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6415344, upper bound: 0.6637412
time: 0.35 seconds

## BFS IS instance: IS_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0107870, 0.6989747, -0.0004439, 0.9531150, -0.9639020, 0.6994187
1: -0.1241605, 0.8672397, -0.1101191, 1.1983986, -1.3225591, 0.9773588
2: -0.0573094, 0.8610535, -0.0610516, 1.0641370, -1.1214464, 0.9221051
3: -0.2815993, 0.8504646, -0.2731256, 1.0821323, -1.3637316, 1.1235902
4: -0.2244935, 0.8760955, -0.2521458, 0.9308868, -1.1553802, 1.1282413

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_B1_A2_A1_B1

### Relational analysis result of IS_B1_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6397003, upper bound: 0.6609718
time: 0.31 seconds

## Relational analysis of IS_B1_B1_A2_A1_B2

### Relational analysis result of IS_B1_B1_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6397003, upper bound: 0.6609718
time: 0.35 seconds

## BFS IS instance: IS_B1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0101330, 0.6956245, -0.0001364, 0.9854338, -0.9955668, 0.6957608
1: -0.1233530, 0.8634801, -0.1093853, 1.2390206, -1.3623736, 0.9728653
2: -0.0566123, 0.8572230, -0.0645683, 1.0958974, -1.1525097, 0.9217913
3: -0.2808638, 0.8473691, -0.2715011, 1.1123974, -1.3932612, 1.1188701
4: -0.2232363, 0.8732710, -0.2628224, 0.9425689, -1.1658052, 1.1360934

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_B2_A2_A1_B1

### Relational analysis result of IS_B1_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6397003, upper bound: 0.6411668
time: 0.32 seconds

## Relational analysis of IS_B1_B2_A2_A1_B2

### Relational analysis result of IS_B1_B2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6438791, upper bound: 0.6427007
time: 0.31 seconds

## BFS IS instance: IS_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0004439, 0.9531150, -0.0107870, 0.6989747, -0.6994187, 0.9639020
1: -0.1101191, 1.1983986, -0.1241605, 0.8672397, -0.9773588, 1.3225591
2: -0.0610516, 1.0641370, -0.0573094, 0.8610535, -0.9221051, 1.1214464
3: -0.2731256, 1.0821323, -0.2815993, 0.8504646, -1.1235902, 1.3637316
4: -0.2521458, 0.9308868, -0.2244935, 0.8760955, -1.1282413, 1.1553802

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_A1_B1_A1

### Relational analysis result of IS_B2_A1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6609718, upper bound: 0.6397003
time: 0.29 seconds

## Relational analysis of IS_B2_A1_A1_B1_A2

### Relational analysis result of IS_B2_A1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6609718, upper bound: 0.6397003
time: 0.30 seconds

## BFS IS instance: IS_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0054424, 0.6499398, 0.0054424, 0.6499398, -0.6444974, 0.6444974
1: -0.1022949, 0.8109143, -0.1022949, 0.8109143, -0.9132092, 0.9132092
2: -0.0393627, 0.7986160, -0.0393627, 0.7986160, -0.8379787, 0.8379787
3: -0.2590857, 0.7998861, -0.2590857, 0.7998861, -1.0589718, 1.0589718
4: -0.1939640, 0.8231770, -0.1939640, 0.8231770, -1.0171410, 1.0171410

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_B1_A1_A1

### Relational analysis result of IS_B2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6648599, upper bound: 0.6813945
time: 0.34 seconds

## Relational analysis of IS_B2_A2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5

Time for candidate selection: 3.69 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B1_A1_A1

### Relational analysis result of IS_B2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6825935, upper bound: 0.6647563
time: 0.32 seconds

## Relational analysis of IS_B2_A2_B1_A1_A2

### Relational analysis result of IS_B2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6615940, upper bound: 0.6729556
time: 0.34 seconds

## BFS IS instance: IS_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0004525, 0.7221838, 0.0054424, 0.6499398, -0.6494873, 0.7167414
1: -0.1048932, 0.8939738, -0.1022949, 0.8109143, -0.9158075, 0.9962687
2: -0.0500441, 0.8598962, -0.0393627, 0.7986160, -0.8486601, 0.8992589
3: -0.2612641, 0.8577256, -0.2590857, 0.7998861, -1.0611502, 1.1168113
4: -0.2176540, 0.8560207, -0.1939640, 0.8231770, -1.0408310, 1.0499847

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_B1_A2_A1

### Relational analysis result of IS_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6648599, upper bound: 0.6813945
time: 0.33 seconds

## Relational analysis of IS_B2_A2_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 5

Time for candidate selection: 3.59 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B1_A2_B1

### Relational analysis result of IS_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6341388, upper bound: 0.6966600
time: 0.35 seconds

## Relational analysis of IS_B2_A2_B1_A2_B2

### Relational analysis result of IS_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6615940, upper bound: 0.6729556
time: 0.35 seconds

## BFS IS instance: IS_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0054424, 0.6499398, 0.0004525, 0.7221838, -0.7167414, 0.6494873
1: -0.1022949, 0.8109143, -0.1048932, 0.8939738, -0.9962687, 0.9158075
2: -0.0393627, 0.7986160, -0.0500441, 0.8598962, -0.8992589, 0.8486601
3: -0.2590857, 0.7998861, -0.2612641, 0.8577256, -1.1168113, 1.0611502
4: -0.1939640, 0.8231770, -0.2176540, 0.8560207, -1.0499847, 1.0408310

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5

Time for candidate selection: 3.37 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B2_A1_A1

### Relational analysis result of IS_B2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6825935, upper bound: 0.6640375
time: 0.36 seconds

## Relational analysis of IS_B2_A2_B2_A1_A2

### Relational analysis result of IS_B2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6530386, upper bound: 0.6755661
time: 0.37 seconds

## BFS IS instance: IS_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0004525, 0.7221838, 0.0004525, 0.7221838, -0.7217313, 0.7217313
1: -0.1048932, 0.8939738, -0.1048932, 0.8939738, -0.9988670, 0.9988670
2: -0.0500441, 0.8598962, -0.0500441, 0.8598962, -0.9099402, 0.9099402
3: -0.2612641, 0.8577256, -0.2612641, 0.8577256, -1.1189897, 1.1189897
4: -0.2176540, 0.8560207, -0.2176540, 0.8560207, -1.0736747, 1.0736747

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5

Time for candidate selection: 3.28 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B2_A2_A1

### Relational analysis result of IS_B2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6825935, upper bound: 0.6645855
time: 0.36 seconds

## Relational analysis of IS_B2_A2_B2_A2_A2

### Relational analysis result of IS_B2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6615940, upper bound: 0.6720749
time: 0.33 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 5.58 seconds
IS_B1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.58
Output dim: 0, lower bound: -0.6404941, upper bound: 0.6802498
IS_B1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.58
Output dim: 0, lower bound: -0.6404941, upper bound: 0.6809969
IS_B1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.58
Output dim: 0, lower bound: -0.6400006, upper bound: 0.6714281
IS_B1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.58
Output dim: 0, lower bound: -0.6415344, upper bound: 0.6637412
IS_B1_B1_A2_A1_B1, status: Status.VERIFIED, split count: 5, time: 5.58
Output dim: 0, lower bound: -0.6397003, upper bound: 0.6609718
IS_B1_B1_A2_A1_B2, status: Status.VERIFIED, split count: 5, time: 5.58
Output dim: 0, lower bound: -0.6397003, upper bound: 0.6609718
IS_B1_B2_A2_A1_B1, status: Status.VERIFIED, split count: 5, time: 5.58
Output dim: 0, lower bound: -0.6397003, upper bound: 0.6411668
IS_B1_B2_A2_A1_B2, status: Status.VERIFIED, split count: 5, time: 5.58
Output dim: 0, lower bound: -0.6438791, upper bound: 0.6427007
IS_B2_A1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 5.58
Output dim: 0, lower bound: -0.6609718, upper bound: 0.6397003
IS_B2_A1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 5.58
Output dim: 0, lower bound: -0.6609718, upper bound: 0.6397003
IS_B2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 5.58
Output dim: 0, lower bound: -0.6825935, upper bound: 0.6647563
IS_B2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 5.58
Output dim: 0, lower bound: -0.6615940, upper bound: 0.6729556
IS_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.58
Output dim: 0, lower bound: -0.6341388, upper bound: 0.6966600
IS_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.58
Output dim: 0, lower bound: -0.6615940, upper bound: 0.6729556
IS_B2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 5.58
Output dim: 0, lower bound: -0.6825935, upper bound: 0.6640375
IS_B2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 5.58
Output dim: 0, lower bound: -0.6530386, upper bound: 0.6755661
IS_B2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 5.58
Output dim: 0, lower bound: -0.6825935, upper bound: 0.6645855
IS_B2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 5.58
Output dim: 0, lower bound: -0.6615940, upper bound: 0.6720749

## BFS IS instance: IS_B1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004439, 0.9531150, 0.0209117, 0.9025295, -0.9029734, 0.9322033
1: -0.1101191, 1.1983986, -0.0776985, 1.1409044, -1.2510235, 1.2760971
2: -0.0610516, 1.0641370, -0.0370891, 0.9958014, -1.0568531, 1.1012261
3: -0.2731256, 1.0821323, -0.2410901, 1.0262597, -1.2993853, 1.3232224
4: -0.2521458, 0.9308868, -0.2204622, 0.8550285, -1.1071743, 1.1513491

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_B1_A1_B1_A1_B1

### Relational analysis result of IS_B1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6392813, upper bound: 0.6689632
time: 0.31 seconds

## Relational analysis of IS_B1_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 5

Time for candidate selection: 3.12 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B1_A1_B1

### Relational analysis result of IS_B1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6339897, upper bound: 0.6795986
time: 0.35 seconds

## Relational analysis of IS_B1_B1_A1_B1_A1_B2

### Relational analysis result of IS_B1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6400722, upper bound: 0.6516630
time: 0.33 seconds

## BFS IS instance: IS_B1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0001364, 0.9854338, 0.0209117, 0.9025295, -0.9026659, 0.9645221
1: -0.1093853, 1.2390206, -0.0776985, 1.1409044, -1.2502897, 1.3167191
2: -0.0645683, 1.0958974, -0.0370891, 0.9958014, -1.0603697, 1.1329865
3: -0.2715011, 1.1123974, -0.2410901, 1.0262597, -1.2977607, 1.3534875
4: -0.2628224, 0.9425689, -0.2204622, 0.8550285, -1.1178509, 1.1630311

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_B1_A1_B1_A2_B1

### Relational analysis result of IS_B1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6392813, upper bound: 0.6731420
time: 0.30 seconds

## Relational analysis of IS_B1_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 5

Time for candidate selection: 3.05 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B1_A2_B1

### Relational analysis result of IS_B1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6339897, upper bound: 0.6804002
time: 0.31 seconds

## Relational analysis of IS_B1_B1_A1_B1_A2_B2

### Relational analysis result of IS_B1_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6400722, upper bound: 0.6524646
time: 0.33 seconds

## BFS IS instance: IS_B1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0022480, 0.9381881, -0.0032020, 1.0339718, -1.0317237, 0.9413900
1: -0.1064734, 1.1815870, -0.1091664, 1.3081079, -1.4145813, 1.2907534
2: -0.0581751, 1.0449984, -0.0733573, 1.1349654, -1.1931405, 1.1183558
3: -0.2690332, 1.0688602, -0.2708402, 1.1731939, -1.4422271, 1.3397003
4: -0.2479466, 0.9189078, -0.2933947, 0.9566256, -1.2045722, 1.2123024

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_B1_A1_B2_A1_B1

### Relational analysis result of IS_B1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6400006, upper bound: 0.6637412
time: 0.33 seconds

## Relational analysis of IS_B1_B1_A1_B2_A1_B2

### Relational analysis result of IS_B1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6400006, upper bound: 0.6637412
time: 0.33 seconds

## BFS IS instance: IS_B1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0460036, 1.8619289, -0.0026793, 1.0310004, -1.0770040, 1.8646083
1: -0.1686013, 2.2772894, -0.1085286, 1.3047872, -1.4733884, 2.3858180
2: -0.1385605, 2.0437284, -0.0727286, 1.1315031, -1.2700636, 2.1164570
3: -0.3297453, 1.8520892, -0.2702293, 1.1705358, -1.5002811, 2.1223185
4: -0.4871181, 1.4866383, -0.2925251, 0.9539694, -1.4410875, 1.7791634

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_B1_A1_B2_A2_B1

### Relational analysis result of IS_B1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6415345, upper bound: 0.6637412
time: 0.32 seconds

## Relational analysis of IS_B1_B1_A1_B2_A2_B2

### Relational analysis result of IS_B1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6415345, upper bound: 0.6637412
time: 0.32 seconds

## BFS IS instance: IS_B2_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, 0.0054424, 0.6499398, -0.6387517, 0.5594192
1: -0.0944340, 0.7130072, -0.1022949, 0.8109143, -0.9053483, 0.8153021
2: -0.0312903, 0.7124612, -0.0393627, 0.7986160, -0.8299063, 0.7518239
3: -0.2500148, 0.7301772, -0.2590857, 0.7998861, -1.0499009, 0.9892629
4: -0.1787794, 0.7720095, -0.1939640, 0.8231770, -1.0019563, 0.9659735

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B1_A1_A1_B1

### Relational analysis result of IS_B2_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6657280
time: 0.37 seconds

## Relational analysis of IS_B2_A2_B1_A1_A1_B2

### Relational analysis result of IS_B2_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6657280
time: 0.37 seconds

## BFS IS instance: IS_B2_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, 0.0054424, 0.6499398, -0.6805067, 0.6652158
1: -0.1440713, 0.8347254, -0.1022949, 0.8109143, -0.9549856, 0.9370203
2: -0.0742025, 0.8280591, -0.0393627, 0.7986160, -0.8728185, 0.8674218
3: -0.2901301, 0.8312570, -0.2590857, 0.7998861, -1.0900162, 1.0903428
4: -0.2451730, 0.8507983, -0.1939640, 0.8231770, -1.0683500, 1.0447624

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 29

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B1_A1_A2_B1

### Relational analysis result of IS_B2_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6758026
time: 0.38 seconds

## Relational analysis of IS_B2_A2_B1_A1_A2_B2

### Relational analysis result of IS_B2_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6758026
time: 0.36 seconds

## BFS IS instance: IS_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0004525, 0.7221838, 0.0111880, 0.5648615, -0.5644090, 0.7109958
1: -0.1048932, 0.8939738, -0.0944340, 0.7130072, -0.8179004, 0.9884079
2: -0.0500441, 0.8598962, -0.0312903, 0.7124612, -0.7625053, 0.8911865
3: -0.2612641, 0.8577256, -0.2500148, 0.7301772, -0.9914413, 1.1077404
4: -0.2176540, 0.8560207, -0.1787794, 0.7720095, -0.9896635, 1.0348001

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6529501, upper bound: 0.6647563
time: 0.36 seconds

## Relational analysis of IS_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6529501, upper bound: 0.6729556
time: 0.35 seconds

## BFS IS instance: IS_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0004525, 0.7221838, -0.0305669, 0.6706582, -0.6702057, 0.7527508
1: -0.1048932, 0.8939738, -0.1440713, 0.8347254, -0.9396186, 1.0380452
2: -0.0500441, 0.8598962, -0.0742025, 0.8280591, -0.8781032, 0.9340987
3: -0.2612641, 0.8577256, -0.2901301, 0.8312570, -1.0925212, 1.1478558
4: -0.2176540, 0.8560207, -0.2451730, 0.8507983, -1.0684524, 1.1011937

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 29

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6615940, upper bound: 0.6647563
time: 0.34 seconds

## Relational analysis of IS_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6615940, upper bound: 0.6729556
time: 0.38 seconds

## BFS IS instance: IS_B2_A2_B2_A1_A1

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, 0.0004525, 0.7221838, -0.7109958, 0.5644090
1: -0.0944340, 0.7130072, -0.1048932, 0.8939738, -0.9884079, 0.8179004
2: -0.0312903, 0.7124612, -0.0500441, 0.8598962, -0.8911865, 0.7625053
3: -0.2500148, 0.7301772, -0.2612641, 0.8577256, -1.1077404, 0.9914413
4: -0.1787794, 0.7720095, -0.2176540, 0.8560207, -1.0348001, 0.9896635

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B2_A1_A1_B1

### Relational analysis result of IS_B2_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6599371, upper bound: 0.6677968
time: 0.35 seconds

## Relational analysis of IS_B2_A2_B2_A1_A1_B2

### Relational analysis result of IS_B2_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6599371, upper bound: 0.6677968
time: 0.35 seconds

## BFS IS instance: IS_B2_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, 0.0004525, 0.7221838, -0.7527508, 0.6702057
1: -0.1440713, 0.8347254, -0.1048932, 0.8939738, -1.0380452, 0.9396186
2: -0.0742025, 0.8280591, -0.0500441, 0.8598962, -0.9340987, 0.8781032
3: -0.2901301, 0.8312570, -0.2612641, 0.8577256, -1.1478558, 1.0925212
4: -0.2451730, 0.8507983, -0.2176540, 0.8560207, -1.1011937, 1.0684524

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 29

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B2_A1_A2_B1

### Relational analysis result of IS_B2_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6599371, upper bound: 0.6757891
time: 0.38 seconds

## Relational analysis of IS_B2_A2_B2_A1_A2_B2

### Relational analysis result of IS_B2_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6599371, upper bound: 0.6757891
time: 0.37 seconds

## BFS IS instance: IS_B2_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: 0.0061345, 0.6507710, 0.0004525, 0.7221838, -0.7160493, 0.6503185
1: -0.0976210, 0.8106803, -0.1048932, 0.8939738, -0.9915948, 0.9155735
2: -0.0417109, 0.7880348, -0.0500441, 0.8598962, -0.9016070, 0.8380789
3: -0.2534082, 0.7965333, -0.2612641, 0.8577256, -1.1111338, 1.0577974
4: -0.2023642, 0.8128391, -0.2176540, 0.8560207, -1.0583849, 1.0304930

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B2_A2_A1_B1

### Relational analysis result of IS_B2_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6530565, upper bound: 0.6645855
time: 0.38 seconds

## Relational analysis of IS_B2_A2_B2_A2_A1_B2

### Relational analysis result of IS_B2_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6530565, upper bound: 0.6645855
time: 0.39 seconds

## BFS IS instance: IS_B2_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0329778, 0.7304660, 0.0004525, 0.7221838, -0.7551616, 0.7300135
1: -0.1447692, 0.9029138, -0.1048932, 0.8939738, -1.0387430, 1.0078070
2: -0.0813575, 0.8816222, -0.0500441, 0.8598962, -0.9412537, 0.9316663
3: -0.2928538, 0.8801655, -0.2612641, 0.8577256, -1.1505795, 1.1414295
4: -0.2609611, 0.8848863, -0.2176540, 0.8560207, -1.1169817, 1.1025403

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 29

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B2_A2_A2_B1

### Relational analysis result of IS_B2_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6530562, upper bound: 0.6752281
time: 0.39 seconds

## Relational analysis of IS_B2_A2_B2_A2_A2_B2

### Relational analysis result of IS_B2_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6530562, upper bound: 0.6752281
time: 0.36 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.55 seconds
IS_B1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -0.6339897, upper bound: 0.6795986
IS_B1_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 0, lower bound: -0.6400722, upper bound: 0.6516630
IS_B1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -0.6339897, upper bound: 0.6804002
IS_B1_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.55
Output dim: 0, lower bound: -0.6400722, upper bound: 0.6524646
IS_B1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -0.6400006, upper bound: 0.6637412
IS_B1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -0.6400006, upper bound: 0.6637412
IS_B1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -0.6415345, upper bound: 0.6637412
IS_B1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -0.6415345, upper bound: 0.6637412
IS_B2_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6657280
IS_B2_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6657280
IS_B2_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6758026
IS_B2_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6758026
IS_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -0.6529501, upper bound: 0.6647563
IS_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -0.6529501, upper bound: 0.6729556
IS_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -0.6615940, upper bound: 0.6647563
IS_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -0.6615940, upper bound: 0.6729556
IS_B2_A2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -0.6599371, upper bound: 0.6677968
IS_B2_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -0.6599371, upper bound: 0.6677968
IS_B2_A2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -0.6599371, upper bound: 0.6757891
IS_B2_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -0.6599371, upper bound: 0.6757891
IS_B2_A2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -0.6530565, upper bound: 0.6645855
IS_B2_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -0.6530565, upper bound: 0.6645855
IS_B2_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -0.6530562, upper bound: 0.6752281
IS_B2_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.55
Output dim: 0, lower bound: -0.6530562, upper bound: 0.6752281

## BFS IS instance: IS_B1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0004439, 0.9531150, 0.0258129, 0.8220325, -0.8224764, 0.9273021
1: -0.1101191, 1.1983986, -0.0706267, 1.0456822, -1.1558013, 1.2690253
2: -0.0610516, 1.0641370, -0.0294187, 0.9115350, -0.9725866, 1.0935557
3: -0.2731256, 1.0821323, -0.2329345, 0.9584060, -1.2315316, 1.3150668
4: -0.2521458, 0.9308868, -0.2036403, 0.8138722, -1.0660180, 1.1345272

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_B1_B1_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6446778, upper bound: 0.6359980
time: 0.29 seconds

## Relational analysis of IS_B1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_B1_B1_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6446778, upper bound: 0.6516630
time: 0.35 seconds

## BFS IS instance: IS_B1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0001364, 0.9854338, 0.0258129, 0.8220325, -0.8221688, 0.9596210
1: -0.1093853, 1.2390206, -0.0706267, 1.0456822, -1.1550674, 1.3096473
2: -0.0645683, 1.0958974, -0.0294187, 0.9115350, -0.9761033, 1.1253161
3: -0.2715011, 1.1123974, -0.2329345, 0.9584060, -1.2299070, 1.3453319
4: -0.2628224, 0.9425689, -0.2036403, 0.8138722, -1.0766946, 1.1462092

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 47

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_B1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6335183, upper bound: 0.6735472
time: 0.34 seconds

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_B1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6197446, upper bound: 0.6747169
time: 0.31 seconds

## BFS IS instance: IS_B1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0022480, 0.9381881, -0.0014365, 1.0292757, -1.0270276, 0.9396245
1: -0.1064734, 1.1815870, -0.1070645, 1.3027349, -1.4092083, 1.2886515
2: -0.0581751, 1.0449984, -0.0715175, 1.1294107, -1.1875858, 1.1165159
3: -0.2690332, 1.0688602, -0.2690842, 1.1685212, -1.4375544, 1.3379444
4: -0.2479466, 0.9189078, -0.2915230, 0.9521359, -1.2000825, 1.2104307

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_B1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_B1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6400006, upper bound: 0.6706810
time: 0.33 seconds

## Relational analysis of IS_B1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_B1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6400006, upper bound: 0.6714281
time: 0.34 seconds

## BFS IS instance: IS_B1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0022480, 0.9381881, -0.0357146, 1.8917470, -1.8894989, 0.9739027
1: -0.1064734, 1.1815870, -0.1529815, 2.3290219, -2.4354954, 1.3345685
2: -0.0581751, 1.0449984, -0.1319740, 2.0703101, -2.1284852, 1.1769724
3: -0.2690332, 1.0688602, -0.3186724, 1.8925552, -2.1615884, 1.3875326
4: -0.2479466, 0.9189078, -0.5038834, 1.4782946, -1.7262412, 1.4227911

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_B1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6400006, upper bound: 0.6706810
time: 0.35 seconds

## Relational analysis of IS_B1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_B1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6400006, upper bound: 0.6714281
time: 0.33 seconds

## BFS IS instance: IS_B1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0460036, 1.8619289, -0.0014365, 1.0292757, -1.0752792, 1.8633654
1: -0.1686013, 2.2772894, -0.1070645, 1.3027349, -1.4713361, 2.3843539
2: -0.1385605, 2.0437284, -0.0715175, 1.1294107, -1.2679713, 2.1152458
3: -0.3297453, 1.8520892, -0.2690842, 1.1685212, -1.4982665, 2.1211734
4: -0.4871181, 1.4866383, -0.2915230, 0.9521359, -1.4392540, 1.7781613

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_B1_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6415344, upper bound: 0.6595624
time: 0.34 seconds

## Relational analysis of IS_B1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_B1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6415344, upper bound: 0.6637412
time: 0.34 seconds

## BFS IS instance: IS_B1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0460036, 1.8619289, -0.0357146, 1.8917470, -1.9377506, 1.8976436
1: -0.1686013, 2.2772894, -0.1529815, 2.3290219, -2.4976232, 2.4302709
2: -0.1385605, 2.0437284, -0.1319740, 2.0703101, -2.2088706, 2.1757023
3: -0.3297453, 1.8520892, -0.3186724, 1.8925552, -2.2223005, 2.1707616
4: -0.4871181, 1.4866383, -0.5038834, 1.4782946, -1.9654127, 1.9905217

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_B1_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6415345, upper bound: 0.6595624
time: 0.34 seconds

## Relational analysis of IS_B1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_B1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6415345, upper bound: 0.6637412
time: 0.34 seconds

## BFS IS instance: IS_B2_A2_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, 0.0111880, 0.5648615, -0.5536735, 0.5536735
1: -0.0944340, 0.7130072, -0.0944340, 0.7130072, -0.8074412, 0.8074412
2: -0.0312903, 0.7124612, -0.0312903, 0.7124612, -0.7437515, 0.7437515
3: -0.2500148, 0.7301772, -0.2500148, 0.7301772, -0.9801920, 0.9801920
4: -0.1787794, 0.7720095, -0.1787794, 0.7720095, -0.9507889, 0.9507889

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B1_A1_A1_B1_A1

### Relational analysis result of IS_B2_A2_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6826392, upper bound: 0.6657280
time: 0.40 seconds

## Relational analysis of IS_B2_A2_B1_A1_A1_B1_A2

### Relational analysis result of IS_B2_A2_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6657280
time: 0.37 seconds

## BFS IS instance: IS_B2_A2_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, -0.0305669, 0.6706582, -0.6594702, 0.5954285
1: -0.0944340, 0.7130072, -0.1440713, 0.8347254, -0.9291594, 0.8570786
2: -0.0312903, 0.7124612, -0.0742025, 0.8280591, -0.8593494, 0.7866638
3: -0.2500148, 0.7301772, -0.2901301, 0.8312570, -1.0812719, 1.0203073
4: -0.1787794, 0.7720095, -0.2451730, 0.8507983, -1.0295777, 1.0171825

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B1_A1_A1_B2_A1

### Relational analysis result of IS_B2_A2_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6909598, upper bound: 0.6657280
time: 0.37 seconds

## Relational analysis of IS_B2_A2_B1_A1_A1_B2_A2

### Relational analysis result of IS_B2_A2_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6700552, upper bound: 0.6657280
time: 0.36 seconds

## BFS IS instance: IS_B2_A2_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, 0.0111880, 0.5648615, -0.5954285, 0.6594702
1: -0.1440713, 0.8347254, -0.0944340, 0.7130072, -0.8570786, 0.9291594
2: -0.0742025, 0.8280591, -0.0312903, 0.7124612, -0.7866638, 0.8593494
3: -0.2901301, 0.8312570, -0.2500148, 0.7301772, -1.0203073, 1.0812719
4: -0.2451730, 0.8507983, -0.1787794, 0.7720095, -1.0171825, 1.0295777

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 29

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B1_A1_A2_B1_A1

### Relational analysis result of IS_B2_A2_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6657280
time: 0.39 seconds

## Relational analysis of IS_B2_A2_B1_A1_A2_B1_A2

### Relational analysis result of IS_B2_A2_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6758026
time: 0.36 seconds

## BFS IS instance: IS_B2_A2_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, -0.0305669, 0.6706582, -0.7012252, 0.7012252
1: -0.1440713, 0.8347254, -0.1440713, 0.8347254, -0.9787967, 0.9787967
2: -0.0742025, 0.8280591, -0.0742025, 0.8280591, -0.9022617, 0.9022617
3: -0.2901301, 0.8312570, -0.2901301, 0.8312570, -1.1213872, 1.1213872
4: -0.2451730, 0.8507983, -0.2451730, 0.8507983, -1.0959713, 1.0959713

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 29

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B1_A1_A2_B2_A1

### Relational analysis result of IS_B2_A2_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6700552, upper bound: 0.6657280
time: 0.35 seconds

## Relational analysis of IS_B2_A2_B1_A1_A2_B2_A2

### Relational analysis result of IS_B2_A2_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6700552, upper bound: 0.6758026
time: 0.37 seconds

## BFS IS instance: IS_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0061345, 0.6507710, 0.0111880, 0.5648615, -0.5587270, 0.6395830
1: -0.0976210, 0.8106803, -0.0944340, 0.7130072, -0.8106282, 0.9051143
2: -0.0417109, 0.7880348, -0.0312903, 0.7124612, -0.7541721, 0.8193251
3: -0.2534082, 0.7965333, -0.2500148, 0.7301772, -0.9835854, 1.0465481
4: -0.2023642, 0.8128391, -0.1787794, 0.7720095, -0.9743737, 0.9916185

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6529501, upper bound: 0.6886226
time: 0.37 seconds

## Relational analysis of IS_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6529501, upper bound: 0.6647563
time: 0.40 seconds

## BFS IS instance: IS_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0329778, 0.7304660, 0.0111880, 0.5648615, -0.5978394, 0.7192780
1: -0.1447692, 0.9029138, -0.0944340, 0.7130072, -0.8577764, 0.9973478
2: -0.0813575, 0.8816222, -0.0312903, 0.7124612, -0.7938187, 0.9129125
3: -0.2928538, 0.8801655, -0.2500148, 0.7301772, -1.0230310, 1.1301802
4: -0.2609611, 0.8848863, -0.1787794, 0.7720095, -1.0329705, 1.0636656

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6442375, upper bound: 0.6966600
time: 0.37 seconds

## Relational analysis of IS_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6529501, upper bound: 0.6729556
time: 0.36 seconds

## BFS IS instance: IS_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0061345, 0.6507710, -0.0305669, 0.6706582, -0.6645237, 0.6813380
1: -0.0976210, 0.8106803, -0.1440713, 0.8347254, -0.9323463, 0.9547516
2: -0.0417109, 0.7880348, -0.0742025, 0.8280591, -0.8697700, 0.8622373
3: -0.2534082, 0.7965333, -0.2901301, 0.8312570, -1.0846653, 1.0866635
4: -0.2023642, 0.8128391, -0.2451730, 0.8507983, -1.0531626, 1.0580120

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6529501, upper bound: 0.6647563
time: 0.39 seconds

## Relational analysis of IS_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6529501, upper bound: 0.6647563
time: 0.35 seconds

## BFS IS instance: IS_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0329778, 0.7304660, -0.0305669, 0.6706582, -0.7036361, 0.7610329
1: -0.1447692, 0.9029138, -0.1440713, 0.8347254, -0.9794946, 1.0469851
2: -0.0813575, 0.8816222, -0.0742025, 0.8280591, -0.9094166, 0.9558247
3: -0.2928538, 0.8801655, -0.2901301, 0.8312570, -1.1241109, 1.1702956
4: -0.2609611, 0.8848863, -0.2451730, 0.8507983, -1.1117594, 1.1300592

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6529501, upper bound: 0.6729556
time: 0.36 seconds

## Relational analysis of IS_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6615940, upper bound: 0.6729556
time: 0.41 seconds

## BFS IS instance: IS_B2_A2_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, 0.0061345, 0.6507710, -0.6395830, 0.5587270
1: -0.0944340, 0.7130072, -0.0976210, 0.8106803, -0.9051143, 0.8106282
2: -0.0312903, 0.7124612, -0.0417109, 0.7880348, -0.8193251, 0.7541721
3: -0.2500148, 0.7301772, -0.2534082, 0.7965333, -1.0465481, 0.9835854
4: -0.1787794, 0.7720095, -0.2023642, 0.8128391, -0.9916185, 0.9743737

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B2_A1_A1_B1_A1

### Relational analysis result of IS_B2_A2_B2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6825553, upper bound: 0.6677968
time: 0.38 seconds

## Relational analysis of IS_B2_A2_B2_A1_A1_B1_A2

### Relational analysis result of IS_B2_A2_B2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6599371, upper bound: 0.6677968
time: 0.35 seconds

## BFS IS instance: IS_B2_A2_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, -0.0329778, 0.7304660, -0.7192780, 0.5978394
1: -0.0944340, 0.7130072, -0.1447692, 0.9029138, -0.9973478, 0.8577764
2: -0.0312903, 0.7124612, -0.0813575, 0.8816222, -0.9129125, 0.7938187
3: -0.2500148, 0.7301772, -0.2928538, 0.8801655, -1.1301802, 1.0230310
4: -0.1787794, 0.7720095, -0.2609611, 0.8848863, -1.0636656, 1.0329705

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B2_A1_A1_B2_A1

### Relational analysis result of IS_B2_A2_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6909142, upper bound: 0.6646636
time: 0.36 seconds

## Relational analysis of IS_B2_A2_B2_A1_A1_B2_A2

### Relational analysis result of IS_B2_A2_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6687333, upper bound: 0.6677968
time: 0.38 seconds

## BFS IS instance: IS_B2_A2_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, 0.0061345, 0.6507710, -0.6813380, 0.6645237
1: -0.1440713, 0.8347254, -0.0976210, 0.8106803, -0.9547516, 0.9323463
2: -0.0742025, 0.8280591, -0.0417109, 0.7880348, -0.8622373, 0.8697700
3: -0.2901301, 0.8312570, -0.2534082, 0.7965333, -1.0866635, 1.0846653
4: -0.2451730, 0.8507983, -0.2023642, 0.8128391, -1.0580120, 1.0531626

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B2_A1_A2_B1_A1

### Relational analysis result of IS_B2_A2_B2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6599371, upper bound: 0.6677968
time: 0.38 seconds

## Relational analysis of IS_B2_A2_B2_A1_A2_B1_A2

### Relational analysis result of IS_B2_A2_B2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6599371, upper bound: 0.6757891
time: 0.36 seconds

## BFS IS instance: IS_B2_A2_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, -0.0329778, 0.7304660, -0.7610329, 0.7036361
1: -0.1440713, 0.8347254, -0.1447692, 0.9029138, -1.0469851, 0.9794946
2: -0.0742025, 0.8280591, -0.0813575, 0.8816222, -0.9558247, 0.9094166
3: -0.2901301, 0.8312570, -0.2928538, 0.8801655, -1.1702956, 1.1241109
4: -0.2451730, 0.8507983, -0.2609611, 0.8848863, -1.1300592, 1.1117594

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B2_A1_A2_B2_A1

### Relational analysis result of IS_B2_A2_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6687333, upper bound: 0.6677968
time: 0.37 seconds

## Relational analysis of IS_B2_A2_B2_A1_A2_B2_A2

### Relational analysis result of IS_B2_A2_B2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6687333, upper bound: 0.6757891
time: 0.37 seconds

## BFS IS instance: IS_B2_A2_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0061345, 0.6507710, 0.0061345, 0.6507710, -0.6446365, 0.6446365
1: -0.0976210, 0.8106803, -0.0976210, 0.8106803, -0.9083012, 0.9083012
2: -0.0417109, 0.7880348, -0.0417109, 0.7880348, -0.8297457, 0.8297457
3: -0.2534082, 0.7965333, -0.2534082, 0.7965333, -1.0499415, 1.0499415
4: -0.2023642, 0.8128391, -0.2023642, 0.8128391, -1.0152032, 1.0152032

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B2_A2_A1_B1_A1

### Relational analysis result of IS_B2_A2_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6736765, upper bound: 0.6645855
time: 0.37 seconds

## Relational analysis of IS_B2_A2_B2_A2_A1_B1_A2

### Relational analysis result of IS_B2_A2_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6530565, upper bound: 0.6645855
time: 0.35 seconds

## BFS IS instance: IS_B2_A2_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0061345, 0.6507710, -0.0329778, 0.7304660, -0.7243315, 0.6837488
1: -0.0976210, 0.8106803, -0.1447692, 0.9029138, -1.0005348, 0.9554495
2: -0.0417109, 0.7880348, -0.0813575, 0.8816222, -0.9233330, 0.8693923
3: -0.2534082, 0.7965333, -0.2928538, 0.8801655, -1.1335737, 1.0893872
4: -0.2023642, 0.8128391, -0.2609611, 0.8848863, -1.0872505, 1.0738001

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B2_A2_A1_B2_A1

### Relational analysis result of IS_B2_A2_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6825935, upper bound: 0.6645855
time: 0.38 seconds

## Relational analysis of IS_B2_A2_B2_A2_A1_B2_A2

### Relational analysis result of IS_B2_A2_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6631956, upper bound: 0.6645855
time: 0.43 seconds

## BFS IS instance: IS_B2_A2_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0329778, 0.7304660, 0.0061345, 0.6507710, -0.6837488, 0.7243315
1: -0.1447692, 0.9029138, -0.0976210, 0.8106803, -0.9554495, 1.0005348
2: -0.0813575, 0.8816222, -0.0417109, 0.7880348, -0.8693923, 0.9233330
3: -0.2928538, 0.8801655, -0.2534082, 0.7965333, -1.0893872, 1.1335737
4: -0.2609611, 0.8848863, -0.2023642, 0.8128391, -1.0738001, 1.0872505

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B2_A2_A2_B1_A1

### Relational analysis result of IS_B2_A2_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6530565, upper bound: 0.6645855
time: 0.37 seconds

## Relational analysis of IS_B2_A2_B2_A2_A2_B1_A2

### Relational analysis result of IS_B2_A2_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6530562, upper bound: 0.6752280
time: 0.37 seconds

## BFS IS instance: IS_B2_A2_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0329778, 0.7304660, -0.0329778, 0.7304660, -0.7634438, 0.7634438
1: -0.1447692, 0.9029138, -0.1447692, 0.9029138, -1.0476830, 1.0476830
2: -0.0813575, 0.8816222, -0.0813575, 0.8816222, -0.9629797, 0.9629797
3: -0.2928538, 0.8801655, -0.2928538, 0.8801655, -1.1730193, 1.1730193
4: -0.2609611, 0.8848863, -0.2609611, 0.8848863, -1.1458473, 1.1458473

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B2_A2_A2_B2_A1

### Relational analysis result of IS_B2_A2_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6631956, upper bound: 0.6645855
time: 0.39 seconds

## Relational analysis of IS_B2_A2_B2_A2_A2_B2_A2

### Relational analysis result of IS_B2_A2_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6631953, upper bound: 0.6752280
time: 0.42 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 2.83 seconds
IS_B1_B1_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.83
Output dim: 0, lower bound: -0.6446778, upper bound: 0.6359980
IS_B1_B1_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.83
Output dim: 0, lower bound: -0.6446778, upper bound: 0.6516630
IS_B1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 0, lower bound: -0.6335183, upper bound: 0.6735472
IS_B1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 0, lower bound: -0.6197446, upper bound: 0.6747169
IS_B1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 0, lower bound: -0.6400006, upper bound: 0.6706810
IS_B1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 0, lower bound: -0.6400006, upper bound: 0.6714281
IS_B1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 0, lower bound: -0.6400006, upper bound: 0.6706810
IS_B1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 0, lower bound: -0.6400006, upper bound: 0.6714281
IS_B1_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.83
Output dim: 0, lower bound: -0.6415344, upper bound: 0.6595624
IS_B1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 0, lower bound: -0.6415344, upper bound: 0.6637412
IS_B1_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.83
Output dim: 0, lower bound: -0.6415345, upper bound: 0.6595624
IS_B1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 0, lower bound: -0.6415345, upper bound: 0.6637412
IS_B2_A2_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 0, lower bound: -0.6826392, upper bound: 0.6657280
IS_B2_A2_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6657280
IS_B2_A2_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 0, lower bound: -0.6909598, upper bound: 0.6657280
IS_B2_A2_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 0, lower bound: -0.6700552, upper bound: 0.6657280
IS_B2_A2_B1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6657280
IS_B2_A2_B1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6758026
IS_B2_A2_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 0, lower bound: -0.6700552, upper bound: 0.6657280
IS_B2_A2_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 0, lower bound: -0.6700552, upper bound: 0.6758026
IS_B2_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 0, lower bound: -0.6529501, upper bound: 0.6886226
IS_B2_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 0, lower bound: -0.6529501, upper bound: 0.6647563
IS_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 0, lower bound: -0.6442375, upper bound: 0.6966600
IS_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 0, lower bound: -0.6529501, upper bound: 0.6729556
IS_B2_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 0, lower bound: -0.6529501, upper bound: 0.6647563
IS_B2_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 0, lower bound: -0.6529501, upper bound: 0.6647563
IS_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 0, lower bound: -0.6529501, upper bound: 0.6729556
IS_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 0, lower bound: -0.6615940, upper bound: 0.6729556
IS_B2_A2_B2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 0, lower bound: -0.6825553, upper bound: 0.6677968
IS_B2_A2_B2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 0, lower bound: -0.6599371, upper bound: 0.6677968
IS_B2_A2_B2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 0, lower bound: -0.6909142, upper bound: 0.6646636
IS_B2_A2_B2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 0, lower bound: -0.6687333, upper bound: 0.6677968
IS_B2_A2_B2_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 0, lower bound: -0.6599371, upper bound: 0.6677968
IS_B2_A2_B2_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 0, lower bound: -0.6599371, upper bound: 0.6757891
IS_B2_A2_B2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 0, lower bound: -0.6687333, upper bound: 0.6677968
IS_B2_A2_B2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 0, lower bound: -0.6687333, upper bound: 0.6757891
IS_B2_A2_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 0, lower bound: -0.6736765, upper bound: 0.6645855
IS_B2_A2_B2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 0, lower bound: -0.6530565, upper bound: 0.6645855
IS_B2_A2_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 0, lower bound: -0.6825935, upper bound: 0.6645855
IS_B2_A2_B2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 0, lower bound: -0.6631956, upper bound: 0.6645855
IS_B2_A2_B2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 0, lower bound: -0.6530565, upper bound: 0.6645855
IS_B2_A2_B2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 0, lower bound: -0.6530562, upper bound: 0.6752280
IS_B2_A2_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 0, lower bound: -0.6631956, upper bound: 0.6645855
IS_B2_A2_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.83
Output dim: 0, lower bound: -0.6631953, upper bound: 0.6752280

## BFS IS instance: IS_B1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0056276, 1.1569200, 0.0264370, 0.8148401, -0.8092124, 1.1304829
1: -0.1015921, 1.4327645, -0.0697827, 1.0368886, -1.1384807, 1.5025473
2: -0.0547483, 1.2686064, -0.0280635, 0.9040742, -0.9588225, 1.2966700
3: -0.2631698, 1.2212660, -0.2319205, 0.9520518, -1.2152215, 1.4531865
4: -0.2540219, 1.0182528, -0.2009447, 0.8100995, -1.0641215, 1.2191975

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 29

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_B1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6335183, upper bound: 0.6735472
time: 0.33 seconds

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_B1_B1_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6335183, upper bound: 0.6485423
time: 0.32 seconds

## BFS IS instance: IS_B1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0068502, 0.9733622, 0.0258129, 0.8220325, -0.8151822, 0.9475493
1: -0.1012402, 1.2255073, -0.0706267, 1.0456822, -1.1469223, 1.2961340
2: -0.0573967, 1.0806024, -0.0294187, 0.9115350, -0.9689317, 1.1100211
3: -0.2653522, 1.1000264, -0.2329345, 0.9584060, -1.2237582, 1.3329608
4: -0.2570790, 0.9276546, -0.2036403, 0.8138722, -1.0709512, 1.1312950

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 29

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_B1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6197446, upper bound: 0.6747169
time: 0.35 seconds

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_B1_B1_A1_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6197446, upper bound: 0.6497120
time: 0.34 seconds

## BFS IS instance: IS_B1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0033877, 0.9334497, -0.0014365, 1.0292757, -1.0258880, 0.9348862
1: -0.1048853, 1.1762533, -0.1070645, 1.3027349, -1.4076202, 1.2833178
2: -0.0566864, 1.0394158, -0.0715175, 1.1294107, -1.1860971, 1.1109333
3: -0.2678039, 1.0640620, -0.2690842, 1.1685212, -1.4363251, 1.3331462
4: -0.2457467, 0.9142159, -0.2915230, 0.9521359, -1.1978827, 1.2057388

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_B1_A1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_B1_A1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 5

Time for candidate selection: 2.13 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_B1_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6348030, upper bound: 0.6844666
time: 0.34 seconds

## Relational analysis of IS_B1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_B1_B1_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6504679, upper bound: 0.6626328
time: 0.35 seconds

## BFS IS instance: IS_B1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0029712, 0.9736321, -0.0014365, 1.0292757, -1.0263045, 0.9750686
1: -0.1048355, 1.2256770, -0.1070645, 1.3027349, -1.4075704, 1.3327415
2: -0.0609729, 1.0797758, -0.0715175, 1.1294107, -1.1903837, 1.1512933
3: -0.2668819, 1.1009192, -0.2690842, 1.1685212, -1.4354031, 1.3700035
4: -0.2582399, 0.9304323, -0.2915230, 0.9521359, -1.2103758, 1.2219553

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_B1_A1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_B1_A1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 5

Time for candidate selection: 2.18 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_B1_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6348030, upper bound: 0.6852682
time: 0.33 seconds

## Relational analysis of IS_B1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_B1_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6504679, upper bound: 0.6634343
time: 0.33 seconds

## BFS IS instance: IS_B1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0033877, 0.9334497, -0.0357146, 1.8917470, -1.8883593, 0.9691644
1: -0.1048853, 1.1762533, -0.1529815, 2.3290219, -2.4339073, 1.3292348
2: -0.0566864, 1.0394158, -0.1319740, 2.0703101, -2.1269965, 1.1713898
3: -0.2678039, 1.0640620, -0.3186724, 1.8925552, -2.1603591, 1.3827344
4: -0.2457467, 0.9142159, -0.5038834, 1.4782946, -1.7240413, 1.4180992

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_B1_A1_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_B1_A1_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 29

Time for candidate selection: 2.14 seconds

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_B1_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_B1_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6202511, upper bound: 0.6700298
time: 0.34 seconds

## Relational analysis of IS_B1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_B1_B1_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6386122, upper bound: 0.6627502
time: 0.38 seconds

## BFS IS instance: IS_B1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0029712, 0.9736321, -0.0357146, 1.8917470, -1.8887758, 1.0093467
1: -0.1048355, 1.2256770, -0.1529815, 2.3290219, -2.4338574, 1.3786585
2: -0.0609729, 1.0797758, -0.1319740, 2.0703101, -2.1312830, 1.2117498
3: -0.2668819, 1.1009192, -0.3186724, 1.8925552, -2.1594372, 1.4195917
4: -0.2582399, 0.9304323, -0.5038834, 1.4782946, -1.7365345, 1.4343157

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_B1_A1_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_B1_A1_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 29

Time for candidate selection: 2.23 seconds

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_B1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_B1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6202511, upper bound: 0.6708313
time: 0.33 seconds

## Relational analysis of IS_B1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_B1_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6386122, upper bound: 0.6635518
time: 0.35 seconds

## BFS IS instance: IS_B1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0395455, 1.8740401, -0.0014365, 1.0292757, -1.0688212, 1.8754766
1: -0.1631434, 2.2948570, -0.1070645, 1.3027349, -1.4658782, 2.4019215
2: -0.1338389, 2.0581985, -0.0715175, 1.1294107, -1.2632496, 2.1297159
3: -0.3277593, 1.8582082, -0.2690842, 1.1685212, -1.4962804, 2.1272924
4: -0.4895988, 1.4944417, -0.2915230, 0.9521359, -1.4417347, 1.7859647

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_B1_A1_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_B1_A1_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 29

Time for candidate selection: 2.16 seconds

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_B1_B1_A1_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_B1_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6323952, upper bound: 0.6771435
time: 0.36 seconds

## Relational analysis of IS_B1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_B1_B1_A1_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6520016, upper bound: 0.6553096
time: 0.36 seconds

## BFS IS instance: IS_B1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0395455, 1.8740401, -0.0357146, 1.8917470, -1.9312925, 1.9097548
1: -0.1631434, 2.2948570, -0.1529815, 2.3290219, -2.4921653, 2.4478385
2: -0.1338389, 2.0581985, -0.1319740, 2.0703101, -2.2041490, 2.1901724
3: -0.3277593, 1.8582082, -0.3186724, 1.8925552, -2.2203145, 2.1768806
4: -0.4895988, 1.4944417, -0.5038834, 1.4782946, -1.9678934, 1.9983251

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_B1_A1_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_B1_A1_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5

Time for candidate selection: 2.28 seconds

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_B1_B1_A1_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_B1_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_B1_B1_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6217849, upper bound: 0.6627066
time: 0.36 seconds

## Relational analysis of IS_B1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_B1_B1_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6401460, upper bound: 0.6553096
time: 0.34 seconds

## BFS IS instance: IS_B2_A2_B1_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, 0.0111880, 0.5648615, -0.5536735, 0.5536735
1: -0.0944340, 0.7130072, -0.0944340, 0.7130072, -0.8074412, 0.8074412
2: -0.0312903, 0.7124612, -0.0312903, 0.7124612, -0.7437515, 0.7437515
3: -0.2500148, 0.7301772, -0.2500148, 0.7301772, -0.9801920, 0.9801920
4: -0.1787794, 0.7720095, -0.1787794, 0.7720095, -0.9507889, 0.9507889

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B1_A1_A1_B1_A1_B1

### Relational analysis result of IS_B2_A2_B1_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6890618
time: 0.38 seconds

## Relational analysis of IS_B2_A2_B1_A1_A1_B1_A1_B2

### Relational analysis result of IS_B2_A2_B1_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6657280
time: 0.36 seconds

## BFS IS instance: IS_B2_A2_B1_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, 0.0111880, 0.5648615, -0.5954285, 0.6594702
1: -0.1440713, 0.8347254, -0.0944340, 0.7130072, -0.8570786, 0.9291594
2: -0.0742025, 0.8280591, -0.0312903, 0.7124612, -0.7866638, 0.8593494
3: -0.2901301, 0.8312570, -0.2500148, 0.7301772, -1.0203073, 1.0812719
4: -0.2451730, 0.8507983, -0.1787794, 0.7720095, -1.0171825, 1.0295777

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B1_A1_A1_B1_A2_B1

### Relational analysis result of IS_B2_A2_B1_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6890618
time: 0.37 seconds

## Relational analysis of IS_B2_A2_B1_A1_A1_B1_A2_B2

### Relational analysis result of IS_B2_A2_B1_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6657280
time: 0.36 seconds

## BFS IS instance: IS_B2_A2_B1_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, -0.0305669, 0.6706582, -0.6594702, 0.5954285
1: -0.0944340, 0.7130072, -0.1440713, 0.8347254, -0.9291594, 0.8570786
2: -0.0312903, 0.7124612, -0.0742025, 0.8280591, -0.8593494, 0.7866638
3: -0.2500148, 0.7301772, -0.2901301, 0.8312570, -1.0812719, 1.0203073
4: -0.1787794, 0.7720095, -0.2451730, 0.8507983, -1.0295777, 1.0171825

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B1_A1_A1_B2_A1_B1

### Relational analysis result of IS_B2_A2_B1_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6657280
time: 0.36 seconds

## Relational analysis of IS_B2_A2_B1_A1_A1_B2_A1_B2

### Relational analysis result of IS_B2_A2_B1_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6657280
time: 0.36 seconds

## BFS IS instance: IS_B2_A2_B1_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, -0.0305669, 0.6706582, -0.7012252, 0.7012252
1: -0.1440713, 0.8347254, -0.1440713, 0.8347254, -0.9787967, 0.9787967
2: -0.0742025, 0.8280591, -0.0742025, 0.8280591, -0.9022617, 0.9022617
3: -0.2901301, 0.8312570, -0.2901301, 0.8312570, -1.1213872, 1.1213872
4: -0.2451730, 0.8507983, -0.2451730, 0.8507983, -1.0959713, 1.0959713

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B1_A1_A1_B2_A2_B1

### Relational analysis result of IS_B2_A2_B1_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6657280
time: 0.36 seconds

## Relational analysis of IS_B2_A2_B1_A1_A1_B2_A2_B2

### Relational analysis result of IS_B2_A2_B1_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6657280
time: 0.35 seconds

## BFS IS instance: IS_B2_A2_B1_A1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, 0.0111880, 0.5648615, -0.5536735, 0.5536735
1: -0.0944340, 0.7130072, -0.0944340, 0.7130072, -0.8074412, 0.8074412
2: -0.0312903, 0.7124612, -0.0312903, 0.7124612, -0.7437515, 0.7437515
3: -0.2500148, 0.7301772, -0.2500148, 0.7301772, -0.9801920, 0.9801920
4: -0.1787794, 0.7720095, -0.1787794, 0.7720095, -0.9507889, 0.9507889

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B1_A1_A2_B1_A1_B1

### Relational analysis result of IS_B2_A2_B1_A1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6890618
time: 0.38 seconds

## Relational analysis of IS_B2_A2_B1_A1_A2_B1_A1_B2

### Relational analysis result of IS_B2_A2_B1_A1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6657280
time: 0.36 seconds

## BFS IS instance: IS_B2_A2_B1_A1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, 0.0111880, 0.5648615, -0.5954285, 0.6594702
1: -0.1440713, 0.8347254, -0.0944340, 0.7130072, -0.8570786, 0.9291594
2: -0.0742025, 0.8280591, -0.0312903, 0.7124612, -0.7866638, 0.8593494
3: -0.2901301, 0.8312570, -0.2500148, 0.7301772, -1.0203073, 1.0812719
4: -0.2451730, 0.8507983, -0.1787794, 0.7720095, -1.0171825, 1.0295777

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B1_A1_A2_B1_A2_B1

### Relational analysis result of IS_B2_A2_B1_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6981418
time: 0.38 seconds

## Relational analysis of IS_B2_A2_B1_A1_A2_B1_A2_B2

### Relational analysis result of IS_B2_A2_B1_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6758026
time: 0.36 seconds

## BFS IS instance: IS_B2_A2_B1_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, -0.0305669, 0.6706582, -0.6594702, 0.5954285
1: -0.0944340, 0.7130072, -0.1440713, 0.8347254, -0.9291594, 0.8570786
2: -0.0312903, 0.7124612, -0.0742025, 0.8280591, -0.8593494, 0.7866638
3: -0.2500148, 0.7301772, -0.2901301, 0.8312570, -1.0812719, 1.0203073
4: -0.1787794, 0.7720095, -0.2451730, 0.8507983, -1.0295777, 1.0171825

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B1_A1_A2_B2_A1_B1

### Relational analysis result of IS_B2_A2_B1_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6657280
time: 0.37 seconds

## Relational analysis of IS_B2_A2_B1_A1_A2_B2_A1_B2

### Relational analysis result of IS_B2_A2_B1_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6657280
time: 0.36 seconds

## BFS IS instance: IS_B2_A2_B1_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, -0.0305669, 0.6706582, -0.7012252, 0.7012252
1: -0.1440713, 0.8347254, -0.1440713, 0.8347254, -0.9787967, 0.9787967
2: -0.0742025, 0.8280591, -0.0742025, 0.8280591, -0.9022617, 0.9022617
3: -0.2901301, 0.8312570, -0.2901301, 0.8312570, -1.1213872, 1.1213872
4: -0.2451730, 0.8507983, -0.2451730, 0.8507983, -1.0959713, 1.0959713

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B1_A1_A2_B2_A2_B1

### Relational analysis result of IS_B2_A2_B1_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6758026
time: 0.38 seconds

## Relational analysis of IS_B2_A2_B1_A1_A2_B2_A2_B2

### Relational analysis result of IS_B2_A2_B1_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6700552, upper bound: 0.6758026
time: 0.38 seconds

## BFS IS instance: IS_B2_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0061345, 0.6507710, 0.0111880, 0.5648615, -0.5587270, 0.6395830
1: -0.0976210, 0.8106803, -0.0944340, 0.7130072, -0.8106282, 0.9051143
2: -0.0417109, 0.7880348, -0.0312903, 0.7124612, -0.7541721, 0.8193251
3: -0.2534082, 0.7965333, -0.2500148, 0.7301772, -0.9835854, 1.0465481
4: -0.2023642, 0.8128391, -0.1787794, 0.7720095, -0.9743737, 0.9916185

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_B2_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6736765, upper bound: 0.6647563
time: 0.38 seconds

## Relational analysis of IS_B2_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_B2_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6529501, upper bound: 0.6647563
time: 0.37 seconds

## BFS IS instance: IS_B2_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0061345, 0.6507710, -0.0305669, 0.6706582, -0.6645237, 0.6813380
1: -0.0976210, 0.8106803, -0.1440713, 0.8347254, -0.9323463, 0.9547516
2: -0.0417109, 0.7880348, -0.0742025, 0.8280591, -0.8697700, 0.8622373
3: -0.2534082, 0.7965333, -0.2901301, 0.8312570, -1.0846653, 1.0866635
4: -0.2023642, 0.8128391, -0.2451730, 0.8507983, -1.0531626, 1.0580120

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 29

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_B2_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6736765, upper bound: 0.6647563
time: 0.38 seconds

## Relational analysis of IS_B2_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_B2_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6529501, upper bound: 0.6647563
time: 0.37 seconds

## BFS IS instance: IS_B2_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0329778, 0.7304660, 0.0111880, 0.5648615, -0.5978394, 0.7192780
1: -0.1447692, 0.9029138, -0.0944340, 0.7130072, -0.8577764, 0.9973478
2: -0.0813575, 0.8816222, -0.0312903, 0.7124612, -0.7938187, 0.9129125
3: -0.2928538, 0.8801655, -0.2500148, 0.7301772, -1.0230310, 1.1301802
4: -0.2609611, 0.8848863, -0.1787794, 0.7720095, -1.0329705, 1.0636656

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_B2_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6529501, upper bound: 0.6647563
time: 0.38 seconds

## Relational analysis of IS_B2_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_B2_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6442375, upper bound: 0.6729554
time: 0.35 seconds

## BFS IS instance: IS_B2_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0329778, 0.7304660, -0.0305669, 0.6706582, -0.7036361, 0.7610329
1: -0.1447692, 0.9029138, -0.1440713, 0.8347254, -0.9794946, 1.0469851
2: -0.0813575, 0.8816222, -0.0742025, 0.8280591, -0.9094166, 0.9558247
3: -0.2928538, 0.8801655, -0.2901301, 0.8312570, -1.1241109, 1.1702956
4: -0.2609611, 0.8848863, -0.2451730, 0.8507983, -1.1117594, 1.1300592

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_B2_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6529501, upper bound: 0.6647563
time: 0.38 seconds

## Relational analysis of IS_B2_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_B2_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6442375, upper bound: 0.6729554
time: 0.34 seconds

## BFS IS instance: IS_B2_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0061345, 0.6507710, 0.0111880, 0.5648615, -0.5587270, 0.6395830
1: -0.0976210, 0.8106803, -0.0944340, 0.7130072, -0.8106282, 0.9051143
2: -0.0417109, 0.7880348, -0.0312903, 0.7124612, -0.7541721, 0.8193251
3: -0.2534082, 0.7965333, -0.2500148, 0.7301772, -0.9835854, 1.0465481
4: -0.2023642, 0.8128391, -0.1787794, 0.7720095, -0.9743737, 0.9916185

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_B2_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6736765, upper bound: 0.6647563
time: 0.39 seconds

## Relational analysis of IS_B2_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_B2_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6529501, upper bound: 0.6647563
time: 0.38 seconds

## BFS IS instance: IS_B2_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0061345, 0.6507710, -0.0305669, 0.6706582, -0.6645237, 0.6813380
1: -0.0976210, 0.8106803, -0.1440713, 0.8347254, -0.9323463, 0.9547516
2: -0.0417109, 0.7880348, -0.0742025, 0.8280591, -0.8697700, 0.8622373
3: -0.2534082, 0.7965333, -0.2901301, 0.8312570, -1.0846653, 1.0866635
4: -0.2023642, 0.8128391, -0.2451730, 0.8507983, -1.0531626, 1.0580120

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_B2_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6776488, upper bound: 0.6647563
time: 0.35 seconds

## Relational analysis of IS_B2_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_B2_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6615940, upper bound: 0.6647563
time: 0.35 seconds

## BFS IS instance: IS_B2_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0329778, 0.7304660, 0.0111880, 0.5648615, -0.5978394, 0.7192780
1: -0.1447692, 0.9029138, -0.0944340, 0.7130072, -0.8577764, 0.9973478
2: -0.0813575, 0.8816222, -0.0312903, 0.7124612, -0.7938187, 0.9129125
3: -0.2928538, 0.8801655, -0.2500148, 0.7301772, -1.0230310, 1.1301802
4: -0.2609611, 0.8848863, -0.1787794, 0.7720095, -1.0329705, 1.0636656

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_B2_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6529501, upper bound: 0.6647563
time: 0.38 seconds

## Relational analysis of IS_B2_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_B2_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6442375, upper bound: 0.6729554
time: 0.35 seconds

## BFS IS instance: IS_B2_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0329778, 0.7304660, -0.0305669, 0.6706582, -0.7036361, 0.7610329
1: -0.1447692, 0.9029138, -0.1440713, 0.8347254, -0.9794946, 1.0469851
2: -0.0813575, 0.8816222, -0.0742025, 0.8280591, -0.9094166, 0.9558247
3: -0.2928538, 0.8801655, -0.2901301, 0.8312570, -1.1241109, 1.1702956
4: -0.2609611, 0.8848863, -0.2451730, 0.8507983, -1.1117594, 1.1300592

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_B2_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6615940, upper bound: 0.6647563
time: 0.37 seconds

## Relational analysis of IS_B2_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_B2_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6615940, upper bound: 0.6729556
time: 0.38 seconds

## BFS IS instance: IS_B2_A2_B2_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, 0.0061345, 0.6507710, -0.6395830, 0.5587270
1: -0.0944340, 0.7130072, -0.0976210, 0.8106803, -0.9051143, 0.8106282
2: -0.0312903, 0.7124612, -0.0417109, 0.7880348, -0.8193251, 0.7541721
3: -0.2500148, 0.7301772, -0.2534082, 0.7965333, -1.0465481, 0.9835854
4: -0.1787794, 0.7720095, -0.2023642, 0.8128391, -0.9916185, 0.9743737

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B2_A1_A1_B1_A1_B1

### Relational analysis result of IS_B2_A2_B2_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6599371, upper bound: 0.6892845
time: 0.34 seconds

## Relational analysis of IS_B2_A2_B2_A1_A1_B1_A1_B2

### Relational analysis result of IS_B2_A2_B2_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6599371, upper bound: 0.6677968
time: 0.37 seconds

## BFS IS instance: IS_B2_A2_B2_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, 0.0061345, 0.6507710, -0.6813380, 0.6645237
1: -0.1440713, 0.8347254, -0.0976210, 0.8106803, -0.9547516, 0.9323463
2: -0.0742025, 0.8280591, -0.0417109, 0.7880348, -0.8622373, 0.8697700
3: -0.2901301, 0.8312570, -0.2534082, 0.7965333, -1.0866635, 1.0846653
4: -0.2451730, 0.8507983, -0.2023642, 0.8128391, -1.0580120, 1.0531626

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B2_A1_A1_B1_A2_B1

### Relational analysis result of IS_B2_A2_B2_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6599371, upper bound: 0.6892845
time: 0.35 seconds

## Relational analysis of IS_B2_A2_B2_A1_A1_B1_A2_B2

### Relational analysis result of IS_B2_A2_B2_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6599371, upper bound: 0.6677968
time: 0.36 seconds

## BFS IS instance: IS_B2_A2_B2_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, -0.0329778, 0.7304660, -0.7192780, 0.5978394
1: -0.0944340, 0.7130072, -0.1447692, 0.9029138, -0.9973478, 0.8577764
2: -0.0312903, 0.7124612, -0.0813575, 0.8816222, -0.9129125, 0.7938187
3: -0.2500148, 0.7301772, -0.2928538, 0.8801655, -1.1301802, 1.0230310
4: -0.1787794, 0.7720095, -0.2609611, 0.8848863, -1.0636656, 1.0329705

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B2_A1_A1_B2_A1_B1

### Relational analysis result of IS_B2_A2_B2_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6599371, upper bound: 0.6677968
time: 0.37 seconds

## Relational analysis of IS_B2_A2_B2_A1_A1_B2_A1_B2

### Relational analysis result of IS_B2_A2_B2_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6599371, upper bound: 0.6677968
time: 0.37 seconds

## BFS IS instance: IS_B2_A2_B2_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, -0.0329778, 0.7304660, -0.7610329, 0.7036361
1: -0.1440713, 0.8347254, -0.1447692, 0.9029138, -1.0469851, 0.9794946
2: -0.0742025, 0.8280591, -0.0813575, 0.8816222, -0.9558247, 0.9094166
3: -0.2901301, 0.8312570, -0.2928538, 0.8801655, -1.1702956, 1.1241109
4: -0.2451730, 0.8507983, -0.2609611, 0.8848863, -1.1300592, 1.1117594

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B2_A1_A1_B2_A2_B1

### Relational analysis result of IS_B2_A2_B2_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6599371, upper bound: 0.6677968
time: 0.39 seconds

## Relational analysis of IS_B2_A2_B2_A1_A1_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6599371, upper bound: 0.6677968
time: 0.39 seconds

## BFS IS instance: IS_B2_A2_B2_A1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, 0.0061345, 0.6507710, -0.6395830, 0.5587270
1: -0.0944340, 0.7130072, -0.0976210, 0.8106803, -0.9051143, 0.8106282
2: -0.0312903, 0.7124612, -0.0417109, 0.7880348, -0.8193251, 0.7541721
3: -0.2500148, 0.7301772, -0.2534082, 0.7965333, -1.0465481, 0.9835854
4: -0.1787794, 0.7720095, -0.2023642, 0.8128391, -0.9916185, 0.9743737

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B2_A1_A2_B1_A1_B1

### Relational analysis result of IS_B2_A2_B2_A1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6599371, upper bound: 0.6892845
time: 0.35 seconds

## Relational analysis of IS_B2_A2_B2_A1_A2_B1_A1_B2

### Relational analysis result of IS_B2_A2_B2_A1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6599371, upper bound: 0.6677968
time: 0.36 seconds

## BFS IS instance: IS_B2_A2_B2_A1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, 0.0061345, 0.6507710, -0.6813380, 0.6645237
1: -0.1440713, 0.8347254, -0.0976210, 0.8106803, -0.9547516, 0.9323463
2: -0.0742025, 0.8280591, -0.0417109, 0.7880348, -0.8622373, 0.8697700
3: -0.2901301, 0.8312570, -0.2534082, 0.7965333, -1.0866635, 1.0846653
4: -0.2451730, 0.8507983, -0.2023642, 0.8128391, -1.0580120, 1.0531626

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B2_A1_A2_B1_A2_B1

### Relational analysis result of IS_B2_A2_B2_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6599371, upper bound: 0.6976855
time: 0.40 seconds

## Relational analysis of IS_B2_A2_B2_A1_A2_B1_A2_B2

### Relational analysis result of IS_B2_A2_B2_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6599371, upper bound: 0.6757891
time: 0.39 seconds

## BFS IS instance: IS_B2_A2_B2_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, -0.0329778, 0.7304660, -0.7192780, 0.5978394
1: -0.0944340, 0.7130072, -0.1447692, 0.9029138, -0.9973478, 0.8577764
2: -0.0312903, 0.7124612, -0.0813575, 0.8816222, -0.9129125, 0.7938187
3: -0.2500148, 0.7301772, -0.2928538, 0.8801655, -1.1301802, 1.0230310
4: -0.1787794, 0.7720095, -0.2609611, 0.8848863, -1.0636656, 1.0329705

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B2_A1_A2_B2_A1_B1

### Relational analysis result of IS_B2_A2_B2_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6599371, upper bound: 0.6677968
time: 0.38 seconds

## Relational analysis of IS_B2_A2_B2_A1_A2_B2_A1_B2

### Relational analysis result of IS_B2_A2_B2_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6599371, upper bound: 0.6677968
time: 0.37 seconds

## BFS IS instance: IS_B2_A2_B2_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, -0.0329778, 0.7304660, -0.7610329, 0.7036361
1: -0.1440713, 0.8347254, -0.1447692, 0.9029138, -1.0469851, 0.9794946
2: -0.0742025, 0.8280591, -0.0813575, 0.8816222, -0.9558247, 0.9094166
3: -0.2901301, 0.8312570, -0.2928538, 0.8801655, -1.1702956, 1.1241109
4: -0.2451730, 0.8507983, -0.2609611, 0.8848863, -1.1300592, 1.1117594

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B2_A1_A2_B2_A2_B1

### Relational analysis result of IS_B2_A2_B2_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6599371, upper bound: 0.6757891
time: 0.38 seconds

## Relational analysis of IS_B2_A2_B2_A1_A2_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6687333, upper bound: 0.6757891
time: 0.39 seconds

## BFS IS instance: IS_B2_A2_B2_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0061345, 0.6507710, 0.0061345, 0.6507710, -0.6446365, 0.6446365
1: -0.0976210, 0.8106803, -0.0976210, 0.8106803, -0.9083012, 0.9083012
2: -0.0417109, 0.7880348, -0.0417109, 0.7880348, -0.8297457, 0.8297457
3: -0.2534082, 0.7965333, -0.2534082, 0.7965333, -1.0499415, 1.0499415
4: -0.2023642, 0.8128391, -0.2023642, 0.8128391, -1.0152032, 1.0152032

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B2_A2_A1_B1_A1_B1

### Relational analysis result of IS_B2_A2_B2_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6530562, upper bound: 0.6886226
time: 0.41 seconds

## Relational analysis of IS_B2_A2_B2_A2_A1_B1_A1_B2

### Relational analysis result of IS_B2_A2_B2_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6530565, upper bound: 0.6645855
time: 0.38 seconds

## BFS IS instance: IS_B2_A2_B2_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0329778, 0.7275667, 0.0061345, 0.6507710, -0.6837488, 0.7214322
1: -0.1447692, 0.8997444, -0.0976210, 0.8106803, -0.9554495, 0.9973654
2: -0.0813575, 0.8785636, -0.0417109, 0.7880348, -0.8693923, 0.9202745
3: -0.2928538, 0.8771129, -0.2534082, 0.7965333, -1.0893872, 1.1305211
4: -0.2609611, 0.8821078, -0.2023642, 0.8128391, -1.0738001, 1.0844719

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B2_A2_A1_B1_A2_B1

### Relational analysis result of IS_B2_A2_B2_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6530562, upper bound: 0.6886226
time: 0.41 seconds

## Relational analysis of IS_B2_A2_B2_A2_A1_B1_A2_B2

### Relational analysis result of IS_B2_A2_B2_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6530565, upper bound: 0.6645855
time: 0.38 seconds

## BFS IS instance: IS_B2_A2_B2_A2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0061345, 0.6507710, -0.0329778, 0.7304660, -0.7243315, 0.6837488
1: -0.0976210, 0.8106803, -0.1447692, 0.9029138, -1.0005348, 0.9554495
2: -0.0417109, 0.7880348, -0.0813575, 0.8816222, -0.9233330, 0.8693923
3: -0.2534082, 0.7965333, -0.2928538, 0.8801655, -1.1335737, 1.0893872
4: -0.2023642, 0.8128391, -0.2609611, 0.8848863, -1.0872505, 1.0738001

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B2_A2_A1_B2_A1_B1

### Relational analysis result of IS_B2_A2_B2_A2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6530565, upper bound: 0.6645855
time: 0.40 seconds

## Relational analysis of IS_B2_A2_B2_A2_A1_B2_A1_B2

### Relational analysis result of IS_B2_A2_B2_A2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6530565, upper bound: 0.6645855
time: 0.40 seconds

## BFS IS instance: IS_B2_A2_B2_A2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0329778, 0.7275667, -0.0329778, 0.7304660, -0.7634438, 0.7605445
1: -0.1447692, 0.8997444, -0.1447692, 0.9029138, -1.0476830, 1.0445136
2: -0.0813575, 0.8785636, -0.0813575, 0.8816222, -0.9629797, 0.9599211
3: -0.2928538, 0.8771129, -0.2928538, 0.8801655, -1.1730193, 1.1699667
4: -0.2609611, 0.8821078, -0.2609611, 0.8848863, -1.1458473, 1.1430688

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B2_A2_A1_B2_A2_B1

### Relational analysis result of IS_B2_A2_B2_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6530565, upper bound: 0.6645855
time: 0.41 seconds

## Relational analysis of IS_B2_A2_B2_A2_A1_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6530565, upper bound: 0.6645855
time: 0.40 seconds

## BFS IS instance: IS_B2_A2_B2_A2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0061345, 0.6507710, 0.0061345, 0.6507710, -0.6446365, 0.6446365
1: -0.0976210, 0.8106803, -0.0976210, 0.8106803, -0.9083012, 0.9083012
2: -0.0417109, 0.7880348, -0.0417109, 0.7880348, -0.8297457, 0.8297457
3: -0.2534082, 0.7965333, -0.2534082, 0.7965333, -1.0499415, 1.0499415
4: -0.2023642, 0.8128391, -0.2023642, 0.8128391, -1.0152032, 1.0152032

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B2_A2_A2_B1_A1_B1

### Relational analysis result of IS_B2_A2_B2_A2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6530562, upper bound: 0.6886226
time: 0.41 seconds

## Relational analysis of IS_B2_A2_B2_A2_A2_B1_A1_B2

### Relational analysis result of IS_B2_A2_B2_A2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6530565, upper bound: 0.6645855
time: 0.37 seconds

## BFS IS instance: IS_B2_A2_B2_A2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0329778, 0.7304660, 0.0061345, 0.6507710, -0.6837488, 0.7243315
1: -0.1447692, 0.9029138, -0.0976210, 0.8106803, -0.9554495, 1.0005348
2: -0.0813575, 0.8816222, -0.0417109, 0.7880348, -0.8693923, 0.9233330
3: -0.2928538, 0.8801655, -0.2534082, 0.7965333, -1.0893872, 1.1335737
4: -0.2609611, 0.8848863, -0.2023642, 0.8128391, -1.0738001, 1.0872505

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B2_A2_A2_B1_A2_B1

### Relational analysis result of IS_B2_A2_B2_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6530565, upper bound: 0.6969876
time: 0.37 seconds

## Relational analysis of IS_B2_A2_B2_A2_A2_B1_A2_B2

### Relational analysis result of IS_B2_A2_B2_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6530562, upper bound: 0.6752281
time: 0.41 seconds

## BFS IS instance: IS_B2_A2_B2_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0061345, 0.6507710, -0.0329778, 0.7304660, -0.7243315, 0.6837488
1: -0.0976210, 0.8106803, -0.1447692, 0.9029138, -1.0005348, 0.9554495
2: -0.0417109, 0.7880348, -0.0813575, 0.8816222, -0.9233330, 0.8693923
3: -0.2534082, 0.7965333, -0.2928538, 0.8801655, -1.1335737, 1.0893872
4: -0.2023642, 0.8128391, -0.2609611, 0.8848863, -1.0872505, 1.0738001

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B2_A2_A2_B2_A1_B1

### Relational analysis result of IS_B2_A2_B2_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6530565, upper bound: 0.6645855
time: 0.40 seconds

## Relational analysis of IS_B2_A2_B2_A2_A2_B2_A1_B2

### Relational analysis result of IS_B2_A2_B2_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6530565, upper bound: 0.6645855
time: 0.41 seconds

## BFS IS instance: IS_B2_A2_B2_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0329778, 0.7304660, -0.0329778, 0.7304660, -0.7634438, 0.7634438
1: -0.1447692, 0.9029138, -0.1447692, 0.9029138, -1.0476830, 1.0476830
2: -0.0813575, 0.8816222, -0.0813575, 0.8816222, -0.9629797, 0.9629797
3: -0.2928538, 0.8801655, -0.2928538, 0.8801655, -1.1730193, 1.1730193
4: -0.2609611, 0.8848863, -0.2609611, 0.8848863, -1.1458473, 1.1458473

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B2_A2_A2_B2_A2_B1

### Relational analysis result of IS_B2_A2_B2_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6530562, upper bound: 0.6752281
time: 0.42 seconds

## Relational analysis of IS_B2_A2_B2_A2_A2_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6631953, upper bound: 0.6752281
time: 0.39 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 2.91 seconds
IS_B1_B1_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6335183, upper bound: 0.6735472
IS_B1_B1_A1_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6335183, upper bound: 0.6485423
IS_B1_B1_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6197446, upper bound: 0.6747169
IS_B1_B1_A1_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6197446, upper bound: 0.6497120
IS_B1_B1_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6348030, upper bound: 0.6844666
IS_B1_B1_A1_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6504679, upper bound: 0.6626328
IS_B1_B1_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6348030, upper bound: 0.6852682
IS_B1_B1_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6504679, upper bound: 0.6634343
IS_B1_B1_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6202511, upper bound: 0.6700298
IS_B1_B1_A1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6386122, upper bound: 0.6627502
IS_B1_B1_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6202511, upper bound: 0.6708313
IS_B1_B1_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6386122, upper bound: 0.6635518
IS_B1_B1_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6323952, upper bound: 0.6771435
IS_B1_B1_A1_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6520016, upper bound: 0.6553096
IS_B1_B1_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6217849, upper bound: 0.6627066
IS_B1_B1_A1_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6401460, upper bound: 0.6553096
IS_B2_A2_B1_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6890618
IS_B2_A2_B1_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6657280
IS_B2_A2_B1_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6890618
IS_B2_A2_B1_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6657280
IS_B2_A2_B1_A1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6657280
IS_B2_A2_B1_A1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6657280
IS_B2_A2_B1_A1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6657280
IS_B2_A2_B1_A1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6657280
IS_B2_A2_B1_A1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6890618
IS_B2_A2_B1_A1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6657280
IS_B2_A2_B1_A1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6981418
IS_B2_A2_B1_A1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6758026
IS_B2_A2_B1_A1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6657280
IS_B2_A2_B1_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6657280
IS_B2_A2_B1_A1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6758026
IS_B2_A2_B1_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6700552, upper bound: 0.6758026
IS_B2_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6736765, upper bound: 0.6647563
IS_B2_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6529501, upper bound: 0.6647563
IS_B2_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6736765, upper bound: 0.6647563
IS_B2_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6529501, upper bound: 0.6647563
IS_B2_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6529501, upper bound: 0.6647563
IS_B2_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6442375, upper bound: 0.6729554
IS_B2_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6529501, upper bound: 0.6647563
IS_B2_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6442375, upper bound: 0.6729554
IS_B2_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6736765, upper bound: 0.6647563
IS_B2_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6529501, upper bound: 0.6647563
IS_B2_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6776488, upper bound: 0.6647563
IS_B2_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6615940, upper bound: 0.6647563
IS_B2_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6529501, upper bound: 0.6647563
IS_B2_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6442375, upper bound: 0.6729554
IS_B2_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6615940, upper bound: 0.6647563
IS_B2_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6615940, upper bound: 0.6729556
IS_B2_A2_B2_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6599371, upper bound: 0.6892845
IS_B2_A2_B2_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6599371, upper bound: 0.6677968
IS_B2_A2_B2_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6599371, upper bound: 0.6892845
IS_B2_A2_B2_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6599371, upper bound: 0.6677968
IS_B2_A2_B2_A1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6599371, upper bound: 0.6677968
IS_B2_A2_B2_A1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6599371, upper bound: 0.6677968
IS_B2_A2_B2_A1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6599371, upper bound: 0.6677968
IS_B2_A2_B2_A1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6599371, upper bound: 0.6677968
IS_B2_A2_B2_A1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6599371, upper bound: 0.6892845
IS_B2_A2_B2_A1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6599371, upper bound: 0.6677968
IS_B2_A2_B2_A1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6599371, upper bound: 0.6976855
IS_B2_A2_B2_A1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6599371, upper bound: 0.6757891
IS_B2_A2_B2_A1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6599371, upper bound: 0.6677968
IS_B2_A2_B2_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6599371, upper bound: 0.6677968
IS_B2_A2_B2_A1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6599371, upper bound: 0.6757891
IS_B2_A2_B2_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6687333, upper bound: 0.6757891
IS_B2_A2_B2_A2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6530562, upper bound: 0.6886226
IS_B2_A2_B2_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6530565, upper bound: 0.6645855
IS_B2_A2_B2_A2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6530562, upper bound: 0.6886226
IS_B2_A2_B2_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6530565, upper bound: 0.6645855
IS_B2_A2_B2_A2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6530565, upper bound: 0.6645855
IS_B2_A2_B2_A2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6530565, upper bound: 0.6645855
IS_B2_A2_B2_A2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6530565, upper bound: 0.6645855
IS_B2_A2_B2_A2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6530565, upper bound: 0.6645855
IS_B2_A2_B2_A2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6530562, upper bound: 0.6886226
IS_B2_A2_B2_A2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6530565, upper bound: 0.6645855
IS_B2_A2_B2_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6530565, upper bound: 0.6969876
IS_B2_A2_B2_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6530562, upper bound: 0.6752281
IS_B2_A2_B2_A2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6530565, upper bound: 0.6645855
IS_B2_A2_B2_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6530565, upper bound: 0.6645855
IS_B2_A2_B2_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6530562, upper bound: 0.6752281
IS_B2_A2_B2_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.91
Output dim: 0, lower bound: -0.6631953, upper bound: 0.6752281

## BFS IS instance: IS_B1_B1_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0056276, 1.1569200, 0.0258129, 0.8220325, -0.8164048, 1.1311071
1: -0.1015921, 1.4327645, -0.0706267, 1.0456822, -1.1472743, 1.5033913
2: -0.0547483, 1.2686064, -0.0294187, 0.9115350, -0.9662833, 1.2980251
3: -0.2631698, 1.2212660, -0.2329345, 0.9584060, -1.2215757, 1.4542005
4: -0.2540219, 1.0182528, -0.2036403, 0.8138722, -1.0678941, 1.2218932

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_B1_B1_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6335183, upper bound: 0.6735472
time: 0.35 seconds

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_B1_B1_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6197446, upper bound: 0.6735472
time: 0.32 seconds

## BFS IS instance: IS_B1_B1_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0068502, 0.9733622, 0.0258129, 0.8220325, -0.8151822, 0.9475493
1: -0.1012402, 1.2255073, -0.0706267, 1.0456822, -1.1469223, 1.2961340
2: -0.0573967, 1.0806024, -0.0294187, 0.9115350, -0.9689317, 1.1100211
3: -0.2653522, 1.1000264, -0.2329345, 0.9584060, -1.2237582, 1.3329608
4: -0.2570790, 0.9276546, -0.2036403, 0.8138722, -1.0709512, 1.1312950

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_B1_B1_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6197446, upper bound: 0.6735472
time: 0.33 seconds

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_B1_B1_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6197446, upper bound: 0.6747169
time: 0.35 seconds

## BFS IS instance: IS_B1_B1_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0033877, 0.9334497, 0.0033717, 0.9581215, -0.9547338, 0.9300780
1: -0.1048853, 1.1762533, -0.1002848, 1.2202435, -1.3251288, 1.2765381
2: -0.0566864, 1.0394158, -0.0643766, 1.0534739, -1.1101604, 1.1037924
3: -0.2678039, 1.0640620, -0.2618394, 1.1076934, -1.3754973, 1.3259014
4: -0.2457467, 0.9142159, -0.2770617, 0.9064128, -1.1521596, 1.1912775

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_B1_B1_A1_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6445603, upper bound: 0.6469678
time: 0.34 seconds

## Relational analysis of IS_B1_B1_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_B1_B1_A1_B2_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6445603, upper bound: 0.6626328
time: 0.33 seconds

## BFS IS instance: IS_B1_B1_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0029712, 0.9736321, 0.0033717, 0.9581215, -0.9551504, 0.9702604
1: -0.1048355, 1.2256770, -0.1002848, 1.2202435, -1.3250790, 1.3259618
2: -0.0609729, 1.0797758, -0.0643766, 1.0534739, -1.1144469, 1.1441524
3: -0.2668819, 1.1009192, -0.2618394, 1.1076934, -1.3745754, 1.3627586
4: -0.2582399, 0.9304323, -0.2770617, 0.9064128, -1.1646527, 1.2074940

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 15

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 47

## Relational analysis of IS_B1_B1_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_B1_B1_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 47

## Relational analysis of IS_B1_B1_A1_B2_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_B1_B1_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_B1_B1_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6314926, upper bound: 0.6734962
time: 0.35 seconds

## Relational analysis of IS_B1_B1_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_B1_B1_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6196762, upper bound: 0.6746658
time: 0.33 seconds

## BFS IS instance: IS_B1_B1_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0029712, 0.9736321, -0.0329940, 1.0375891, -1.0346179, 1.0066261
1: -0.1048355, 1.2256770, -0.1455390, 1.3108253, -1.4156609, 1.3712161
2: -0.0609729, 1.0797758, -0.1032507, 1.1475852, -1.2085581, 1.1830266
3: -0.2668819, 1.1009192, -0.3013754, 1.1888072, -1.4556892, 1.4022946
4: -0.2582399, 0.9304323, -0.3237462, 0.9872974, -1.2455373, 1.2541785

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_B1_B1_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_B1_B1_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_B1_B1_A1_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6495653, upper bound: 0.6510793
time: 0.37 seconds

## Relational analysis of IS_B1_B1_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_B1_B1_A1_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6377489, upper bound: 0.6522490
time: 0.35 seconds

## BFS IS instance: IS_B1_B1_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0033877, 0.9334497, -0.0268860, 1.7603378, -1.7569501, 0.9603357
1: -0.1048853, 1.1762533, -0.1414425, 2.1766334, -2.2815187, 1.3176959
2: -0.0566864, 1.0394158, -0.1192527, 1.9263821, -1.9830685, 1.1586685
3: -0.2678039, 1.0640620, -0.3063855, 1.7812052, -2.0490091, 1.3704475
4: -0.2457467, 0.9142159, -0.4786245, 1.3885493, -1.6342961, 1.3928404

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_B1_B1_A1_B2_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_B1_B1_A1_B2_A1_B2_A1_B1_B1

### Relational analysis result of IS_B1_B1_A1_B2_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6212308, upper bound: 0.6698095
time: 0.36 seconds

## Relational analysis of IS_B1_B1_A1_B2_A1_B2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_B1_B1_A1_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6324162, upper bound: 0.6470853
time: 0.36 seconds

## Relational analysis of IS_B1_B1_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_B1_B1_A1_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6324162, upper bound: 0.6627502
time: 0.37 seconds

## BFS IS instance: IS_B1_B1_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0029712, 0.9736321, -0.0268860, 1.7603378, -1.7573667, 1.0005181
1: -0.1048355, 1.2256770, -0.1414425, 2.1766334, -2.2814689, 1.3671196
2: -0.0609729, 1.0797758, -0.1192527, 1.9263821, -1.9873550, 1.1990285
3: -0.2668819, 1.1009192, -0.3063855, 1.7812052, -2.0480871, 1.4073048
4: -0.2582399, 0.9304323, -0.4786245, 1.3885493, -1.6467892, 1.4090568

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_B1_B1_A1_B2_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_B1_B1_A1_B2_A1_B2_A2_B1_B1

### Relational analysis result of IS_B1_B1_A1_B2_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6075321, upper bound: 0.6706111
time: 0.39 seconds

## Relational analysis of IS_B1_B1_A1_B2_A1_B2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 5

## Relational analysis of IS_B1_B1_A1_B2_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 47

## Relational analysis of IS_B1_B1_A1_B2_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_B1_B1_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 47

## Relational analysis of IS_B1_B1_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_B1_B1_A1_B2_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_B1_B1_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_B1_B1_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_B1_B1_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5
type: B, layer: 5, pos: 1
type: A, layer: 5, pos: 1
type: B, layer: 5, pos: 8
type: B, layer: 5, pos: 25
type: A, layer: 5, pos: 8
type: B, layer: 5, pos: 27
type: A, layer: 5, pos: 27
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 40
type: B, layer: 5, pos: 21
type: A, layer: 5, pos: 21
type: B, layer: 5, pos: 40
type: B, layer: 5, pos: 22

Time for candidate selection: 5.61 seconds

### Candidate
type: B, layer: 5, pos: 1

## Relational analysis of IS_B1_B1_A1_B2_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 1

## Relational analysis of IS_B1_B1_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_B1_B1_A1_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6148590, upper bound: 0.6561047
time: 0.35 seconds

## Relational analysis of IS_B1_B1_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_B1_B1_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6186506, upper bound: 0.6707525
time: 0.36 seconds

## BFS IS instance: IS_B1_B1_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0029712, 0.9736321, -0.0607445, 1.8971348, -1.8941636, 1.0343766
1: -0.1048355, 1.2256770, -0.1830909, 2.3339677, -2.4388032, 1.4087679
2: -0.0609729, 1.0797758, -0.1553247, 2.0793376, -2.1403105, 1.2351005
3: -0.2668819, 1.1009192, -0.3476024, 1.9050694, -2.1719513, 1.4485216
4: -0.2582399, 0.9304323, -0.5239611, 1.4932396, -1.7514795, 1.4543934

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_B1_B1_A1_B2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_B1_B1_A1_B2_A1_B2_A2_B2_B1

### Relational analysis result of IS_B1_B1_A1_B2_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6258932, upper bound: 0.6631460
time: 0.36 seconds

## Relational analysis of IS_B1_B1_A1_B2_A1_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 5

## Relational analysis of IS_B1_B1_A1_B2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 47

## Relational analysis of IS_B1_B1_A1_B2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_B1_B1_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 47

## Relational analysis of IS_B1_B1_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_B1_B1_A1_B2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_B1_B1_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_B1_B1_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_B1_B1_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5
type: B, layer: 5, pos: 1
type: A, layer: 5, pos: 1
type: B, layer: 5, pos: 8
type: B, layer: 5, pos: 25
type: A, layer: 5, pos: 8
type: A, layer: 5, pos: 27
type: B, layer: 5, pos: 27
type: A, layer: 5, pos: 25
type: A, layer: 5, pos: 40
type: B, layer: 5, pos: 21
type: A, layer: 5, pos: 21
type: B, layer: 5, pos: 40
type: B, layer: 5, pos: 22

Time for candidate selection: 5.64 seconds

### Candidate
type: B, layer: 5, pos: 1

## Relational analysis of IS_B1_B1_A1_B2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 1

## Relational analysis of IS_B1_B1_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_B1_B1_A1_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6332200, upper bound: 0.6482649
time: 0.34 seconds

## Relational analysis of IS_B1_B1_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_B1_B1_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6370117, upper bound: 0.6629127
time: 0.37 seconds

## BFS IS instance: IS_B1_B1_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0395455, 1.8740401, 0.0033717, 0.9581215, -0.9976671, 1.8706684
1: -0.1631434, 2.2948570, -0.1002848, 1.2202435, -1.3833869, 2.3951418
2: -0.1338389, 2.0581985, -0.0643766, 1.0534739, -1.1873128, 2.1225750
3: -0.3277593, 1.8582082, -0.2618394, 1.1076934, -1.4354527, 2.1200476
4: -0.4895988, 1.4944417, -0.2770617, 0.9064128, -1.3960116, 1.7715034

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_B1_B1_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_B1_B1_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_B1_B1_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6314926, upper bound: 0.6665411
time: 0.36 seconds

## Relational analysis of IS_B1_B1_A1_B2_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 47

## Relational analysis of IS_B1_B1_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 47

## Relational analysis of IS_B1_B1_A1_B2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_B1_B1_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_B1_B1_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 5

## Relational analysis of IS_B1_B1_A1_B2_A2_B1_A2_B1_B1

### Relational analysis result of IS_B1_B1_A1_B2_A2_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5618846, upper bound: 0.6584155
time: 0.33 seconds

## Relational analysis of IS_B1_B1_A1_B2_A2_B1_A2_B1_B2

### Relational analysis result of IS_B1_B1_A1_B2_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6363368, upper bound: 0.6770387
time: 0.36 seconds

## BFS IS instance: IS_B2_A2_B1_A1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, 0.0111880, 0.5648615, -0.5536735, 0.5536735
1: -0.0944340, 0.7130072, -0.0944340, 0.7130072, -0.8074412, 0.8074412
2: -0.0312903, 0.7124612, -0.0312903, 0.7124612, -0.7437515, 0.7437515
3: -0.2500148, 0.7301772, -0.2500148, 0.7301772, -0.9801920, 0.9801920
4: -0.1787794, 0.7720095, -0.1787794, 0.7720095, -0.9507889, 0.9507889

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B1_A1_A1_B1_A1_B1_A1

### Relational analysis result of IS_B2_A2_B1_A1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6826392, upper bound: 0.6657280
time: 0.39 seconds

## Relational analysis of IS_B2_A2_B1_A1_A1_B1_A1_B1_A2

### Relational analysis result of IS_B2_A2_B1_A1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6657280
time: 0.38 seconds

## BFS IS instance: IS_B2_A2_B1_A1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, -0.0305669, 0.6706582, -0.6594702, 0.5954285
1: -0.0944340, 0.7130072, -0.1440713, 0.8347254, -0.9291594, 0.8570786
2: -0.0312903, 0.7124612, -0.0742025, 0.8280591, -0.8593494, 0.7866638
3: -0.2500148, 0.7301772, -0.2901301, 0.8312570, -1.0812719, 1.0203073
4: -0.1787794, 0.7720095, -0.2451730, 0.8507983, -1.0295777, 1.0171825

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B1_A1_A1_B1_A1_B2_A1

### Relational analysis result of IS_B2_A2_B1_A1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6826392, upper bound: 0.6657280
time: 0.38 seconds

## Relational analysis of IS_B2_A2_B1_A1_A1_B1_A1_B2_A2

### Relational analysis result of IS_B2_A2_B1_A1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6657280
time: 0.38 seconds

## BFS IS instance: IS_B2_A2_B1_A1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, 0.0111880, 0.5648615, -0.5954285, 0.6594702
1: -0.1440713, 0.8347254, -0.0944340, 0.7130072, -0.8570786, 0.9291594
2: -0.0742025, 0.8280591, -0.0312903, 0.7124612, -0.7866638, 0.8593494
3: -0.2901301, 0.8312570, -0.2500148, 0.7301772, -1.0203073, 1.0812719
4: -0.2451730, 0.8507983, -0.1787794, 0.7720095, -1.0171825, 1.0295777

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B1_A1_A1_B1_A2_B1_A1

### Relational analysis result of IS_B2_A2_B1_A1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6657280
time: 0.38 seconds

## Relational analysis of IS_B2_A2_B1_A1_A1_B1_A2_B1_A2

### Relational analysis result of IS_B2_A2_B1_A1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6657280
time: 0.38 seconds

## BFS IS instance: IS_B2_A2_B1_A1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, -0.0305669, 0.6706582, -0.7012252, 0.7012252
1: -0.1440713, 0.8347254, -0.1440713, 0.8347254, -0.9787967, 0.9787967
2: -0.0742025, 0.8280591, -0.0742025, 0.8280591, -0.9022617, 0.9022617
3: -0.2901301, 0.8312570, -0.2901301, 0.8312570, -1.1213872, 1.1213872
4: -0.2451730, 0.8507983, -0.2451730, 0.8507983, -1.0959713, 1.0959713

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B1_A1_A1_B1_A2_B2_A1

### Relational analysis result of IS_B2_A2_B1_A1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6657280
time: 0.40 seconds

## Relational analysis of IS_B2_A2_B1_A1_A1_B1_A2_B2_A2

### Relational analysis result of IS_B2_A2_B1_A1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6657280
time: 0.39 seconds

## BFS IS instance: IS_B2_A2_B1_A1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, 0.0111880, 0.5648615, -0.5536735, 0.5536735
1: -0.0944340, 0.7130072, -0.0944340, 0.7130072, -0.8074412, 0.8074412
2: -0.0312903, 0.7124612, -0.0312903, 0.7124612, -0.7437515, 0.7437515
3: -0.2500148, 0.7301772, -0.2500148, 0.7301772, -0.9801920, 0.9801920
4: -0.1787794, 0.7720095, -0.1787794, 0.7720095, -0.9507889, 0.9507889

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B1_A1_A1_B2_A1_B1_A1

### Relational analysis result of IS_B2_A2_B1_A1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6826392, upper bound: 0.6657280
time: 0.38 seconds

## Relational analysis of IS_B2_A2_B1_A1_A1_B2_A1_B1_A2

### Relational analysis result of IS_B2_A2_B1_A1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6657280
time: 0.40 seconds

## BFS IS instance: IS_B2_A2_B1_A1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, -0.0305669, 0.6706582, -0.6594702, 0.5954285
1: -0.0944340, 0.7130072, -0.1440713, 0.8347254, -0.9291594, 0.8570786
2: -0.0312903, 0.7124612, -0.0742025, 0.8280591, -0.8593494, 0.7866638
3: -0.2500148, 0.7301772, -0.2901301, 0.8312570, -1.0812719, 1.0203073
4: -0.1787794, 0.7720095, -0.2451730, 0.8507983, -1.0295777, 1.0171825

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B1_A1_A1_B2_A1_B2_A1

### Relational analysis result of IS_B2_A2_B1_A1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6909598, upper bound: 0.6657280
time: 0.37 seconds

## Relational analysis of IS_B2_A2_B1_A1_A1_B2_A1_B2_A2

### Relational analysis result of IS_B2_A2_B1_A1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6700552, upper bound: 0.6657280
time: 0.41 seconds

## BFS IS instance: IS_B2_A2_B1_A1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, 0.0111880, 0.5648615, -0.5954285, 0.6594702
1: -0.1440713, 0.8347254, -0.0944340, 0.7130072, -0.8570786, 0.9291594
2: -0.0742025, 0.8280591, -0.0312903, 0.7124612, -0.7866638, 0.8593494
3: -0.2901301, 0.8312570, -0.2500148, 0.7301772, -1.0203073, 1.0812719
4: -0.2451730, 0.8507983, -0.1787794, 0.7720095, -1.0171825, 1.0295777

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B1_A1_A1_B2_A2_B1_A1

### Relational analysis result of IS_B2_A2_B1_A1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6657280
time: 0.39 seconds

## Relational analysis of IS_B2_A2_B1_A1_A1_B2_A2_B1_A2

### Relational analysis result of IS_B2_A2_B1_A1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6657280
time: 0.37 seconds

## BFS IS instance: IS_B2_A2_B1_A1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, -0.0305669, 0.6706582, -0.7012252, 0.7012252
1: -0.1440713, 0.8347254, -0.1440713, 0.8347254, -0.9787967, 0.9787967
2: -0.0742025, 0.8280591, -0.0742025, 0.8280591, -0.9022617, 0.9022617
3: -0.2901301, 0.8312570, -0.2901301, 0.8312570, -1.1213872, 1.1213872
4: -0.2451730, 0.8507983, -0.2451730, 0.8507983, -1.0959713, 1.0959713

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B1_A1_A1_B2_A2_B2_A1

### Relational analysis result of IS_B2_A2_B1_A1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6700552, upper bound: 0.6657280
time: 0.38 seconds

## Relational analysis of IS_B2_A2_B1_A1_A1_B2_A2_B2_A2

### Relational analysis result of IS_B2_A2_B1_A1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6700552, upper bound: 0.6657280
time: 0.41 seconds

## BFS IS instance: IS_B2_A2_B1_A1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, 0.0111880, 0.5648615, -0.5536735, 0.5536735
1: -0.0944340, 0.7130072, -0.0944340, 0.7130072, -0.8074412, 0.8074412
2: -0.0312903, 0.7124612, -0.0312903, 0.7124612, -0.7437515, 0.7437515
3: -0.2500148, 0.7301772, -0.2500148, 0.7301772, -0.9801920, 0.9801920
4: -0.1787794, 0.7720095, -0.1787794, 0.7720095, -0.9507889, 0.9507889

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B1_A1_A2_B1_A1_B1_A1

### Relational analysis result of IS_B2_A2_B1_A1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6826392, upper bound: 0.6657280
time: 0.38 seconds

## Relational analysis of IS_B2_A2_B1_A1_A2_B1_A1_B1_A2

### Relational analysis result of IS_B2_A2_B1_A1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6657280
time: 0.38 seconds

## BFS IS instance: IS_B2_A2_B1_A1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, -0.0305669, 0.6706582, -0.6594702, 0.5954285
1: -0.0944340, 0.7130072, -0.1440713, 0.8347254, -0.9291594, 0.8570786
2: -0.0312903, 0.7124612, -0.0742025, 0.8280591, -0.8593494, 0.7866638
3: -0.2500148, 0.7301772, -0.2901301, 0.8312570, -1.0812719, 1.0203073
4: -0.1787794, 0.7720095, -0.2451730, 0.8507983, -1.0295777, 1.0171825

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B1_A1_A2_B1_A1_B2_A1

### Relational analysis result of IS_B2_A2_B1_A1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6826392, upper bound: 0.6657280
time: 0.40 seconds

## Relational analysis of IS_B2_A2_B1_A1_A2_B1_A1_B2_A2

### Relational analysis result of IS_B2_A2_B1_A1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6657280
time: 0.38 seconds

## BFS IS instance: IS_B2_A2_B1_A1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, 0.0111880, 0.5648615, -0.5954285, 0.6594702
1: -0.1440713, 0.8347254, -0.0944340, 0.7130072, -0.8570786, 0.9291594
2: -0.0742025, 0.8280591, -0.0312903, 0.7124612, -0.7866638, 0.8593494
3: -0.2901301, 0.8312570, -0.2500148, 0.7301772, -1.0203073, 1.0812719
4: -0.2451730, 0.8507983, -0.1787794, 0.7720095, -1.0171825, 1.0295777

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B1_A1_A2_B1_A2_B1_A1

### Relational analysis result of IS_B2_A2_B1_A1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6657280
time: 0.39 seconds

## Relational analysis of IS_B2_A2_B1_A1_A2_B1_A2_B1_A2

### Relational analysis result of IS_B2_A2_B1_A1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6758026
time: 0.39 seconds

## BFS IS instance: IS_B2_A2_B1_A1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, -0.0305669, 0.6706582, -0.7012252, 0.7012252
1: -0.1440713, 0.8347254, -0.1440713, 0.8347254, -0.9787967, 0.9787967
2: -0.0742025, 0.8280591, -0.0742025, 0.8280591, -0.9022617, 0.9022617
3: -0.2901301, 0.8312570, -0.2901301, 0.8312570, -1.1213872, 1.1213872
4: -0.2451730, 0.8507983, -0.2451730, 0.8507983, -1.0959713, 1.0959713

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B1_A1_A2_B1_A2_B2_A1

### Relational analysis result of IS_B2_A2_B1_A1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6657280
time: 0.39 seconds

## Relational analysis of IS_B2_A2_B1_A1_A2_B1_A2_B2_A2

### Relational analysis result of IS_B2_A2_B1_A1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6758026
time: 0.38 seconds

## BFS IS instance: IS_B2_A2_B1_A1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, 0.0111880, 0.5648615, -0.5536735, 0.5536735
1: -0.0944340, 0.7130072, -0.0944340, 0.7130072, -0.8074412, 0.8074412
2: -0.0312903, 0.7124612, -0.0312903, 0.7124612, -0.7437515, 0.7437515
3: -0.2500148, 0.7301772, -0.2500148, 0.7301772, -0.9801920, 0.9801920
4: -0.1787794, 0.7720095, -0.1787794, 0.7720095, -0.9507889, 0.9507889

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B1_A1_A2_B2_A1_B1_A1

### Relational analysis result of IS_B2_A2_B1_A1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6826392, upper bound: 0.6657280
time: 0.38 seconds

## Relational analysis of IS_B2_A2_B1_A1_A2_B2_A1_B1_A2

### Relational analysis result of IS_B2_A2_B1_A1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6657280
time: 0.39 seconds

## BFS IS instance: IS_B2_A2_B1_A1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, -0.0305669, 0.6706582, -0.6594702, 0.5954285
1: -0.0944340, 0.7130072, -0.1440713, 0.8347254, -0.9291594, 0.8570786
2: -0.0312903, 0.7124612, -0.0742025, 0.8280591, -0.8593494, 0.7866638
3: -0.2500148, 0.7301772, -0.2901301, 0.8312570, -1.0812719, 1.0203073
4: -0.1787794, 0.7720095, -0.2451730, 0.8507983, -1.0295777, 1.0171825

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B1_A1_A2_B2_A1_B2_A1

### Relational analysis result of IS_B2_A2_B1_A1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6909598, upper bound: 0.6657280
time: 0.38 seconds

## Relational analysis of IS_B2_A2_B1_A1_A2_B2_A1_B2_A2

### Relational analysis result of IS_B2_A2_B1_A1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6700552, upper bound: 0.6657280
time: 0.40 seconds

## BFS IS instance: IS_B2_A2_B1_A1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, 0.0111880, 0.5648615, -0.5954285, 0.6594702
1: -0.1440713, 0.8347254, -0.0944340, 0.7130072, -0.8570786, 0.9291594
2: -0.0742025, 0.8280591, -0.0312903, 0.7124612, -0.7866638, 0.8593494
3: -0.2901301, 0.8312570, -0.2500148, 0.7301772, -1.0203073, 1.0812719
4: -0.2451730, 0.8507983, -0.1787794, 0.7720095, -1.0171825, 1.0295777

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B1_A1_A2_B2_A2_B1_A1

### Relational analysis result of IS_B2_A2_B1_A1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6657280
time: 0.41 seconds

## Relational analysis of IS_B2_A2_B1_A1_A2_B2_A2_B1_A2

### Relational analysis result of IS_B2_A2_B1_A1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6609832, upper bound: 0.6758026
time: 0.38 seconds

## BFS IS instance: IS_B2_A2_B1_A1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, -0.0305669, 0.6706582, -0.7012252, 0.7012252
1: -0.1440713, 0.8347254, -0.1440713, 0.8347254, -0.9787967, 0.9787967
2: -0.0742025, 0.8280591, -0.0742025, 0.8280591, -0.9022617, 0.9022617
3: -0.2901301, 0.8312570, -0.2901301, 0.8312570, -1.1213872, 1.1213872
4: -0.2451730, 0.8507983, -0.2451730, 0.8507983, -1.0959713, 1.0959713

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B1_A1_A2_B2_A2_B2_A1

### Relational analysis result of IS_B2_A2_B1_A1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6700552, upper bound: 0.6657280
time: 0.37 seconds

## Relational analysis of IS_B2_A2_B1_A1_A2_B2_A2_B2_A2

### Relational analysis result of IS_B2_A2_B1_A1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6700552, upper bound: 0.6758026
time: 0.39 seconds

## BFS IS instance: IS_B2_A2_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0061345, 0.6507710, 0.0111880, 0.5648615, -0.5587270, 0.6395830
1: -0.0976210, 0.8106803, -0.0944340, 0.7130072, -0.8106282, 0.9051143
2: -0.0417109, 0.7880348, -0.0312903, 0.7124612, -0.7541721, 0.8193251
3: -0.2534082, 0.7965333, -0.2500148, 0.7301772, -0.9835854, 1.0465481
4: -0.2023642, 0.8128391, -0.1787794, 0.7720095, -0.9743737, 0.9916185

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_B2_A2_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6529501, upper bound: 0.6886226
time: 0.39 seconds

## Relational analysis of IS_B2_A2_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_B2_A2_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6529501, upper bound: 0.6647563
time: 0.40 seconds

## BFS IS instance: IS_B2_A2_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0329778, 0.7275667, 0.0111880, 0.5648615, -0.5978394, 0.7163787
1: -0.1447692, 0.8997444, -0.0944340, 0.7130072, -0.8577764, 0.9941784
2: -0.0813575, 0.8785636, -0.0312903, 0.7124612, -0.7938187, 0.9098539
3: -0.2928538, 0.8771129, -0.2500148, 0.7301772, -1.0230310, 1.1271276
4: -0.2609611, 0.8821078, -0.1787794, 0.7720095, -1.0329705, 1.0608871

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_B2_A2_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6529501, upper bound: 0.6886226
time: 0.38 seconds

## Relational analysis of IS_B2_A2_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_B2_A2_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6529501, upper bound: 0.6647563
time: 0.39 seconds

## BFS IS instance: IS_B2_A2_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0061345, 0.6507710, -0.0305669, 0.6706582, -0.6645237, 0.6813380
1: -0.0976210, 0.8106803, -0.1440713, 0.8347254, -0.9323463, 0.9547516
2: -0.0417109, 0.7880348, -0.0742025, 0.8280591, -0.8697700, 0.8622373
3: -0.2534082, 0.7965333, -0.2901301, 0.8312570, -1.0846653, 1.0866635
4: -0.2023642, 0.8128391, -0.2451730, 0.8507983, -1.0531626, 1.0580120

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_B2_A2_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6529501, upper bound: 0.6647563
time: 0.39 seconds

## Relational analysis of IS_B2_A2_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_B2_A2_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6529501, upper bound: 0.6647563
time: 0.40 seconds

## BFS IS instance: IS_B2_A2_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0329778, 0.7275667, -0.0305669, 0.6706582, -0.7036361, 0.7581336
1: -0.1447692, 0.8997444, -0.1440713, 0.8347254, -0.9794946, 1.0438157
2: -0.0813575, 0.8785636, -0.0742025, 0.8280591, -0.9094166, 0.9527662
3: -0.2928538, 0.8771129, -0.2901301, 0.8312570, -1.1241109, 1.1672430
4: -0.2609611, 0.8821078, -0.2451730, 0.8507983, -1.1117594, 1.1272807

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_B2_A2_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6529501, upper bound: 0.6647563
time: 0.38 seconds

## Relational analysis of IS_B2_A2_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_B2_A2_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6529501, upper bound: 0.6647563
time: 0.41 seconds

## BFS IS instance: IS_B2_A2_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0061345, 0.6507710, 0.0111880, 0.5648615, -0.5587270, 0.6395830
1: -0.0976210, 0.8106803, -0.0944340, 0.7130072, -0.8106282, 0.9051143
2: -0.0417109, 0.7880348, -0.0312903, 0.7124612, -0.7541721, 0.8193251
3: -0.2534082, 0.7965333, -0.2500148, 0.7301772, -0.9835854, 1.0465481
4: -0.2023642, 0.8128391, -0.1787794, 0.7720095, -0.9743737, 0.9916185

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_B2_A2_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6529501, upper bound: 0.6886226
time: 0.39 seconds

## Relational analysis of IS_B2_A2_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_B2_A2_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6529501, upper bound: 0.6647563
time: 0.40 seconds

## BFS IS instance: IS_B2_A2_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0329778, 0.7304660, 0.0111880, 0.5648615, -0.5978394, 0.7192780
1: -0.1447692, 0.9029138, -0.0944340, 0.7130072, -0.8577764, 0.9973478
2: -0.0813575, 0.8816222, -0.0312903, 0.7124612, -0.7938187, 0.9129125
3: -0.2928538, 0.8801655, -0.2500148, 0.7301772, -1.0230310, 1.1301802
4: -0.2609611, 0.8848863, -0.1787794, 0.7720095, -1.0329705, 1.0636656

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_B2_A2_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6442375, upper bound: 0.6966600
time: 0.37 seconds

## Relational analysis of IS_B2_A2_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_B2_A2_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6529501, upper bound: 0.6729556
time: 0.37 seconds

## BFS IS instance: IS_B2_A2_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0061345, 0.6507710, -0.0305669, 0.6706582, -0.6645237, 0.6813380
1: -0.0976210, 0.8106803, -0.1440713, 0.8347254, -0.9323463, 0.9547516
2: -0.0417109, 0.7880348, -0.0742025, 0.8280591, -0.8697700, 0.8622373
3: -0.2534082, 0.7965333, -0.2901301, 0.8312570, -1.0846653, 1.0866635
4: -0.2023642, 0.8128391, -0.2451730, 0.8507983, -1.0531626, 1.0580120

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_B2_A2_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6529501, upper bound: 0.6647563
time: 0.39 seconds

## Relational analysis of IS_B2_A2_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_B2_A2_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6529501, upper bound: 0.6647563
time: 0.41 seconds

## BFS IS instance: IS_B2_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0329778, 0.7304660, -0.0305669, 0.6706582, -0.7036361, 0.7610329
1: -0.1447692, 0.9029138, -0.1440713, 0.8347254, -0.9794946, 1.0469851
2: -0.0813575, 0.8816222, -0.0742025, 0.8280591, -0.9094166, 0.9558247
3: -0.2928538, 0.8801655, -0.2901301, 0.8312570, -1.1241109, 1.1702956
4: -0.2609611, 0.8848863, -0.2451730, 0.8507983, -1.1117594, 1.1300592

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_B2_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6529501, upper bound: 0.6729556
time: 0.37 seconds

## Relational analysis of IS_B2_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_B2_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6410595, upper bound: 0.6729556
time: 0.38 seconds

## BFS IS instance: IS_B2_A2_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0061345, 0.6507710, 0.0111880, 0.5648615, -0.5587270, 0.6395830
1: -0.0976210, 0.8106803, -0.0944340, 0.7130072, -0.8106282, 0.9051143
2: -0.0417109, 0.7880348, -0.0312903, 0.7124612, -0.7541721, 0.8193251
3: -0.2534082, 0.7965333, -0.2500148, 0.7301772, -0.9835854, 1.0465481
4: -0.2023642, 0.8128391, -0.1787794, 0.7720095, -0.9743737, 0.9916185

Time for backsubstitution: 1.89 seconds
Binary search (step 1): status=Status.UNKNOWN, low=0.0017312, high=0.0467529, mid=0.0467529, abs_max=0.7819017171859741
rel_dist={0: [-0.7029083079558228, 0.7029083079558232]}

## Binary search (step 2) starts
Candidate diff: 0.0242421


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6862236, upper bound: 0.6933877
time: 0.33 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6937108, upper bound: 0.6937108
time: 0.35 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.85 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 0.85
Output dim: 0, lower bound: -0.6862236, upper bound: 0.6933877
IS_B2, status: Status.UNKNOWN, split count: 1, time: 0.85
Output dim: 0, lower bound: -0.6937108, upper bound: 0.6937108

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -0.0226004, 0.6992232, -0.0017378, 0.9577341, -0.9803345, 0.7009610
1: -0.1409330, 0.8667333, -0.1117451, 1.2035816, -1.3445146, 0.9784784
2: -0.0639827, 0.8644909, -0.0625789, 1.0695953, -1.1335781, 0.9270698
3: -0.3000648, 0.8557575, -0.2743888, 1.0868173, -1.3868821, 1.1301463
4: -0.2339711, 0.8990978, -0.2543875, 0.9355542, -1.1695254, 1.1534853

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_B1

### Relational analysis result of IS_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6514900, upper bound: 0.6923342
time: 0.29 seconds

## Relational analysis of IS_B1_B2

### Relational analysis result of IS_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6521396, upper bound: 0.6592597
time: 0.31 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -0.0323386, 0.7495631, -0.0124662, 0.7046853, -0.7370239, 0.7620293
1: -0.1552227, 0.9255341, -0.1260490, 0.8739170, -1.0291396, 1.0515832
2: -0.0741289, 0.9267865, -0.0592167, 0.8673251, -0.9414539, 0.9860032
3: -0.3143692, 0.9027434, -0.2830637, 0.8561093, -1.1704785, 1.1858070
4: -0.2515757, 0.9410419, -0.2274244, 0.8807485, -1.1323242, 1.1684663

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6933184, upper bound: 0.6862236
time: 0.30 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6933184, upper bound: 0.6937108
time: 0.31 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.32 seconds
IS_B1_B1, status: Status.UNKNOWN, split count: 2, time: 2.32
Output dim: 0, lower bound: -0.6514900, upper bound: 0.6923342
IS_B1_B2, status: Status.VERIFIED, split count: 2, time: 2.32
Output dim: 0, lower bound: -0.6521396, upper bound: 0.6592597
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 2.32
Output dim: 0, lower bound: -0.6933184, upper bound: 0.6862236
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 2.32
Output dim: 0, lower bound: -0.6933184, upper bound: 0.6937108

## BFS IS instance: IS_B1_B1

### Backsubstitution after applying IS history:
0: -0.0226004, 0.6992232, -0.0004439, 0.9531150, -0.9757154, 0.6996671
1: -0.1409330, 0.8667333, -0.1101191, 1.1983986, -1.3393316, 0.9768524
2: -0.0639827, 0.8644909, -0.0610516, 1.0641370, -1.1281197, 0.9255425
3: -0.3000648, 0.8557575, -0.2731256, 1.0821323, -1.3821971, 1.1288831
4: -0.2339711, 0.8990978, -0.2521458, 0.9308868, -1.1648579, 1.1512436

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_B1_A1

### Relational analysis result of IS_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6514900, upper bound: 0.6866269
time: 0.32 seconds

## Relational analysis of IS_B1_B1_A2

### Relational analysis result of IS_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6514900, upper bound: 0.6923342
time: 0.29 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -0.0017378, 0.9577341, -0.0124662, 0.7046853, -0.7064232, 0.9702003
1: -0.1117451, 1.2035816, -0.1260490, 0.8739170, -0.9856621, 1.3296306
2: -0.0625789, 1.0695953, -0.0592167, 0.8673251, -0.9299040, 1.1288121
3: -0.2743888, 1.0868173, -0.2830637, 0.8561093, -1.1304981, 1.3698809
4: -0.2543875, 0.9355542, -0.2274244, 0.8807485, -1.1351360, 1.1629786

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_A1

### Relational analysis result of IS_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6862148, upper bound: 0.6514900
time: 0.31 seconds

## Relational analysis of IS_B2_A1_A2

### Relational analysis result of IS_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6521396, upper bound: 0.6521396
time: 0.31 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -0.0124662, 0.7046853, -0.0124662, 0.7046853, -0.7171515, 0.7171515
1: -0.1260490, 0.8739170, -0.1260490, 0.8739170, -0.9999660, 0.9999660
2: -0.0592167, 0.8673251, -0.0592167, 0.8673251, -0.9265418, 0.9265418
3: -0.2830637, 0.8561093, -0.2830637, 0.8561093, -1.1391729, 1.1391729
4: -0.2274244, 0.8807485, -0.2274244, 0.8807485, -1.1081729, 1.1081729

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_A2_A1

### Relational analysis result of IS_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6809564, upper bound: 0.6917761
time: 0.31 seconds

## Relational analysis of IS_B2_A2_A2

### Relational analysis result of IS_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6809564, upper bound: 0.6920244
time: 0.32 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.02 seconds
IS_B1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.02
Output dim: 0, lower bound: -0.6514900, upper bound: 0.6866269
IS_B1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.02
Output dim: 0, lower bound: -0.6514900, upper bound: 0.6923342
IS_B2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 3.02
Output dim: 0, lower bound: -0.6862148, upper bound: 0.6514900
IS_B2_A1_A2, status: Status.VERIFIED, split count: 3, time: 3.02
Output dim: 0, lower bound: -0.6521396, upper bound: 0.6521396
IS_B2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 3.02
Output dim: 0, lower bound: -0.6809564, upper bound: 0.6917761
IS_B2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 3.02
Output dim: 0, lower bound: -0.6809564, upper bound: 0.6920244

## BFS IS instance: IS_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0017378, 0.9577341, -0.0004439, 0.9531150, -0.9548528, 0.9581780
1: -0.1117451, 1.2035816, -0.1101191, 1.1983986, -1.3101437, 1.3137007
2: -0.0625789, 1.0695953, -0.0610516, 1.0641370, -1.1267159, 1.1306469
3: -0.2743888, 1.0868173, -0.2731256, 1.0821323, -1.3565211, 1.3599429
4: -0.2543875, 0.9355542, -0.2521458, 0.9308868, -1.1852744, 1.1877000

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B1_B1_A1_B1

### Relational analysis result of IS_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6083539, upper bound: 0.6809481
time: 0.32 seconds

## Relational analysis of IS_B1_B1_A1_B2

### Relational analysis result of IS_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6513009, upper bound: 0.6860459
time: 0.32 seconds

## BFS IS instance: IS_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0124662, 0.7046853, -0.0004439, 0.9531150, -0.9655812, 0.7051293
1: -0.1260490, 0.8739170, -0.1101191, 1.1983986, -1.3244476, 0.9840361
2: -0.0592167, 0.8673251, -0.0610516, 1.0641370, -1.1233537, 0.9283767
3: -0.2830637, 0.8561093, -0.2731256, 1.0821323, -1.3651960, 1.1292349
4: -0.2274244, 0.8807485, -0.2521458, 0.9308868, -1.1583111, 1.1328943

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_B1_A2_A1

### Relational analysis result of IS_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6399833, upper bound: 0.6774702
time: 0.31 seconds

## Relational analysis of IS_B1_B1_A2_A2

### Relational analysis result of IS_B1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6397003, upper bound: 0.6609718
time: 0.31 seconds

## BFS IS instance: IS_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0004439, 0.9531150, -0.0124662, 0.7046853, -0.7051293, 0.9655812
1: -0.1101191, 1.1983986, -0.1260490, 0.8739170, -0.9840361, 1.3244476
2: -0.0610516, 1.0641370, -0.0592167, 0.8673251, -0.9283767, 1.1233537
3: -0.2731256, 1.0821323, -0.2830637, 0.8561093, -1.1292349, 1.3651960
4: -0.2521458, 0.9308868, -0.2274244, 0.8807485, -1.1328943, 1.1583111

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_A1_B1

### Relational analysis result of IS_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6774702, upper bound: 0.6399833
time: 0.34 seconds

## Relational analysis of IS_B2_A1_A1_B2

### Relational analysis result of IS_B2_A1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6609718, upper bound: 0.6397003
time: 0.33 seconds

## BFS IS instance: IS_B2_A2_A1

### Backsubstitution after applying IS history:
0: 0.0054424, 0.6499398, -0.0124662, 0.7046853, -0.6992429, 0.6624060
1: -0.1022949, 0.8109143, -0.1260490, 0.8739170, -0.9762119, 0.9369633
2: -0.0393627, 0.7986160, -0.0592167, 0.8673251, -0.9066877, 0.8578327
3: -0.2590857, 0.7998861, -0.2830637, 0.8561093, -1.1151949, 1.0829498
4: -0.1939640, 0.8231770, -0.2274244, 0.8807485, -1.0747125, 1.0506014

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A1_B1

### Relational analysis result of IS_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6717107, upper bound: 0.6584709
time: 0.34 seconds

## Relational analysis of IS_B2_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_A2_A1_B1

### Relational analysis result of IS_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6798252, upper bound: 0.6917454
time: 0.37 seconds

## Relational analysis of IS_B2_A2_A1_B2

### Relational analysis result of IS_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6798252, upper bound: 0.6917725
time: 0.35 seconds

## BFS IS instance: IS_B2_A2_A2

### Backsubstitution after applying IS history:
0: 0.0004525, 0.7221838, -0.0050101, 0.6788989, -0.6784464, 0.7271940
1: -0.1048932, 0.8939738, -0.1146951, 0.8445600, -0.9494532, 1.0086689
2: -0.0500441, 0.8598962, -0.0525062, 0.8296809, -0.8797249, 0.9124024
3: -0.2612641, 0.8577256, -0.2720425, 0.8305112, -1.0917752, 1.1297681
4: -0.2176540, 0.8560207, -0.2159448, 0.8508037, -1.0684577, 1.0719655

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A2_A2_B1

### Relational analysis result of IS_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6515660, upper bound: 0.6746649
time: 0.36 seconds

## Relational analysis of IS_B2_A2_A2_B2

### Relational analysis result of IS_B2_A2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6408665, upper bound: 0.6408665
time: 0.34 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.27 seconds
IS_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 0, lower bound: -0.6083539, upper bound: 0.6809481
IS_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 0, lower bound: -0.6513009, upper bound: 0.6860459
IS_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 0, lower bound: -0.6399833, upper bound: 0.6774702
IS_B1_B1_A2_A2, status: Status.VERIFIED, split count: 4, time: 3.27
Output dim: 0, lower bound: -0.6397003, upper bound: 0.6609718
IS_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 0, lower bound: -0.6774702, upper bound: 0.6399833
IS_B2_A1_A1_B2, status: Status.VERIFIED, split count: 4, time: 3.27
Output dim: 0, lower bound: -0.6609718, upper bound: 0.6397003
IS_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 0, lower bound: -0.6798252, upper bound: 0.6917454
IS_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 0, lower bound: -0.6798252, upper bound: 0.6917725
IS_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 0, lower bound: -0.6515660, upper bound: 0.6746649
IS_B2_A2_A2_B2, status: Status.VERIFIED, split count: 4, time: 3.27
Output dim: 0, lower bound: -0.6408665, upper bound: 0.6408665

## BFS IS instance: IS_B1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0017378, 0.9577341, 0.0209117, 0.9025295, -0.9042673, 0.9368224
1: -0.1117451, 1.2035816, -0.0776985, 1.1409044, -1.2526495, 1.2812800
2: -0.0625789, 1.0695953, -0.0370891, 0.9958014, -1.0583804, 1.1066844
3: -0.2743888, 1.0868173, -0.2410901, 1.0262597, -1.3006485, 1.3279073
4: -0.2543875, 0.9355542, -0.2204622, 0.8550285, -1.1094160, 1.1560163

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_B1_A1_B1_A1

### Relational analysis result of IS_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6084165, upper bound: 0.6802498
time: 0.32 seconds

## Relational analysis of IS_B1_B1_A1_B1_A2

### Relational analysis result of IS_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6084165, upper bound: 0.6809481
time: 0.30 seconds

## BFS IS instance: IS_B1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0016496, 0.9380729, -0.0032020, 1.0339718, -1.0323222, 0.9412749
1: -0.1070411, 1.1816590, -0.1091664, 1.3081079, -1.4151490, 1.2908254
2: -0.0588033, 1.0435266, -0.0733573, 1.1349654, -1.1937687, 1.1168840
3: -0.2689416, 1.0690438, -0.2708402, 1.1731939, -1.4421356, 1.3398839
4: -0.2483846, 0.9189640, -0.2933947, 0.9566256, -1.2050102, 1.2123587

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_B1_A1_B2_A1

### Relational analysis result of IS_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6400006, upper bound: 0.6714281
time: 0.38 seconds

## Relational analysis of IS_B1_B1_A1_B2_A2

### Relational analysis result of IS_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6415344, upper bound: 0.6637412
time: 0.36 seconds

## BFS IS instance: IS_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0107870, 0.6989747, -0.0004439, 0.9531150, -0.9639020, 0.6994187
1: -0.1241605, 0.8672397, -0.1101191, 1.1983986, -1.3225591, 0.9773588
2: -0.0573094, 0.8610535, -0.0610516, 1.0641370, -1.1214464, 0.9221051
3: -0.2815993, 0.8504646, -0.2731256, 1.0821323, -1.3637316, 1.1235902
4: -0.2244935, 0.8760955, -0.2521458, 0.9308868, -1.1553802, 1.1282413

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_B1_A2_A1_B1

### Relational analysis result of IS_B1_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6397003, upper bound: 0.6609718
time: 0.32 seconds

## Relational analysis of IS_B1_B1_A2_A1_B2

### Relational analysis result of IS_B1_B1_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6397003, upper bound: 0.6609718
time: 0.30 seconds

## BFS IS instance: IS_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0004439, 0.9531150, -0.0107870, 0.6989747, -0.6994187, 0.9639020
1: -0.1101191, 1.1983986, -0.1241605, 0.8672397, -0.9773588, 1.3225591
2: -0.0610516, 1.0641370, -0.0573094, 0.8610535, -0.9221051, 1.1214464
3: -0.2731256, 1.0821323, -0.2815993, 0.8504646, -1.1235902, 1.3637316
4: -0.2521458, 0.9308868, -0.2244935, 0.8760955, -1.1282413, 1.1553802

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_A1_B1_A1

### Relational analysis result of IS_B2_A1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6609718, upper bound: 0.6397003
time: 0.29 seconds

## Relational analysis of IS_B2_A1_A1_B1_A2

### Relational analysis result of IS_B2_A1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6609718, upper bound: 0.6397003
time: 0.32 seconds

## BFS IS instance: IS_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0054424, 0.6499398, 0.0054424, 0.6499398, -0.6444974, 0.6444974
1: -0.1022949, 0.8109143, -0.1022949, 0.8109143, -0.9132092, 0.9132092
2: -0.0393627, 0.7986160, -0.0393627, 0.7986160, -0.8379787, 0.8379787
3: -0.2590857, 0.7998861, -0.2590857, 0.7998861, -1.0589718, 1.0589718
4: -0.1939640, 0.8231770, -0.1939640, 0.8231770, -1.0171410, 1.0171410

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A1_B1_B1

### Relational analysis result of IS_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6717107, upper bound: 0.6584709
time: 0.34 seconds

## Relational analysis of IS_B2_A2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 5

Time for candidate selection: 3.61 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_A1_B1_B1

### Relational analysis result of IS_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6563800, upper bound: 0.6910580
time: 0.37 seconds

## Relational analysis of IS_B2_A2_A1_B1_B2

### Relational analysis result of IS_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6659790, upper bound: 0.6672156
time: 0.38 seconds

## BFS IS instance: IS_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0054424, 0.6499398, 0.0004525, 0.7221838, -0.7167414, 0.6494873
1: -0.1022949, 0.8109143, -0.1048932, 0.8939738, -0.9962687, 0.9158075
2: -0.0393627, 0.7986160, -0.0500441, 0.8598962, -0.8992589, 0.8486601
3: -0.2590857, 0.7998861, -0.2612641, 0.8577256, -1.1168113, 1.0611502
4: -0.1939640, 0.8231770, -0.2176540, 0.8560207, -1.0499847, 1.0408310

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A1_B2_B1

### Relational analysis result of IS_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6717107, upper bound: 0.6584709
time: 0.35 seconds

## Relational analysis of IS_B2_A2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5

Time for candidate selection: 3.80 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_A1_B2_A1

### Relational analysis result of IS_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6891706, upper bound: 0.6597359
time: 0.34 seconds

## Relational analysis of IS_B2_A2_A1_B2_A2

### Relational analysis result of IS_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6659790, upper bound: 0.6683080
time: 0.37 seconds

## BFS IS instance: IS_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0004525, 0.7221838, -0.0032587, 0.6730454, -0.6725929, 0.7254425
1: -0.1048932, 0.8939738, -0.1126518, 0.8378215, -0.9427147, 1.0066257
2: -0.0500441, 0.8598962, -0.0505674, 0.8232774, -0.8733214, 0.9104636
3: -0.2612641, 0.8577256, -0.2704487, 0.8248124, -1.0860765, 1.1281743
4: -0.2176540, 0.8560207, -0.2129583, 0.8459927, -1.0636468, 1.0689790

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_A2_B1_B1

### Relational analysis result of IS_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6508189, upper bound: 0.6746561
time: 0.34 seconds

## Relational analysis of IS_B2_A2_A2_B1_B2

### Relational analysis result of IS_B2_A2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6515660, upper bound: 0.6411495
time: 0.37 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.23 seconds
IS_B1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.23
Output dim: 0, lower bound: -0.6084165, upper bound: 0.6802498
IS_B1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.23
Output dim: 0, lower bound: -0.6084165, upper bound: 0.6809481
IS_B1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.23
Output dim: 0, lower bound: -0.6400006, upper bound: 0.6714281
IS_B1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.23
Output dim: 0, lower bound: -0.6415344, upper bound: 0.6637412
IS_B1_B1_A2_A1_B1, status: Status.VERIFIED, split count: 5, time: 3.23
Output dim: 0, lower bound: -0.6397003, upper bound: 0.6609718
IS_B1_B1_A2_A1_B2, status: Status.VERIFIED, split count: 5, time: 3.23
Output dim: 0, lower bound: -0.6397003, upper bound: 0.6609718
IS_B2_A1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.23
Output dim: 0, lower bound: -0.6609718, upper bound: 0.6397003
IS_B2_A1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.23
Output dim: 0, lower bound: -0.6609718, upper bound: 0.6397003
IS_B2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.23
Output dim: 0, lower bound: -0.6563800, upper bound: 0.6910580
IS_B2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.23
Output dim: 0, lower bound: -0.6659790, upper bound: 0.6672156
IS_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.23
Output dim: 0, lower bound: -0.6891706, upper bound: 0.6597359
IS_B2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.23
Output dim: 0, lower bound: -0.6659790, upper bound: 0.6683080
IS_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.23
Output dim: 0, lower bound: -0.6508189, upper bound: 0.6746561
IS_B2_A2_A2_B1_B2, status: Status.VERIFIED, split count: 5, time: 3.23
Output dim: 0, lower bound: -0.6515660, upper bound: 0.6411495

## BFS IS instance: IS_B1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004439, 0.9531150, 0.0209117, 0.9025295, -0.9029734, 0.9322033
1: -0.1101191, 1.1983986, -0.0776985, 1.1409044, -1.2510235, 1.2760971
2: -0.0610516, 1.0641370, -0.0370891, 0.9958014, -1.0568531, 1.1012261
3: -0.2731256, 1.0821323, -0.2410901, 1.0262597, -1.2993853, 1.3232224
4: -0.2521458, 0.9308868, -0.2204622, 0.8550285, -1.1071743, 1.1513491

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_B1_A1_B1_A1_B1

### Relational analysis result of IS_B1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5849023, upper bound: 0.6688821
time: 0.31 seconds

## Relational analysis of IS_B1_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 5

Time for candidate selection: 3.06 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B1_A1_B1

### Relational analysis result of IS_B1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6036691, upper bound: 0.6795986
time: 0.31 seconds

## Relational analysis of IS_B1_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 47

## Relational analysis of IS_B1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 47

## Relational analysis of IS_B1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_B1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_B1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_B1_B1_A1_B1_A1_B1

### Relational analysis result of IS_B1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5555441, upper bound: 0.6794671
time: 0.31 seconds

## Relational analysis of IS_B1_B1_A1_B1_A1_B2

### Relational analysis result of IS_B1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5963971, upper bound: 0.6745666
time: 0.32 seconds

## BFS IS instance: IS_B1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0001364, 0.9854338, 0.0209117, 0.9025295, -0.9026659, 0.9645221
1: -0.1093853, 1.2390206, -0.0776985, 1.1409044, -1.2502897, 1.3167191
2: -0.0645683, 1.0958974, -0.0370891, 0.9958014, -1.0603697, 1.1329865
3: -0.2715011, 1.1123974, -0.2410901, 1.0262597, -1.2977607, 1.3534875
4: -0.2628224, 0.9425689, -0.2204622, 0.8550285, -1.1178509, 1.1630311

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_B1_A1_B1_A2_B1

### Relational analysis result of IS_B1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5849023, upper bound: 0.6727472
time: 0.33 seconds

## Relational analysis of IS_B1_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 5

Time for candidate selection: 3.27 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B1_A2_B1

### Relational analysis result of IS_B1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6036691, upper bound: 0.6803481
time: 0.35 seconds

## Relational analysis of IS_B1_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 47

## Relational analysis of IS_B1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_B1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 47

## Relational analysis of IS_B1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_B1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_B1_B1_A1_B1_A2_B1

### Relational analysis result of IS_B1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5555441, upper bound: 0.6801654
time: 0.32 seconds

## Relational analysis of IS_B1_B1_A1_B1_A2_B2

### Relational analysis result of IS_B1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5963971, upper bound: 0.6753137
time: 0.35 seconds

## BFS IS instance: IS_B1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0033271, 0.9327366, -0.0032020, 1.0339718, -1.0306447, 0.9359386
1: -0.1048460, 1.1755776, -0.1091664, 1.3081079, -1.4129539, 1.2847440
2: -0.0569408, 1.0371912, -0.0733573, 1.1349654, -1.1919062, 1.1105485
3: -0.2671270, 1.0637906, -0.2708402, 1.1731939, -1.4403210, 1.3346307
4: -0.2460732, 0.9137013, -0.2933947, 0.9566256, -1.2026988, 1.2070960

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_B1_A1_B2_A1_A1

### Relational analysis result of IS_B1_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6400006, upper bound: 0.6706810
time: 0.35 seconds

## Relational analysis of IS_B1_B1_A1_B2_A1_A2

### Relational analysis result of IS_B1_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6400006, upper bound: 0.6714281
time: 0.38 seconds

## BFS IS instance: IS_B1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0426288, 1.8511992, -0.0020766, 1.0276093, -1.0702381, 1.8532758
1: -0.1642268, 2.2651000, -0.1077940, 1.3009942, -1.4652209, 2.3728940
2: -0.1343973, 2.0305896, -0.0720024, 1.1275635, -1.2619607, 2.1025920
3: -0.3265591, 1.8406374, -0.2695229, 1.1675057, -1.4940648, 2.1101604
4: -0.4800941, 1.4741251, -0.2915139, 0.9509519, -1.4310460, 1.7656391

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_B1_A1_B2_A2_B1

### Relational analysis result of IS_B1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6415345, upper bound: 0.6637412
time: 0.32 seconds

## Relational analysis of IS_B1_B1_A1_B2_A2_B2

### Relational analysis result of IS_B1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6415345, upper bound: 0.6637412
time: 0.33 seconds

## BFS IS instance: IS_B2_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0054424, 0.6499398, 0.0111880, 0.5648615, -0.5594192, 0.6387517
1: -0.1022949, 0.8109143, -0.0944340, 0.7130072, -0.8153021, 0.9053483
2: -0.0393627, 0.7986160, -0.0312903, 0.7124612, -0.7518239, 0.8299063
3: -0.2590857, 0.7998861, -0.2500148, 0.7301772, -0.9892629, 1.0499009
4: -0.1939640, 0.8231770, -0.1787794, 0.7720095, -0.9659735, 1.0019563

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 29

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_A1_B1_B1_A1

### Relational analysis result of IS_B2_A2_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6586239, upper bound: 0.6594773
time: 0.37 seconds

## Relational analysis of IS_B2_A2_A1_B1_B1_A2

### Relational analysis result of IS_B2_A2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6586239, upper bound: 0.6698469
time: 0.39 seconds

## BFS IS instance: IS_B2_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0054424, 0.6499398, -0.0305669, 0.6706582, -0.6652158, 0.6805067
1: -0.1022949, 0.8109143, -0.1440713, 0.8347254, -0.9370203, 0.9549856
2: -0.0393627, 0.7986160, -0.0742025, 0.8280591, -0.8674218, 0.8728185
3: -0.2590857, 0.7998861, -0.2901301, 0.8312570, -1.0903428, 1.0900162
4: -0.1939640, 0.8231770, -0.2451730, 0.8507983, -1.0447624, 1.0683500

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 29

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_A1_B1_B2_A1

### Relational analysis result of IS_B2_A2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6683065, upper bound: 0.6594773
time: 0.38 seconds

## Relational analysis of IS_B2_A2_A1_B1_B2_A2

### Relational analysis result of IS_B2_A2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6683065, upper bound: 0.6698469
time: 0.36 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, 0.0004525, 0.7221838, -0.7109958, 0.5644090
1: -0.0944340, 0.7130072, -0.1048932, 0.8939738, -0.9884079, 0.8179004
2: -0.0312903, 0.7124612, -0.0500441, 0.8598962, -0.8911865, 0.7625053
3: -0.2500148, 0.7301772, -0.2612641, 0.8577256, -1.1077404, 0.9914413
4: -0.1787794, 0.7720095, -0.2176540, 0.8560207, -1.0348001, 0.9896635

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_A1_B2_A1_B1

### Relational analysis result of IS_B2_A2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6563801, upper bound: 0.6597357
time: 0.36 seconds

## Relational analysis of IS_B2_A2_A1_B2_A1_B2

### Relational analysis result of IS_B2_A2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6563801, upper bound: 0.6597357
time: 0.37 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, 0.0004525, 0.7221838, -0.7527508, 0.6702057
1: -0.1440713, 0.8347254, -0.1048932, 0.8939738, -1.0380452, 0.9396186
2: -0.0742025, 0.8280591, -0.0500441, 0.8598962, -0.9340987, 0.8781032
3: -0.2901301, 0.8312570, -0.2612641, 0.8577256, -1.1478558, 1.0925212
4: -0.2451730, 0.8507983, -0.2176540, 0.8560207, -1.1011937, 1.0684524

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 29

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_A1_B2_A2_B1

### Relational analysis result of IS_B2_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6563801, upper bound: 0.6683078
time: 0.38 seconds

## Relational analysis of IS_B2_A2_A1_B2_A2_B2

### Relational analysis result of IS_B2_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6563801, upper bound: 0.6683079
time: 0.38 seconds

## BFS IS instance: IS_B2_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0004525, 0.7221838, -0.0020461, 0.6678331, -0.6673806, 0.7242299
1: -0.1048932, 0.8939738, -0.1111848, 0.8318986, -0.9367918, 1.0051587
2: -0.0500441, 0.8598962, -0.0491319, 0.8174112, -0.8674553, 0.9090281
3: -0.2612641, 0.8577256, -0.2692239, 0.8197592, -1.0810232, 1.1269495
4: -0.2176540, 0.8560207, -0.2104611, 0.8413419, -1.0589958, 1.0664818

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_A2_A2_B1_B1_B1

### Relational analysis result of IS_B2_A2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5671587, upper bound: 0.6693510
time: 0.44 seconds

## Relational analysis of IS_B2_A2_A2_B1_B1_B2

### Relational analysis result of IS_B2_A2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5671587, upper bound: 0.6746561
time: 0.37 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.29 seconds
IS_B1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 0, lower bound: -0.5555441, upper bound: 0.6794671
IS_B1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 0, lower bound: -0.5963971, upper bound: 0.6745666
IS_B1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 0, lower bound: -0.5555441, upper bound: 0.6801654
IS_B1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 0, lower bound: -0.5963971, upper bound: 0.6753137
IS_B1_B1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 0, lower bound: -0.6400006, upper bound: 0.6706810
IS_B1_B1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 0, lower bound: -0.6400006, upper bound: 0.6714281
IS_B1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 0, lower bound: -0.6415345, upper bound: 0.6637412
IS_B1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 0, lower bound: -0.6415345, upper bound: 0.6637412
IS_B2_A2_A1_B1_B1_A1, status: Status.VERIFIED, split count: 6, time: 3.29
Output dim: 0, lower bound: -0.6586239, upper bound: 0.6594773
IS_B2_A2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 0, lower bound: -0.6586239, upper bound: 0.6698469
IS_B2_A2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 0, lower bound: -0.6683065, upper bound: 0.6594773
IS_B2_A2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 0, lower bound: -0.6683065, upper bound: 0.6698469
IS_B2_A2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.29
Output dim: 0, lower bound: -0.6563801, upper bound: 0.6597357
IS_B2_A2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.29
Output dim: 0, lower bound: -0.6563801, upper bound: 0.6597357
IS_B2_A2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 0, lower bound: -0.6563801, upper bound: 0.6683078
IS_B2_A2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 0, lower bound: -0.6563801, upper bound: 0.6683079
IS_B2_A2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 0, lower bound: -0.5671587, upper bound: 0.6693510
IS_B2_A2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 0, lower bound: -0.5671587, upper bound: 0.6746561

## BFS IS instance: IS_B1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0017576, 0.9255085, 0.0266294, 1.0644715, -1.0627139, 0.8988791
1: -0.1074193, 1.1653817, -0.0699260, 1.3233504, -1.4307697, 1.2353077
2: -0.0565500, 1.0352211, -0.0273945, 1.1610806, -1.2176306, 1.0626156
3: -0.2699270, 1.0574689, -0.2324786, 1.1265240, -1.3964510, 1.2899475
4: -0.2435467, 0.9154017, -0.2112482, 0.9326113, -1.1761581, 1.1266499

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 29

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B1_A1_B1_B1

### Relational analysis result of IS_B1_B1_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6220397, upper bound: 0.6788159
time: 0.29 seconds

## Relational analysis of IS_B1_B1_A1_B1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_B1_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 47

## Relational analysis of IS_B1_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 47

## Relational analysis of IS_B1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_B1_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_B1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_B1_B1_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6381538, upper bound: 0.5934661
time: 0.34 seconds

## Relational analysis of IS_B1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_B1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6478744, upper bound: 0.6794671
time: 0.32 seconds

## BFS IS instance: IS_B1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0004439, 0.9531150, 0.0269289, 0.8937442, -0.8941882, 0.9261861
1: -0.1101191, 1.1983986, -0.0707574, 1.1311307, -1.2412498, 1.2691560
2: -0.0610516, 1.0641370, -0.0308993, 0.9843822, -1.0454338, 1.0950363
3: -0.2731256, 1.0821323, -0.2359240, 1.0170076, -1.2901332, 1.3180563
4: -0.2521458, 0.9308868, -0.2156389, 0.8432156, -1.0953614, 1.1465256

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_B1_B1_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6625438, upper bound: 0.6320923
time: 0.33 seconds

## Relational analysis of IS_B1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_B1_B1_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6444077, upper bound: 0.6473885
time: 0.33 seconds

## BFS IS instance: IS_B1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0015855, 0.9643674, 0.0266294, 1.0644715, -1.0628860, 0.9377379
1: -0.1072004, 1.2137957, -0.0699260, 1.3233504, -1.4305508, 1.2837217
2: -0.0610156, 1.0738001, -0.0273945, 1.1610806, -1.2220962, 1.1011946
3: -0.2690787, 1.0935032, -0.2324786, 1.1265240, -1.3956027, 1.3259819
4: -0.2552183, 0.9304721, -0.2112482, 0.9326113, -1.1878296, 1.1417203

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_B1

### Relational analysis result of IS_B1_B1_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5502470, upper bound: 0.6795654
time: 0.37 seconds

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 47

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 47

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 5

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_B1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5555441, upper bound: 0.6745061
time: 0.34 seconds

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_B1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5555441, upper bound: 0.6753137
time: 0.38 seconds

## BFS IS instance: IS_B1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0001364, 0.9854338, 0.0269289, 0.8937442, -0.8938806, 0.9585049
1: -0.1093853, 1.2390206, -0.0707574, 1.1311307, -1.2405159, 1.3097780
2: -0.0645683, 1.0958974, -0.0308993, 0.9843822, -1.0489504, 1.1267967
3: -0.2715011, 1.1123974, -0.2359240, 1.0170076, -1.2885087, 1.3483214
4: -0.2628224, 0.9425689, -0.2156389, 0.8432156, -1.1060380, 1.1582078

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B1_A2_B2_B1

### Relational analysis result of IS_B1_B1_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5929165, upper bound: 0.6747169
time: 0.34 seconds

## Relational analysis of IS_B1_B1_A1_B1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 47

## Relational analysis of IS_B1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 47

## Relational analysis of IS_B1_B1_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_B1_B1_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_B1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_B1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_B1_B1_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_B1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_B1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5963971, upper bound: 0.6745061
time: 0.31 seconds

## Relational analysis of IS_B1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_B1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5963971, upper bound: 0.6753137
time: 0.34 seconds

## BFS IS instance: IS_B1_B1_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: 0.0045321, 0.9268601, -0.0032020, 1.0339718, -1.0294397, 0.9300621
1: -0.1031680, 1.1689000, -0.1091664, 1.3081079, -1.4112759, 1.2780664
2: -0.0553467, 1.0304039, -0.0733573, 1.1349654, -1.1903121, 1.1037612
3: -0.2658038, 1.0580137, -0.2708402, 1.1731939, -1.4389977, 1.3288538
4: -0.2436506, 0.9083723, -0.2933947, 0.9566256, -1.2002761, 1.2017670

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_B1_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_B1_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_B1_A1_B2_A1_A1_B1

### Relational analysis result of IS_B1_B1_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6400006, upper bound: 0.6706810
time: 0.40 seconds

## Relational analysis of IS_B1_B1_A1_B2_A1_A1_B2

### Relational analysis result of IS_B1_B1_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6400006, upper bound: 0.6706810
time: 0.33 seconds

## BFS IS instance: IS_B1_B1_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: 0.0039101, 0.9696600, -0.0032020, 1.0339718, -1.0300617, 0.9728620
1: -0.1033678, 1.2211857, -0.1091664, 1.3081079, -1.4114757, 1.3303521
2: -0.0598848, 1.0737200, -0.0733573, 1.1349654, -1.1948502, 1.1470773
3: -0.2651293, 1.0970562, -0.2708402, 1.1731939, -1.4383233, 1.3678963
4: -0.2566521, 0.9262514, -0.2933947, 0.9566256, -1.2132777, 1.2196461

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_B1_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_B1_A1_B2_A1_A2_B1

### Relational analysis result of IS_B1_B1_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6400006, upper bound: 0.6714281
time: 0.34 seconds

## Relational analysis of IS_B1_B1_A1_B2_A1_A2_B2

### Relational analysis result of IS_B1_B1_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6400006, upper bound: 0.6714281
time: 0.38 seconds

## BFS IS instance: IS_B1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0426288, 1.8511992, -0.0014365, 1.0292757, -1.0719044, 1.8526356
1: -0.1642268, 2.2651000, -0.1070645, 1.3027349, -1.4669616, 2.3721645
2: -0.1343973, 2.0305896, -0.0715175, 1.1294107, -1.2638080, 2.1021070
3: -0.3265591, 1.8406374, -0.2690842, 1.1685212, -1.4950802, 2.1097217
4: -0.4800941, 1.4741251, -0.2915230, 0.9521359, -1.4322300, 1.7656481

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_B1_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6415344, upper bound: 0.6595624
time: 0.32 seconds

## Relational analysis of IS_B1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_B1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6415344, upper bound: 0.6637412
time: 0.33 seconds

## BFS IS instance: IS_B1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0426288, 1.8511992, -0.0357146, 1.8917470, -1.9343758, 1.8869138
1: -0.1642268, 2.2651000, -0.1529815, 2.3290219, -2.4932487, 2.4180815
2: -0.1343973, 2.0305896, -0.1319740, 2.0703101, -2.2047074, 2.1625636
3: -0.3265591, 1.8406374, -0.3186724, 1.8925552, -2.2191143, 2.1593099
4: -0.4800941, 1.4741251, -0.5038834, 1.4782946, -1.9583887, 1.9780085

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_B1_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6415345, upper bound: 0.6595624
time: 0.37 seconds

## Relational analysis of IS_B1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_B1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6415345, upper bound: 0.6637412
time: 0.34 seconds

## BFS IS instance: IS_B2_A2_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, 0.0111880, 0.5648615, -0.5954285, 0.6594702
1: -0.1440713, 0.8347254, -0.0944340, 0.7130072, -0.8570786, 0.9291594
2: -0.0742025, 0.8280591, -0.0312903, 0.7124612, -0.7866638, 0.8593494
3: -0.2901301, 0.8312570, -0.2500148, 0.7301772, -1.0203073, 1.0812719
4: -0.2451730, 0.8507983, -0.1787794, 0.7720095, -1.0171825, 1.0295777

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 29

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_A1_B1_B1_A2_B1

### Relational analysis result of IS_B2_A2_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6586238, upper bound: 0.6917820
time: 0.34 seconds

## Relational analysis of IS_B2_A2_A1_B1_B1_A2_B2

### Relational analysis result of IS_B2_A2_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6586238, upper bound: 0.6698469
time: 0.36 seconds

## BFS IS instance: IS_B2_A2_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, -0.0305669, 0.6706582, -0.6594702, 0.5954285
1: -0.0944340, 0.7130072, -0.1440713, 0.8347254, -0.9291594, 0.8570786
2: -0.0312903, 0.7124612, -0.0742025, 0.8280591, -0.8593494, 0.7866638
3: -0.2500148, 0.7301772, -0.2901301, 0.8312570, -1.0812719, 1.0203073
4: -0.1787794, 0.7720095, -0.2451730, 0.8507983, -1.0295777, 1.0171825

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_A1_B1_B2_A1_B1

### Relational analysis result of IS_B2_A2_A1_B1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6586239, upper bound: 0.6594773
time: 0.37 seconds

## Relational analysis of IS_B2_A2_A1_B1_B2_A1_B2

### Relational analysis result of IS_B2_A2_A1_B1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6586239, upper bound: 0.6594773
time: 0.39 seconds

## BFS IS instance: IS_B2_A2_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, -0.0305669, 0.6706582, -0.7012252, 0.7012252
1: -0.1440713, 0.8347254, -0.1440713, 0.8347254, -0.9787967, 0.9787967
2: -0.0742025, 0.8280591, -0.0742025, 0.8280591, -0.9022617, 0.9022617
3: -0.2901301, 0.8312570, -0.2901301, 0.8312570, -1.1213872, 1.1213872
4: -0.2451730, 0.8507983, -0.2451730, 0.8507983, -1.0959713, 1.0959713

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_A1_B1_B2_A2_B1

### Relational analysis result of IS_B2_A2_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6586238, upper bound: 0.6698469
time: 0.37 seconds

## Relational analysis of IS_B2_A2_A1_B1_B2_A2_B2

### Relational analysis result of IS_B2_A2_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6683065, upper bound: 0.6698469
time: 0.38 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, 0.0061345, 0.6507710, -0.6813380, 0.6645237
1: -0.1440713, 0.8347254, -0.0976210, 0.8106803, -0.9547516, 0.9323463
2: -0.0742025, 0.8280591, -0.0417109, 0.7880348, -0.8622373, 0.8697700
3: -0.2901301, 0.8312570, -0.2534082, 0.7965333, -1.0866635, 1.0846653
4: -0.2451730, 0.8507983, -0.2023642, 0.8128391, -1.0580120, 1.0531626

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 29

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_A1_B2_A2_B1_A1

### Relational analysis result of IS_B2_A2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6563801, upper bound: 0.6597357
time: 0.38 seconds

## Relational analysis of IS_B2_A2_A1_B2_A2_B1_A2

### Relational analysis result of IS_B2_A2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6563801, upper bound: 0.6683078
time: 0.43 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, -0.0329778, 0.7304660, -0.7610329, 0.7036361
1: -0.1440713, 0.8347254, -0.1447692, 0.9029138, -1.0469851, 0.9794946
2: -0.0742025, 0.8280591, -0.0813575, 0.8816222, -0.9558247, 0.9094166
3: -0.2901301, 0.8312570, -0.2928538, 0.8801655, -1.1702956, 1.1241109
4: -0.2451730, 0.8507983, -0.2609611, 0.8848863, -1.1300592, 1.1117594

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_A1_B2_A2_B2_A1

### Relational analysis result of IS_B2_A2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6659789, upper bound: 0.6597357
time: 0.38 seconds

## Relational analysis of IS_B2_A2_A1_B2_A2_B2_A2

### Relational analysis result of IS_B2_A2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6659790, upper bound: 0.6683078
time: 0.36 seconds

## BFS IS instance: IS_B2_A2_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0004525, 0.7221838, 0.0082216, 0.6401650, -0.6397125, 0.7139622
1: -0.1048932, 0.8939738, -0.0990725, 0.7995652, -0.9044584, 0.9930463
2: -0.0500441, 0.8598962, -0.0360949, 0.7876253, -0.8376693, 0.8959911
3: -0.2612641, 0.8577256, -0.2565451, 0.7902064, -1.0514705, 1.1142707
4: -0.2176540, 0.8560207, -0.1891084, 0.8149492, -1.0326033, 1.0451291

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A2_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_A2_A2_B1_B1_B1_A1

### Relational analysis result of IS_B2_A2_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5671587, upper bound: 0.6693510
time: 0.38 seconds

## Relational analysis of IS_B2_A2_A2_B1_B1_B1_A2

### Relational analysis result of IS_B2_A2_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5671587, upper bound: 0.6693510
time: 0.34 seconds

## BFS IS instance: IS_B2_A2_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0004525, 0.7221838, 0.0034440, 0.7121074, -0.7116549, 0.7187399
1: -0.1048932, 0.8939738, -0.1012318, 0.8823657, -0.9872589, 0.9952056
2: -0.0500441, 0.8598962, -0.0465870, 0.8481500, -0.8981941, 0.9064832
3: -0.2612641, 0.8577256, -0.2582860, 0.8474270, -1.1086910, 1.1160116
4: -0.2176540, 0.8560207, -0.2120385, 0.8468478, -1.0645018, 1.0680592

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_A2_A2_B1_B1_B2_A1

### Relational analysis result of IS_B2_A2_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5671587, upper bound: 0.6746561
time: 0.37 seconds

## Relational analysis of IS_B2_A2_A2_B1_B1_B2_A2

### Relational analysis result of IS_B2_A2_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5671587, upper bound: 0.6746561
time: 0.36 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.75 seconds
IS_B1_B1_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.75
Output dim: 0, lower bound: -0.6381538, upper bound: 0.5934661
IS_B1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 0, lower bound: -0.6478744, upper bound: 0.6794671
IS_B1_B1_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.75
Output dim: 0, lower bound: -0.6625438, upper bound: 0.6320923
IS_B1_B1_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.75
Output dim: 0, lower bound: -0.6444077, upper bound: 0.6473885
IS_B1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 0, lower bound: -0.5555441, upper bound: 0.6745061
IS_B1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 0, lower bound: -0.5555441, upper bound: 0.6753137
IS_B1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 0, lower bound: -0.5963971, upper bound: 0.6745061
IS_B1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 0, lower bound: -0.5963971, upper bound: 0.6753137
IS_B1_B1_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 0, lower bound: -0.6400006, upper bound: 0.6706810
IS_B1_B1_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 0, lower bound: -0.6400006, upper bound: 0.6706810
IS_B1_B1_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 0, lower bound: -0.6400006, upper bound: 0.6714281
IS_B1_B1_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 0, lower bound: -0.6400006, upper bound: 0.6714281
IS_B1_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.75
Output dim: 0, lower bound: -0.6415344, upper bound: 0.6595624
IS_B1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 0, lower bound: -0.6415344, upper bound: 0.6637412
IS_B1_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.75
Output dim: 0, lower bound: -0.6415345, upper bound: 0.6595624
IS_B1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 0, lower bound: -0.6415345, upper bound: 0.6637412
IS_B2_A2_A1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 0, lower bound: -0.6586238, upper bound: 0.6917820
IS_B2_A2_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 0, lower bound: -0.6586238, upper bound: 0.6698469
IS_B2_A2_A1_B1_B2_A1_B1, status: Status.VERIFIED, split count: 7, time: 3.75
Output dim: 0, lower bound: -0.6586239, upper bound: 0.6594773
IS_B2_A2_A1_B1_B2_A1_B2, status: Status.VERIFIED, split count: 7, time: 3.75
Output dim: 0, lower bound: -0.6586239, upper bound: 0.6594773
IS_B2_A2_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 0, lower bound: -0.6586238, upper bound: 0.6698469
IS_B2_A2_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 0, lower bound: -0.6683065, upper bound: 0.6698469
IS_B2_A2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.75
Output dim: 0, lower bound: -0.6563801, upper bound: 0.6597357
IS_B2_A2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 0, lower bound: -0.6563801, upper bound: 0.6683078
IS_B2_A2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 0, lower bound: -0.6659789, upper bound: 0.6597357
IS_B2_A2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 0, lower bound: -0.6659790, upper bound: 0.6683078
IS_B2_A2_A2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 0, lower bound: -0.5671587, upper bound: 0.6693510
IS_B2_A2_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 0, lower bound: -0.5671587, upper bound: 0.6693510
IS_B2_A2_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 0, lower bound: -0.5671587, upper bound: 0.6746561
IS_B2_A2_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.75
Output dim: 0, lower bound: -0.5671587, upper bound: 0.6746561

## BFS IS instance: IS_B1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0215945, 0.8004961, 0.0266294, 1.0644715, -1.0428770, 0.7738667
1: -0.0762348, 1.0046895, -0.0699260, 1.3233504, -1.3995852, 1.0746155
2: -0.0275140, 0.9088185, -0.0273945, 1.1610806, -1.1885946, 0.9362130
3: -0.2369068, 0.9284792, -0.2324786, 1.1265240, -1.3634307, 1.1609578
4: -0.1912906, 0.8270108, -0.2112482, 0.9326113, -1.1239020, 1.0382590

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 29

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_B1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_B1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6478744, upper bound: 0.6794671
time: 0.32 seconds

## Relational analysis of IS_B1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_B1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6478744, upper bound: 0.6745666
time: 0.37 seconds

## BFS IS instance: IS_B1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0056276, 1.1569200, 0.0266294, 1.0644715, -1.0588439, 1.1302905
1: -0.1015921, 1.4327645, -0.0699260, 1.3233504, -1.4249425, 1.5026906
2: -0.0547483, 1.2686064, -0.0273945, 1.1610806, -1.2158289, 1.2960010
3: -0.2631698, 1.2212660, -0.2324786, 1.1265240, -1.3896937, 1.4537446
4: -0.2540219, 1.0182528, -0.2112482, 0.9326113, -1.1866332, 1.2295010

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_B1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5502470, upper bound: 0.6783971
time: 0.31 seconds

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 47

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 47

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_B1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5555441, upper bound: 0.6793608
time: 0.34 seconds

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_B1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5555441, upper bound: 0.6745061
time: 0.34 seconds

## BFS IS instance: IS_B1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0068502, 0.9733622, 0.0266294, 1.0644715, -1.0576212, 0.9467328
1: -0.1012402, 1.2255073, -0.0699260, 1.3233504, -1.4245906, 1.2954333
2: -0.0573967, 1.0806024, -0.0273945, 1.1610806, -1.2184772, 1.1079969
3: -0.2653522, 1.1000264, -0.2324786, 1.1265240, -1.3918762, 1.3325050
4: -0.2570790, 0.9276546, -0.2112482, 0.9326113, -1.1896904, 1.1389028

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_B1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5502470, upper bound: 0.6795654
time: 0.35 seconds

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_B1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5555441, upper bound: 0.6801654
time: 0.34 seconds

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_B1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5555441, upper bound: 0.6753137
time: 0.32 seconds

## BFS IS instance: IS_B1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0056276, 1.1569200, 0.0269289, 0.8937442, -0.8881166, 1.1299911
1: -0.1015921, 1.4327645, -0.0707574, 1.1311307, -1.2327228, 1.5035219
2: -0.0547483, 1.2686064, -0.0308993, 0.9843822, -1.0391304, 1.2995057
3: -0.2631698, 1.2212660, -0.2359240, 1.0170076, -1.2801774, 1.4571900
4: -0.2540219, 1.0182528, -0.2156389, 0.8432156, -1.0972375, 1.2338917

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 29

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_B1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5502470, upper bound: 0.6735472
time: 0.32 seconds

## Relational analysis of IS_B1_B1_A1_B1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_B1_B1_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 47

## Relational analysis of IS_B1_B1_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 47

## Relational analysis of IS_B1_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_B1_B1_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_B1_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 5

## Relational analysis of IS_B1_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_B1_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_B1_B1_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_B1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_B1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5555441, upper bound: 0.6745061
time: 0.34 seconds

## Relational analysis of IS_B1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_B1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5555441, upper bound: 0.6745061
time: 0.35 seconds

## BFS IS instance: IS_B1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0068502, 0.9733622, 0.0269289, 0.8937442, -0.8868940, 0.9464333
1: -0.1012402, 1.2255073, -0.0707574, 1.1311307, -1.2323709, 1.2962646
2: -0.0573967, 1.0806024, -0.0308993, 0.9843822, -1.0417788, 1.1115017
3: -0.2653522, 1.1000264, -0.2359240, 1.0170076, -1.2823598, 1.3359504
4: -0.2570790, 0.9276546, -0.2156389, 0.8432156, -1.1002946, 1.1432935

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_B1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5502470, upper bound: 0.6747169
time: 0.35 seconds

## Relational analysis of IS_B1_B1_A1_B1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 47

## Relational analysis of IS_B1_B1_A1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_B1_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 47

## Relational analysis of IS_B1_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_B1_B1_A1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_B1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_B1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5555441, upper bound: 0.6753137
time: 0.33 seconds

## Relational analysis of IS_B1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_B1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5963971, upper bound: 0.6753137
time: 0.34 seconds

## BFS IS instance: IS_B1_B1_A1_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0045321, 0.9268601, -0.0014365, 1.0292757, -1.0247436, 0.9282966
1: -0.1031680, 1.1689000, -0.1070645, 1.3027349, -1.4059029, 1.2759645
2: -0.0553467, 1.0304039, -0.0715175, 1.1294107, -1.1847575, 1.1019213
3: -0.2658038, 1.0580137, -0.2690842, 1.1685212, -1.4343250, 1.3270979
4: -0.2436506, 0.9083723, -0.2915230, 0.9521359, -1.1957865, 1.1998954

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_B1_A1_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_B1_A1_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_B1_A1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 5

Time for candidate selection: 2.18 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B2_A1_A1_B1_B1

### Relational analysis result of IS_B1_B1_A1_B2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6324162, upper bound: 0.6700298
time: 0.35 seconds

## Relational analysis of IS_B1_B1_A1_B2_A1_A1_B1_B2

### Relational analysis result of IS_B1_B1_A1_B2_A1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6507773, upper bound: 0.6626328
time: 0.36 seconds

## BFS IS instance: IS_B1_B1_A1_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0045321, 0.9268601, -0.0357146, 1.8917470, -1.8872149, 0.9625747
1: -0.1031680, 1.1689000, -0.1529815, 2.3290219, -2.4321899, 1.3218815
2: -0.0553467, 1.0304039, -0.1319740, 2.0703101, -2.1256568, 1.1623778
3: -0.2658038, 1.0580137, -0.3186724, 1.8925552, -2.1583591, 1.3766861
4: -0.2436506, 0.9083723, -0.5038834, 1.4782946, -1.7219452, 1.4122558

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_B1_A1_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_B1_A1_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_B1_A1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 29

Time for candidate selection: 2.17 seconds

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_B1_B1_A1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B2_A1_A1_B2_B1

### Relational analysis result of IS_B1_B1_A1_B2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6324162, upper bound: 0.6700298
time: 0.32 seconds

## Relational analysis of IS_B1_B1_A1_B2_A1_A1_B2_B2

### Relational analysis result of IS_B1_B1_A1_B2_A1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6507773, upper bound: 0.6627502
time: 0.37 seconds

## BFS IS instance: IS_B1_B1_A1_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0039101, 0.9696600, -0.0014365, 1.0292757, -1.0253656, 0.9710965
1: -0.1033678, 1.2211857, -0.1070645, 1.3027349, -1.4061027, 1.3282502
2: -0.0598848, 1.0737200, -0.0715175, 1.1294107, -1.1892955, 1.1452374
3: -0.2651293, 1.0970562, -0.2690842, 1.1685212, -1.4336505, 1.3661404
4: -0.2566521, 0.9262514, -0.2915230, 0.9521359, -1.2087880, 1.2177744

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_B1_A1_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_B1_A1_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_B1_A1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 5

Time for candidate selection: 2.18 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B2_A1_A2_B1_B1

### Relational analysis result of IS_B1_B1_A1_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6202511, upper bound: 0.6708313
time: 0.34 seconds

## Relational analysis of IS_B1_B1_A1_B2_A1_A2_B1_B2

### Relational analysis result of IS_B1_B1_A1_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6386122, upper bound: 0.6634343
time: 0.36 seconds

## BFS IS instance: IS_B1_B1_A1_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0039101, 0.9696600, -0.0357146, 1.8917470, -1.8878369, 1.0053747
1: -0.1033678, 1.2211857, -0.1529815, 2.3290219, -2.4323897, 1.3741672
2: -0.0598848, 1.0737200, -0.1319740, 2.0703101, -2.1301949, 1.2056940
3: -0.2651293, 1.0970562, -0.3186724, 1.8925552, -2.1576846, 1.4157286
4: -0.2566521, 0.9262514, -0.5038834, 1.4782946, -1.7349467, 1.4301348

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_B1_A1_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_B1_A1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_B1_A1_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 29

Time for candidate selection: 2.15 seconds

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_B1_B1_A1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B2_A1_A2_B2_B1

### Relational analysis result of IS_B1_B1_A1_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6202511, upper bound: 0.6708313
time: 0.36 seconds

## Relational analysis of IS_B1_B1_A1_B2_A1_A2_B2_B2

### Relational analysis result of IS_B1_B1_A1_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6386122, upper bound: 0.6635518
time: 0.36 seconds

## BFS IS instance: IS_B1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0362997, 1.8639178, -0.0014365, 1.0292757, -1.0655754, 1.8653543
1: -0.1589222, 2.2834001, -0.1070645, 1.3027349, -1.4616570, 2.3904645
2: -0.1298122, 2.0459809, -0.0715175, 1.1294107, -1.2592230, 2.1174984
3: -0.3246737, 1.8475180, -0.2690842, 1.1685212, -1.4931948, 2.1166022
4: -0.4830998, 1.4824998, -0.2915230, 0.9521359, -1.4352357, 1.7740228

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_B1_A1_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_B1_A1_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 29

Time for candidate selection: 2.22 seconds

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_B1_B1_A1_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_B1_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6363368, upper bound: 0.6771435
time: 0.34 seconds

## Relational analysis of IS_B1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_B1_B1_A1_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6520016, upper bound: 0.6553096
time: 0.38 seconds

## BFS IS instance: IS_B1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0362997, 1.8639178, -0.0357146, 1.8917470, -1.9280467, 1.8996325
1: -0.1589222, 2.2834001, -0.1529815, 2.3290219, -2.4879441, 2.4363816
2: -0.1298122, 2.0459809, -0.1319740, 2.0703101, -2.2001224, 2.1779549
3: -0.3246737, 1.8475180, -0.3186724, 1.8925552, -2.2172289, 2.1661904
4: -0.4830998, 1.4824998, -0.5038834, 1.4782946, -1.9613944, 1.9863832

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_B1_A1_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_B1_A1_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5

Time for candidate selection: 2.21 seconds

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_B1_B1_A1_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_B1_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_B1_B1_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6217849, upper bound: 0.6627066
time: 0.36 seconds

## Relational analysis of IS_B1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_B1_B1_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6401460, upper bound: 0.6553096
time: 0.34 seconds

## BFS IS instance: IS_B2_A2_A1_B1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, 0.0111880, 0.5648615, -0.5954285, 0.6594702
1: -0.1440713, 0.8347254, -0.0944340, 0.7130072, -0.8570786, 0.9291594
2: -0.0742025, 0.8280591, -0.0312903, 0.7124612, -0.7866638, 0.8593494
3: -0.2901301, 0.8312570, -0.2500148, 0.7301772, -1.0203073, 1.0812719
4: -0.2451730, 0.8507983, -0.1787794, 0.7720095, -1.0171825, 1.0295777

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_A1_B1_B1_A2_B1_A1

### Relational analysis result of IS_B2_A2_A1_B1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6586239, upper bound: 0.6594773
time: 0.39 seconds

## Relational analysis of IS_B2_A2_A1_B1_B1_A2_B1_A2

### Relational analysis result of IS_B2_A2_A1_B1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6586238, upper bound: 0.6698469
time: 0.38 seconds

## BFS IS instance: IS_B2_A2_A1_B1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, -0.0305669, 0.6706582, -0.7012252, 0.7012252
1: -0.1440713, 0.8347254, -0.1440713, 0.8347254, -0.9787967, 0.9787967
2: -0.0742025, 0.8280591, -0.0742025, 0.8280591, -0.9022617, 0.9022617
3: -0.2901301, 0.8312570, -0.2901301, 0.8312570, -1.1213872, 1.1213872
4: -0.2451730, 0.8507983, -0.2451730, 0.8507983, -1.0959713, 1.0959713

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_A1_B1_B1_A2_B2_A1

### Relational analysis result of IS_B2_A2_A1_B1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6586239, upper bound: 0.6594773
time: 0.39 seconds

## Relational analysis of IS_B2_A2_A1_B1_B1_A2_B2_A2

### Relational analysis result of IS_B2_A2_A1_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6586238, upper bound: 0.6698469
time: 0.38 seconds

## BFS IS instance: IS_B2_A2_A1_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, 0.0111880, 0.5648615, -0.5954285, 0.6594702
1: -0.1440713, 0.8347254, -0.0944340, 0.7130072, -0.8570786, 0.9291594
2: -0.0742025, 0.8280591, -0.0312903, 0.7124612, -0.7866638, 0.8593494
3: -0.2901301, 0.8312570, -0.2500148, 0.7301772, -1.0203073, 1.0812719
4: -0.2451730, 0.8507983, -0.1787794, 0.7720095, -1.0171825, 1.0295777

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_A1_B1_B2_A2_B1_A1

### Relational analysis result of IS_B2_A2_A1_B1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6586239, upper bound: 0.6594773
time: 0.40 seconds

## Relational analysis of IS_B2_A2_A1_B1_B2_A2_B1_A2

### Relational analysis result of IS_B2_A2_A1_B1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6586238, upper bound: 0.6698469
time: 0.39 seconds

## BFS IS instance: IS_B2_A2_A1_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, -0.0305669, 0.6706582, -0.7012252, 0.7012252
1: -0.1440713, 0.8347254, -0.1440713, 0.8347254, -0.9787967, 0.9787967
2: -0.0742025, 0.8280591, -0.0742025, 0.8280591, -0.9022617, 0.9022617
3: -0.2901301, 0.8312570, -0.2901301, 0.8312570, -1.1213872, 1.1213872
4: -0.2451730, 0.8507983, -0.2451730, 0.8507983, -1.0959713, 1.0959713

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_A1_B1_B2_A2_B2_A1

### Relational analysis result of IS_B2_A2_A1_B1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6683067, upper bound: 0.6594773
time: 0.39 seconds

## Relational analysis of IS_B2_A2_A1_B1_B2_A2_B2_A2

### Relational analysis result of IS_B2_A2_A1_B1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6683067, upper bound: 0.6698469
time: 0.37 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, 0.0061345, 0.6507710, -0.6813380, 0.6645237
1: -0.1440713, 0.8347254, -0.0976210, 0.8106803, -0.9547516, 0.9323463
2: -0.0742025, 0.8280591, -0.0417109, 0.7880348, -0.8622373, 0.8697700
3: -0.2901301, 0.8312570, -0.2534082, 0.7965333, -1.0866635, 1.0846653
4: -0.2451730, 0.8507983, -0.2023642, 0.8128391, -1.0580120, 1.0531626

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_B2_A2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6563801, upper bound: 0.6910580
time: 0.39 seconds

## Relational analysis of IS_B2_A2_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_B2_A2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6563801, upper bound: 0.6683078
time: 0.42 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0111880, 0.5648615, -0.0329778, 0.7304660, -0.7192780, 0.5978394
1: -0.0944340, 0.7130072, -0.1447692, 0.9029138, -0.9973478, 0.8577764
2: -0.0312903, 0.7124612, -0.0813575, 0.8816222, -0.9129125, 0.7938187
3: -0.2500148, 0.7301772, -0.2928538, 0.8801655, -1.1301802, 1.0230310
4: -0.1787794, 0.7720095, -0.2609611, 0.8848863, -1.0636656, 1.0329705

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_B2_A2_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6563801, upper bound: 0.6597357
time: 0.37 seconds

## Relational analysis of IS_B2_A2_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_B2_A2_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6563801, upper bound: 0.6597357
time: 0.40 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0305669, 0.6706582, -0.0329778, 0.7304660, -0.7610329, 0.7036361
1: -0.1440713, 0.8347254, -0.1447692, 0.9029138, -1.0469851, 0.9794946
2: -0.0742025, 0.8280591, -0.0813575, 0.8816222, -0.9558247, 0.9094166
3: -0.2901301, 0.8312570, -0.2928538, 0.8801655, -1.1702956, 1.1241109
4: -0.2451730, 0.8507983, -0.2609611, 0.8848863, -1.1300592, 1.1117594

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_B2_A2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6563801, upper bound: 0.6683078
time: 0.37 seconds

## Relational analysis of IS_B2_A2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_B2_A2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6659790, upper bound: 0.6683078
time: 0.41 seconds

## BFS IS instance: IS_B2_A2_A2_B1_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0021317, 0.7169888, 0.0082216, 0.6401650, -0.6380333, 0.7087672
1: -0.1028535, 0.8879859, -0.0990725, 0.7995652, -0.9024187, 0.9870584
2: -0.0482047, 0.8538951, -0.0360949, 0.7876253, -0.8358299, 0.8899900
3: -0.2595775, 0.8524666, -0.2565451, 0.7902064, -1.0497839, 1.1090117
4: -0.2148404, 0.8513455, -0.1891084, 0.8149492, -1.0297897, 1.0404539

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A2_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A2_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A2_A2_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A2_A2_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_A2_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 5

Time for candidate selection: 2.38 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_A2_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_A2_B1_B1_B1_A1_B1

### Relational analysis result of IS_B2_A2_A2_B1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5636006, upper bound: 0.6682976
time: 0.40 seconds

## Relational analysis of IS_B2_A2_A2_B1_B1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_B2_A2_A2_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_B2_A2_A2_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_B2_A2_A2_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 47

## Relational analysis of IS_B2_A2_A2_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_B2_A2_A2_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 47

## Relational analysis of IS_B2_A2_A2_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_B2_A2_A2_B1_B1_B1_A1_A1

### Relational analysis result of IS_B2_A2_A2_B1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5662680, upper bound: 0.6645631
time: 0.40 seconds

## Relational analysis of IS_B2_A2_A2_B1_B1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_B2_A2_A2_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_B2_A2_A2_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 5

## Relational analysis of IS_B2_A2_A2_B1_B1_B1_A1_B1

### Relational analysis result of IS_B2_A2_A2_B1_B1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5649692, upper bound: 0.6554637
time: 0.36 seconds

## Relational analysis of IS_B2_A2_A2_B1_B1_B1_A1_B2

### Relational analysis result of IS_B2_A2_A2_B1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5671587, upper bound: 0.6690073
time: 0.36 seconds

## BFS IS instance: IS_B2_A2_A2_B1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0419519, 1.5706639, 0.0082216, 0.6401650, -0.6821169, 1.5624423
1: -0.1653557, 1.8964453, -0.0990725, 0.7995652, -0.9649209, 1.9955177
2: -0.1181295, 1.7439981, -0.0360949, 0.7876253, -0.9057547, 1.7800930
3: -0.3246105, 1.5603621, -0.2565451, 0.7902064, -1.1148169, 1.8169072
4: -0.3863105, 1.3762889, -0.1891084, 0.8149492, -1.2012596, 1.5653973

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A2_B1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A2_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A2_A2_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A2_A2_B1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_A2_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 29

Time for candidate selection: 2.35 seconds

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_B2_A2_A2_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_A2_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_B2_A2_A2_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_A2_B1_B1_B1_A2_B1

### Relational analysis result of IS_B2_A2_A2_B1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5636006, upper bound: 0.6682976
time: 0.36 seconds

## Relational analysis of IS_B2_A2_A2_B1_B1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_B2_A2_A2_B1_B1_B1_A2_A1

### Relational analysis result of IS_B2_A2_A2_B1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5662680, upper bound: 0.6645631
time: 0.42 seconds

## Relational analysis of IS_B2_A2_A2_B1_B1_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_B2_A2_A2_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 47

## Relational analysis of IS_B2_A2_A2_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 47

## Relational analysis of IS_B2_A2_A2_B1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_B2_A2_A2_B1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 5

## Relational analysis of IS_B2_A2_A2_B1_B1_B1_A2_B1

### Relational analysis result of IS_B2_A2_A2_B1_B1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5649692, upper bound: 0.6554637
time: 0.36 seconds

## Relational analysis of IS_B2_A2_A2_B1_B1_B1_A2_B2

### Relational analysis result of IS_B2_A2_A2_B1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5671587, upper bound: 0.6690073
time: 0.39 seconds

## BFS IS instance: IS_B2_A2_A2_B1_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0021317, 0.7169888, 0.0034440, 0.7121074, -0.7099757, 0.7135448
1: -0.1028535, 0.8879859, -0.1012318, 0.8823657, -0.9852192, 0.9892178
2: -0.0482047, 0.8538951, -0.0465870, 0.8481500, -0.8963547, 0.9004821
3: -0.2595775, 0.8524666, -0.2582860, 0.8474270, -1.1070044, 1.1107526
4: -0.2148404, 0.8513455, -0.2120385, 0.8468478, -1.0616882, 1.0633841

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A2_B1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_A2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A2_A2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A2_A2_B1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 5

Time for candidate selection: 2.38 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_A2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_A2_B1_B1_B2_A1_B1

### Relational analysis result of IS_B2_A2_A2_B1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6345027, upper bound: 0.6736101
time: 0.39 seconds

## Relational analysis of IS_B2_A2_A2_B1_B1_B2_A1_B2

### Relational analysis result of IS_B2_A2_A2_B1_B1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6501676, upper bound: 0.6515093
time: 0.41 seconds

## BFS IS instance: IS_B2_A2_A2_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0419519, 1.5706639, 0.0034440, 0.7121074, -0.7540593, 1.5672200
1: -0.1653557, 1.8964453, -0.1012318, 0.8823657, -1.0477214, 1.9976771
2: -0.1181295, 1.7439981, -0.0465870, 0.8481500, -0.9662795, 1.7905850
3: -0.3246105, 1.5603621, -0.2582860, 0.8474270, -1.1720374, 1.8186481
4: -0.3863105, 1.3762889, -0.2120385, 0.8468478, -1.2331582, 1.5883274

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A2_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A2_A2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A2_A2_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_A2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 29

Time for candidate selection: 2.39 seconds

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_B2_A2_A2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_A2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B2_A2_A2_B1_B1_B2_A2_B1

### Relational analysis result of IS_B2_A2_A2_B1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6345027, upper bound: 0.6736101
time: 0.36 seconds

## Relational analysis of IS_B2_A2_A2_B1_B1_B2_A2_B2

### Relational analysis result of IS_B2_A2_A2_B1_B1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6501676, upper bound: 0.6515093
time: 0.37 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 5.93 seconds
IS_B1_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.93
Output dim: 0, lower bound: -0.6478744, upper bound: 0.6794671
IS_B1_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.93
Output dim: 0, lower bound: -0.6478744, upper bound: 0.6745666
IS_B1_B1_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.93
Output dim: 0, lower bound: -0.5555441, upper bound: 0.6793608
IS_B1_B1_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.93
Output dim: 0, lower bound: -0.5555441, upper bound: 0.6745061
IS_B1_B1_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.93
Output dim: 0, lower bound: -0.5555441, upper bound: 0.6801654
IS_B1_B1_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.93
Output dim: 0, lower bound: -0.5555441, upper bound: 0.6753137
IS_B1_B1_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.93
Output dim: 0, lower bound: -0.5555441, upper bound: 0.6745061
IS_B1_B1_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.93
Output dim: 0, lower bound: -0.5555441, upper bound: 0.6745061
IS_B1_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.93
Output dim: 0, lower bound: -0.5555441, upper bound: 0.6753137
IS_B1_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.93
Output dim: 0, lower bound: -0.5963971, upper bound: 0.6753137
IS_B1_B1_A1_B2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 5.93
Output dim: 0, lower bound: -0.6324162, upper bound: 0.6700298
IS_B1_B1_A1_B2_A1_A1_B1_B2, status: Status.VERIFIED, split count: 8, time: 5.93
Output dim: 0, lower bound: -0.6507773, upper bound: 0.6626328
IS_B1_B1_A1_B2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 5.93
Output dim: 0, lower bound: -0.6324162, upper bound: 0.6700298
IS_B1_B1_A1_B2_A1_A1_B2_B2, status: Status.VERIFIED, split count: 8, time: 5.93
Output dim: 0, lower bound: -0.6507773, upper bound: 0.6627502
IS_B1_B1_A1_B2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 5.93
Output dim: 0, lower bound: -0.6202511, upper bound: 0.6708313
IS_B1_B1_A1_B2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 5.93
Output dim: 0, lower bound: -0.6386122, upper bound: 0.6634343
IS_B1_B1_A1_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 5.93
Output dim: 0, lower bound: -0.6202511, upper bound: 0.6708313
IS_B1_B1_A1_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 5.93
Output dim: 0, lower bound: -0.6386122, upper bound: 0.6635518
IS_B1_B1_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.93
Output dim: 0, lower bound: -0.6363368, upper bound: 0.6771435
IS_B1_B1_A1_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 5.93
Output dim: 0, lower bound: -0.6520016, upper bound: 0.6553096
IS_B1_B1_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 5.93
Output dim: 0, lower bound: -0.6217849, upper bound: 0.6627066
IS_B1_B1_A1_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 5.93
Output dim: 0, lower bound: -0.6401460, upper bound: 0.6553096
IS_B2_A2_A1_B1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 8, time: 5.93
Output dim: 0, lower bound: -0.6586239, upper bound: 0.6594773
IS_B2_A2_A1_B1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 5.93
Output dim: 0, lower bound: -0.6586238, upper bound: 0.6698469
IS_B2_A2_A1_B1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 8, time: 5.93
Output dim: 0, lower bound: -0.6586239, upper bound: 0.6594773
IS_B2_A2_A1_B1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 5.93
Output dim: 0, lower bound: -0.6586238, upper bound: 0.6698469
IS_B2_A2_A1_B1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 8, time: 5.93
Output dim: 0, lower bound: -0.6586239, upper bound: 0.6594773
IS_B2_A2_A1_B1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 5.93
Output dim: 0, lower bound: -0.6586238, upper bound: 0.6698469
IS_B2_A2_A1_B1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 5.93
Output dim: 0, lower bound: -0.6683067, upper bound: 0.6594773
IS_B2_A2_A1_B1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 5.93
Output dim: 0, lower bound: -0.6683067, upper bound: 0.6698469
IS_B2_A2_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.93
Output dim: 0, lower bound: -0.6563801, upper bound: 0.6910580
IS_B2_A2_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.93
Output dim: 0, lower bound: -0.6563801, upper bound: 0.6683078
IS_B2_A2_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 5.93
Output dim: 0, lower bound: -0.6563801, upper bound: 0.6597357
IS_B2_A2_A1_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 5.93
Output dim: 0, lower bound: -0.6563801, upper bound: 0.6597357
IS_B2_A2_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.93
Output dim: 0, lower bound: -0.6563801, upper bound: 0.6683078
IS_B2_A2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.93
Output dim: 0, lower bound: -0.6659790, upper bound: 0.6683078
IS_B2_A2_A2_B1_B1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 5.93
Output dim: 0, lower bound: -0.5649692, upper bound: 0.6554637
IS_B2_A2_A2_B1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.93
Output dim: 0, lower bound: -0.5671587, upper bound: 0.6690073
IS_B2_A2_A2_B1_B1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 5.93
Output dim: 0, lower bound: -0.5649692, upper bound: 0.6554637
IS_B2_A2_A2_B1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.93
Output dim: 0, lower bound: -0.5671587, upper bound: 0.6690073
IS_B2_A2_A2_B1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.93
Output dim: 0, lower bound: -0.6345027, upper bound: 0.6736101
IS_B2_A2_A2_B1_B1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 5.93
Output dim: 0, lower bound: -0.6501676, upper bound: 0.6515093
IS_B2_A2_A2_B1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.93
Output dim: 0, lower bound: -0.6345027, upper bound: 0.6736101
IS_B2_A2_A2_B1_B1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 5.93
Output dim: 0, lower bound: -0.6501676, upper bound: 0.6515093

## BFS IS instance: IS_B1_B1_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0241051, 0.7727232, 0.0266294, 1.0644715, -1.0403664, 0.7460938
1: -0.0729418, 0.9717798, -0.0699260, 1.3233504, -1.3962922, 1.0417058
2: -0.0226438, 0.8791733, -0.0273945, 1.1610806, -1.1837244, 0.9065678
3: -0.2333536, 0.9036698, -0.2324786, 1.1265240, -1.3598776, 1.1361485
4: -0.1821667, 0.8096986, -0.2112482, 0.9326113, -1.1147780, 1.0209467

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B1_A1_B1_A2_B1_B1

### Relational analysis result of IS_B1_B1_A1_B1_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6220397, upper bound: 0.6788159
time: 0.36 seconds

## Relational analysis of IS_B1_B1_A1_B1_A1_B1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_B1_B1_A1_B1_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 47

## Relational analysis of IS_B1_B1_A1_B1_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 47

## Relational analysis of IS_B1_B1_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_B1_B1_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_B1_B1_A1_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6381538, upper bound: 0.5934661
time: 0.33 seconds

## Relational analysis of IS_B1_B1_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_B1_B1_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6478744, upper bound: 0.6794671
time: 0.34 seconds

## BFS IS instance: IS_B1_B1_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0215945, 0.8004961, 0.0269289, 0.8937442, -0.8721497, 0.7735672
1: -0.0762348, 1.0046895, -0.0707574, 1.1311307, -1.2073655, 1.0754468
2: -0.0275140, 0.9088185, -0.0308993, 0.9843822, -1.0118961, 0.9397178
3: -0.2369068, 0.9284792, -0.2359240, 1.0170076, -1.2539144, 1.1644032
4: -0.1912906, 0.8270108, -0.2156389, 0.8432156, -1.0345062, 1.0426497

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B1_A1_B1_A2_B2_B1

### Relational analysis result of IS_B1_B1_A1_B1_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6220397, upper bound: 0.6739154
time: 0.37 seconds

## Relational analysis of IS_B1_B1_A1_B1_A1_B1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_B1_B1_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 47

## Relational analysis of IS_B1_B1_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 47

## Relational analysis of IS_B1_B1_A1_B1_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_B1_B1_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_B1_B1_A1_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.6381538, upper bound: 0.5885656
time: 0.35 seconds

## Relational analysis of IS_B1_B1_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_B1_B1_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.6478744, upper bound: 0.6745666
time: 0.35 seconds

## BFS IS instance: IS_B1_B1_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0056276, 1.1569200, 0.0266294, 1.0644715, -1.0588439, 1.1302905
1: -0.1015921, 1.4327645, -0.0699260, 1.3233504, -1.4249425, 1.5026906
2: -0.0547483, 1.2686064, -0.0273945, 1.1610806, -1.2158289, 1.2960010
3: -0.2631698, 1.2212660, -0.2324786, 1.1265240, -1.3896937, 1.4537446
4: -0.2540219, 1.0182528, -0.2112482, 0.9326113, -1.1866332, 1.2295010

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A1_B1_B1

### Relational analysis result of IS_B1_B1_A1_B1_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5595860, upper bound: 0.6735472
time: 0.33 seconds

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 47

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 47

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_B1_B1_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5645868, upper bound: 0.6745061
time: 0.35 seconds

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_B1_B1_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5555441, upper bound: 0.6745061
time: 0.34 seconds

## BFS IS instance: IS_B1_B1_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0056276, 1.1569200, 0.0269289, 0.8937442, -0.8881166, 1.1299911
1: -0.1015921, 1.4327645, -0.0707574, 1.1311307, -1.2327228, 1.5035219
2: -0.0547483, 1.2686064, -0.0308993, 0.9843822, -1.0391304, 1.2995057
3: -0.2631698, 1.2212660, -0.2359240, 1.0170076, -1.2801774, 1.4571900
4: -0.2540219, 1.0182528, -0.2156389, 0.8432156, -1.0972375, 1.2338917

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A1_B2_B1

### Relational analysis result of IS_B1_B1_A1_B1_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5595860, upper bound: 0.6735472
time: 0.36 seconds

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_B1_B1_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5645868, upper bound: 0.6745061
time: 0.36 seconds

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_B1_B1_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5555441, upper bound: 0.6745061
time: 0.35 seconds

## BFS IS instance: IS_B1_B1_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0068502, 0.9733622, 0.0266294, 1.0644715, -1.0576212, 0.9467328
1: -0.1012402, 1.2255073, -0.0699260, 1.3233504, -1.4245906, 1.2954333
2: -0.0573967, 1.0806024, -0.0273945, 1.1610806, -1.2184772, 1.1079969
3: -0.2653522, 1.1000264, -0.2324786, 1.1265240, -1.3918762, 1.3325050
4: -0.2570790, 0.9276546, -0.2112482, 0.9326113, -1.1896904, 1.1389028

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A2_B1_B1

### Relational analysis result of IS_B1_B1_A1_B1_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5502470, upper bound: 0.6747169
time: 0.35 seconds

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 47

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 47

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 5

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 5

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_B1_B1_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5555441, upper bound: 0.6745061
time: 0.35 seconds

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_B1_B1_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5555441, upper bound: 0.6753137
time: 0.35 seconds

## BFS IS instance: IS_B1_B1_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0068502, 0.9733622, 0.0269289, 0.8937442, -0.8868940, 0.9464333
1: -0.1012402, 1.2255073, -0.0707574, 1.1311307, -1.2323709, 1.2962646
2: -0.0573967, 1.0806024, -0.0308993, 0.9843822, -1.0417788, 1.1115017
3: -0.2653522, 1.1000264, -0.2359240, 1.0170076, -1.2823598, 1.3359504
4: -0.2570790, 0.9276546, -0.2156389, 0.8432156, -1.1002946, 1.1432935

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A2_B2_B1

### Relational analysis result of IS_B1_B1_A1_B1_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5502470, upper bound: 0.6747169
time: 0.34 seconds

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 47

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 47

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_B1_B1_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5555441, upper bound: 0.6745061
time: 0.37 seconds

## Relational analysis of IS_B1_B1_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_B1_B1_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5555441, upper bound: 0.6753137
time: 0.36 seconds

## BFS IS instance: IS_B1_B1_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0056276, 1.1569200, 0.0266294, 1.0644715, -1.0588439, 1.1302905
1: -0.1015921, 1.4327645, -0.0699260, 1.3233504, -1.4249425, 1.5026906
2: -0.0547483, 1.2686064, -0.0273945, 1.1610806, -1.2158289, 1.2960010
3: -0.2631698, 1.2212660, -0.2324786, 1.1265240, -1.3896937, 1.4537446
4: -0.2540219, 1.0182528, -0.2112482, 0.9326113, -1.1866332, 1.2295010

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 47
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_B1_B1_A1_B1_A2_B2_A1_B1_B1

### Relational analysis result of IS_B1_B1_A1_B1_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5595860, upper bound: 0.6735472
time: 0.33 seconds

## Relational analysis of IS_B1_B1_A1_B1_A2_B2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_B1_B1_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_B1_B1_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_B1_B1_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_B1_B1_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 47

## Relational analysis of IS_B1_B1_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 47

## Relational analysis of IS_B1_B1_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 2): status=Status.UNKNOWN, low=0.0017312, high=0.0242421, mid=0.0242421, abs_max=0.7819017171859741
rel_dist={0: [-0.6965375425073153, 0.6965375425073155]}

## Binary Search with IS_dual Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0017311790288658813
execution time: 1149.01 seconds
