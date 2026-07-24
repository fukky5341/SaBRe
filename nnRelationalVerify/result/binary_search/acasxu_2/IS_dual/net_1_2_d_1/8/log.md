## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_2.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 43.3827531155


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307)
1: (-8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275)
2: (-9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216)
3: (-14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580)
4: (-14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412)

## BASE Result
execution time: IAR + LP analysis = 2.01 + 1.50 = 3.51 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -43.6007569, upper bound: 43.6007569


# Binary Search by BASE starts (time budget: 1196.49 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.0625000


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=46.890357971191406
rel_dist={3: [-43.60075692157075, 43.60075692157075]}

## Binary search (step 1) starts
Candidate diff: 0.0312500


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0312500, mid=0.0312500, abs_max=46.890357971191406
rel_dist={3: [-43.60075692157075, 43.60075692157075]}

## Binary search (step 2) starts
Candidate diff: 0.0156250


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0156250, mid=0.0156250, abs_max=46.890357971191406
rel_dist={3: [-43.60075692157075, 43.60075692157075]}

## Binary search (step 3) starts
Candidate diff: 0.0078125


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0078125, mid=0.0078125, abs_max=46.890357971191406
rel_dist={3: [-43.60075692157075, 43.60075692157075]}

## Binary search (step 4) starts
Candidate diff: 0.0039062


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0039062, mid=0.0039062, abs_max=46.890357971191406
rel_dist={3: [-43.60068373509955, 43.60068373509955]}

## Binary search (step 5) starts
Candidate diff: 0.0019531


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0019531, mid=0.0019531, abs_max=46.890357971191406
rel_dist={3: [-43.6005733842848, 43.6005733842848]}

## Binary search (step 6) starts
Candidate diff: 0.0009766


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0009766, mid=0.0009766, abs_max=46.890357971191406
rel_dist={3: [-43.60049262479107, 43.60049262479107]}

## Binary search (step 7) starts
Candidate diff: 0.0004883


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0004883, mid=0.0004883, abs_max=46.890357971191406
rel_dist={3: [-43.600442096761945, 43.60044209676194]}

## Binary search (step 8) starts
Candidate diff: 0.0002441


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0002441, mid=0.0002441, abs_max=46.890357971191406
rel_dist={3: [-43.600415482278315, 43.60041548227832]}

## Binary search (step 9) starts
Candidate diff: 0.0001221


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0001221, mid=0.0001221, abs_max=46.890357971191406
rel_dist={3: [-43.60040197566741, 43.60040197566741]}

## Binary search (step 10) starts
Candidate diff: 0.0000610


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0000610, mid=0.0000610, abs_max=46.890357971191406
rel_dist={3: [-43.600395222369386, 43.600395222369386]}

## Binary search (step 11) starts
Candidate diff: 0.0000305


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0000305, mid=0.0000305, abs_max=46.890357971191406
rel_dist={3: [-43.600391845735146, 43.600391845735146]}

## Binary search (step 12) starts
Candidate diff: 0.0000153


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0000153, mid=0.0000153, abs_max=46.890357971191406
rel_dist={3: [-43.60039015744708, 43.60039015744708]}

## Binary search (step 13) starts
Candidate diff: 0.0000076


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000076, mid=0.0000076, abs_max=46.890357971191406
rel_dist={3: [-43.60038931335932, 43.60038931335933]}

## Binary search (step 14) starts
Candidate diff: 0.0000038


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000038, mid=0.0000038, abs_max=46.890357971191406
rel_dist={3: [-43.60038889142116, 43.60038889142115]}

## Binary search (step 15) starts
Candidate diff: 0.0000019


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000019, mid=0.0000019, abs_max=46.890357971191406
rel_dist={3: [-43.60038868063948, 43.60038868930823]}

## Binary search (step 16) starts
Candidate diff: 0.0000010


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000010, mid=0.0000010, abs_max=46.890357971191406
rel_dist={3: [-43.60038857723265, 43.60038857813507]}

## Binary Search Result
Binary search time: 63.63 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 1132.86 seconds

## Binary search (step 0) starts
Candidate diff: 0.0625000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5760261, upper bound: 43.5949019
time: 0.48 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
time: 0.87 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.55 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 1.55
Output dim: 3, lower bound: -43.5760261, upper bound: 43.5949019
IS_B2, status: Status.UNKNOWN, split count: 1, time: 1.55
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -7.6606002, 28.3044319, -6.4284191, 24.1388493, -31.7994499, 34.7328491
1: -8.9601765, 31.9744511, -7.4645872, 27.3031864, -36.2633629, 39.4390373
2: -9.4893942, 32.1866379, -7.9618182, 27.3991623, -36.8885536, 40.1484489
3: -14.0368652, 32.8534927, -11.7890568, 27.9809723, -42.0178375, 44.6425476
4: -14.5853920, 32.1437492, -12.4559155, 27.1424770, -41.7278671, 44.5996590

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
time: 0.48 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
time: 0.92 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -7.6574574, 28.2928791, -8.8551903, 33.0941162, -40.7515717, 37.1480713
1: -8.9563580, 31.9612103, -10.4291992, 37.5969582, -46.5533142, 42.3904037
2: -9.4855747, 32.1733856, -10.9747181, 37.6314697, -47.1170425, 43.1481018
3: -14.0309029, 32.8395958, -16.3464375, 38.8504906, -52.8813896, 49.1860275
4: -14.5792150, 32.1307449, -17.2999840, 37.0951767, -51.6743851, 49.4307289

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
time: 0.53 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
time: 0.50 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.13 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 3.13
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 3.13
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 3.13
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 3.13
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -6.4284191, 24.1388493, -6.4284191, 24.1388493, -30.5672684, 30.5672684
1: -7.4645872, 27.3031864, -7.4645872, 27.3031864, -34.7677727, 34.7677727
2: -7.9618182, 27.3991623, -7.9618182, 27.3991623, -35.3609810, 35.3609810
3: -11.7890568, 27.9809723, -11.7890568, 27.9809723, -39.7700233, 39.7700272
4: -12.4559155, 27.1424770, -12.4559155, 27.1424770, -39.5983925, 39.5983925

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B1_A1_B1

### Relational analysis result of IS_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5760261, upper bound: 43.5949019
time: 0.61 seconds

## Relational analysis of IS_B1_A1_B2

### Relational analysis result of IS_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5726843, upper bound: 43.5814188
time: 0.51 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -8.8551903, 33.0941162, -6.4284191, 24.1388493, -32.9940414, 39.5225296
1: -10.4291992, 37.5969582, -7.4645872, 27.3031864, -37.7323837, 45.0615425
2: -10.9747181, 37.6314697, -7.9618182, 27.3991623, -38.3738785, 45.5932884
3: -16.3464375, 38.8504906, -11.7890568, 27.9809723, -44.3274078, 50.6395416
4: -17.2999840, 37.0951767, -12.4559155, 27.1424770, -44.4424591, 49.5510902

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B1

### Relational analysis result of IS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5760261, upper bound: 43.5949019
time: 0.52 seconds

## Relational analysis of IS_B1_A2_B2

### Relational analysis result of IS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5726843, upper bound: 43.5814188
time: 0.55 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -6.4284191, 24.1388493, -8.8551903, 33.0941162, -39.5225296, 32.9940414
1: -7.4645872, 27.3031864, -10.4291992, 37.5969582, -45.0615425, 37.7323837
2: -7.9618182, 27.3991623, -10.9747181, 37.6314697, -45.5932884, 38.3738785
3: -11.7890568, 27.9809723, -16.3464375, 38.8504906, -50.6395416, 44.3274078
4: -12.4559155, 27.1424770, -17.2999840, 37.0951767, -49.5510902, 44.4424591

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_B1

### Relational analysis result of IS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5442300, upper bound: 43.5658637
time: 0.50 seconds

## Relational analysis of IS_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5409015, upper bound: 43.5409015
time: 0.48 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -8.8551903, 33.0941162, -8.8551903, 33.0941162, -41.9493065, 41.9493065
1: -10.4291992, 37.5969582, -10.4291992, 37.5969582, -48.0261536, 48.0261536
2: -10.9747181, 37.6314697, -10.9747181, 37.6314697, -48.6061859, 48.6061859
3: -16.3464375, 38.8504906, -16.3464375, 38.8504906, -55.1969299, 55.1969299
4: -17.2999840, 37.0951767, -17.2999840, 37.0951767, -54.3951607, 54.3951607

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_A2_A1

### Relational analysis result of IS_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5658637, upper bound: 43.5442300
time: 0.49 seconds

## Relational analysis of IS_B2_A2_A2

### Relational analysis result of IS_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5409015, upper bound: 43.5409015
time: 0.90 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.50 seconds
IS_B1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 3.50
Output dim: 3, lower bound: -43.5760261, upper bound: 43.5949019
IS_B1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 3.50
Output dim: 3, lower bound: -43.5726843, upper bound: 43.5814188
IS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 3.50
Output dim: 3, lower bound: -43.5760261, upper bound: 43.5949019
IS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 3.50
Output dim: 3, lower bound: -43.5726843, upper bound: 43.5814188
IS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 3.50
Output dim: 3, lower bound: -43.5442300, upper bound: 43.5658637
IS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 3.50
Output dim: 3, lower bound: -43.5409015, upper bound: 43.5409015
IS_B2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 3.50
Output dim: 3, lower bound: -43.5658637, upper bound: 43.5442300
IS_B2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 3.50
Output dim: 3, lower bound: -43.5409015, upper bound: 43.5409015

## BFS IS instance: IS_B1_A1_B1

### Backsubstitution after applying IS history:
0: -6.4284191, 24.1388493, -5.2551436, 20.1315994, -26.5600185, 29.3939915
1: -7.4645872, 27.3031864, -6.0467672, 22.8425674, -30.3071537, 33.3499527
2: -7.9618182, 27.3991623, -6.4991312, 22.7556610, -30.7174797, 33.8982925
3: -11.7890568, 27.9809723, -9.6570997, 23.3396664, -35.1287231, 37.6380692
4: -12.4559155, 27.1424770, -10.4297295, 22.2815781, -34.7374878, 37.5722046

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A1_B1_A1

### Relational analysis result of IS_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5839320, upper bound: 43.5839320
time: 0.52 seconds

## Relational analysis of IS_B1_A1_B1_A2

### Relational analysis result of IS_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5839320, upper bound: 43.5839320
time: 0.54 seconds

## BFS IS instance: IS_B1_A1_B2

### Backsubstitution after applying IS history:
0: -6.4284191, 24.1388493, -6.7437096, 25.4687443, -31.8971615, 30.8825588
1: -7.4645872, 27.3031864, -7.8409715, 28.8711681, -36.3357468, 35.1441574
2: -7.9618182, 27.3991623, -8.3597012, 28.9183235, -36.8801422, 35.7588577
3: -11.7890568, 27.9809723, -12.3940144, 29.6448441, -41.4338913, 40.3749847
4: -12.4559155, 27.1424770, -13.1956530, 28.4903889, -40.9463043, 40.3381310

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A1_B2_A1

### Relational analysis result of IS_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5839320, upper bound: 43.5839320
time: 0.50 seconds

## Relational analysis of IS_B1_A1_B2_A2

### Relational analysis result of IS_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5839320, upper bound: 43.5839320
time: 0.51 seconds

## BFS IS instance: IS_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.8551903, 33.0941162, -5.2551436, 20.1315994, -28.9867897, 38.3492584
1: -10.4291992, 37.5969582, -6.0467672, 22.8425674, -33.2717590, 43.6437263
2: -10.9747181, 37.6314697, -6.4991312, 22.7556610, -33.7303772, 44.1306000
3: -16.3464375, 38.8504906, -9.6570997, 23.3396664, -39.6861038, 48.5075874
4: -17.2999840, 37.0951767, -10.4297295, 22.2815781, -39.5815620, 47.5249062

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_B1_A1

### Relational analysis result of IS_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5726330, upper bound: 43.5725862
time: 0.49 seconds

## Relational analysis of IS_B1_A2_B1_A2

### Relational analysis result of IS_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5476707, upper bound: 43.5692578
time: 0.53 seconds

## BFS IS instance: IS_B1_A2_B2

### Backsubstitution after applying IS history:
0: -8.8551903, 33.0941162, -6.7437096, 25.4687443, -34.3239365, 39.8378220
1: -10.4291992, 37.5969582, -7.8409715, 28.8711681, -39.3003616, 45.4379311
2: -10.9747181, 37.6314697, -8.3597012, 28.9183235, -39.8930435, 45.9911652
3: -16.3464375, 38.8504906, -12.3940144, 29.6448441, -45.9912758, 51.2445068
4: -17.2999840, 37.0951767, -13.1956530, 28.4903889, -45.7903748, 50.2908287

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_B2_A1

### Relational analysis result of IS_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5692031, upper bound: 43.5586702
time: 0.68 seconds

## Relational analysis of IS_B1_A2_B2_A2

### Relational analysis result of IS_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5442409, upper bound: 43.5553418
time: 0.48 seconds

## BFS IS instance: IS_B2_A1_B1

### Backsubstitution after applying IS history:
0: -6.4284191, 24.1388493, -8.6322088, 32.3104172, -38.7388306, 32.7710571
1: -7.4645872, 27.3031864, -10.1549702, 36.7054749, -44.1700592, 37.4581566
2: -7.9618182, 27.3991623, -10.6975584, 36.7274017, -44.6892204, 38.0967178
3: -11.7890568, 27.9809723, -15.9301043, 37.9201851, -49.7092400, 43.9110756
4: -12.4559155, 27.1424770, -16.8876266, 36.1860275, -48.6419373, 44.0301056

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_A1_B1_A1

### Relational analysis result of IS_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5725862, upper bound: 43.5726330
time: 0.49 seconds

## Relational analysis of IS_B2_A1_B1_A2

### Relational analysis result of IS_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5586702, upper bound: 43.5692031
time: 0.77 seconds

## BFS IS instance: IS_B2_A1_B2

### Backsubstitution after applying IS history:
0: -6.4284191, 24.1388493, -8.8174801, 33.0339127, -39.4623260, 32.9563293
1: -7.4645872, 27.3031864, -10.3915501, 37.5448494, -45.0094337, 37.6947365
2: -7.9618182, 27.3991623, -10.9315701, 37.5578690, -45.5196877, 38.3307343
3: -11.7890568, 27.9809723, -16.3067951, 38.8084564, -50.5975113, 44.2877617
4: -12.4559155, 27.1424770, -17.2786331, 36.9893837, -49.4452972, 44.4211121

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_A1_B2_A1

### Relational analysis result of IS_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5692578, upper bound: 43.5476707
time: 0.56 seconds

## Relational analysis of IS_B2_A1_B2_A2

### Relational analysis result of IS_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5553418, upper bound: 43.5442409
time: 0.51 seconds

## BFS IS instance: IS_B2_A2_A1

### Backsubstitution after applying IS history:
0: -8.6322088, 32.3104172, -8.8551903, 33.0941162, -41.7263222, 41.1656075
1: -10.1549702, 36.7054749, -10.4291992, 37.5969582, -47.7519264, 47.1346703
2: -10.6975584, 36.7274017, -10.9747181, 37.6314697, -48.3290291, 47.7021179
3: -15.9301043, 37.9201851, -16.3464375, 38.8504906, -54.7805939, 54.2666245
4: -16.8876266, 36.1860275, -17.2999840, 37.0951767, -53.9828033, 53.4860115

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B2_A2_A1_B1

### Relational analysis result of IS_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4317113, upper bound: 43.4934255
time: 0.63 seconds

## Relational analysis of IS_B2_A2_A1_B2

### Relational analysis result of IS_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3821234, upper bound: 43.3843393
time: 1.01 seconds

## BFS IS instance: IS_B2_A2_A2

### Backsubstitution after applying IS history:
0: -8.8174801, 33.0339127, -8.8551903, 33.0941162, -41.9115982, 41.8891029
1: -10.3915501, 37.5448494, -10.4291992, 37.5969582, -47.9885063, 47.9740448
2: -10.9315701, 37.5578690, -10.9747181, 37.6314697, -48.5630417, 48.5325851
3: -16.3067951, 38.8084564, -16.3464375, 38.8504906, -55.1572800, 55.1548920
4: -17.2786331, 36.9893837, -17.2999840, 37.0951767, -54.3738098, 54.2893677

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B2_A2_A2_A1

### Relational analysis result of IS_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5387238, upper bound: 43.5154913
time: 0.54 seconds

## Relational analysis of IS_B2_A2_A2_A2

### Relational analysis result of IS_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476
time: 0.50 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.65 seconds
IS_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.65
Output dim: 3, lower bound: -43.5839320, upper bound: 43.5839320
IS_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.65
Output dim: 3, lower bound: -43.5839320, upper bound: 43.5839320
IS_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.65
Output dim: 3, lower bound: -43.5839320, upper bound: 43.5839320
IS_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.65
Output dim: 3, lower bound: -43.5839320, upper bound: 43.5839320
IS_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.65
Output dim: 3, lower bound: -43.5726330, upper bound: 43.5725862
IS_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.65
Output dim: 3, lower bound: -43.5476707, upper bound: 43.5692578
IS_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.65
Output dim: 3, lower bound: -43.5692031, upper bound: 43.5586702
IS_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.65
Output dim: 3, lower bound: -43.5442409, upper bound: 43.5553418
IS_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.65
Output dim: 3, lower bound: -43.5725862, upper bound: 43.5726330
IS_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.65
Output dim: 3, lower bound: -43.5586702, upper bound: 43.5692031
IS_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.65
Output dim: 3, lower bound: -43.5692578, upper bound: 43.5476707
IS_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.65
Output dim: 3, lower bound: -43.5553418, upper bound: 43.5442409
IS_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.65
Output dim: 3, lower bound: -43.4317113, upper bound: 43.4934255
IS_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.65
Output dim: 3, lower bound: -43.3821234, upper bound: 43.3843393
IS_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 3.65
Output dim: 3, lower bound: -43.5387238, upper bound: 43.5154913
IS_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 3.65
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476

## BFS IS instance: IS_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -5.2551436, 20.1315994, -5.2551436, 20.1315994, -25.3867397, 25.3867416
1: -6.0467672, 22.8425674, -6.0467672, 22.8425674, -28.8893318, 28.8893318
2: -6.4991312, 22.7556610, -6.4991312, 22.7556610, -29.2547913, 29.2547913
3: -9.6570997, 23.3396664, -9.6570997, 23.3396664, -32.9967651, 32.9967651
4: -10.4297295, 22.2815781, -10.4297295, 22.2815781, -32.7113037, 32.7113037

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B1_A1_A1

### Relational analysis result of IS_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5794141, upper bound: 43.5370581
time: 0.55 seconds

## Relational analysis of IS_B1_A1_B1_A1_A2

### Relational analysis result of IS_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5325401, upper bound: 43.5325401
time: 0.50 seconds

## BFS IS instance: IS_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -6.7437096, 25.4687443, -5.2551436, 20.1315994, -26.8753090, 30.7238846
1: -7.8409715, 28.8711681, -6.0467672, 22.8425674, -30.6835365, 34.9179344
2: -8.3597012, 28.9183235, -6.4991312, 22.7556610, -31.1153622, 35.4174538
3: -12.3940144, 29.6448441, -9.6570997, 23.3396664, -35.7336807, 39.3019371
4: -13.1956530, 28.4903889, -10.4297295, 22.2815781, -35.4772301, 38.9201202

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B1_A2_A1

### Relational analysis result of IS_B1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5794141, upper bound: 43.5370581
time: 0.47 seconds

## Relational analysis of IS_B1_A1_B1_A2_A2

### Relational analysis result of IS_B1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5325401, upper bound: 43.5367519
time: 0.49 seconds

## BFS IS instance: IS_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -5.2551436, 20.1315994, -6.7437096, 25.4687443, -30.7238846, 26.8753090
1: -6.0467672, 22.8425674, -7.8409715, 28.8711681, -34.9179344, 30.6835365
2: -6.4991312, 22.7556610, -8.3597012, 28.9183235, -35.4174538, 31.1153622
3: -9.6570997, 23.3396664, -12.3940144, 29.6448441, -39.3019371, 35.7336807
4: -10.4297295, 22.2815781, -13.1956530, 28.4903889, -38.9201202, 35.4772301

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B2_A1_B1

### Relational analysis result of IS_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5370581, upper bound: 43.5794141
time: 0.54 seconds

## Relational analysis of IS_B1_A1_B2_A1_B2

### Relational analysis result of IS_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5325401, upper bound: 43.5767383
time: 0.53 seconds

## BFS IS instance: IS_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -6.7437096, 25.4687443, -6.7437096, 25.4687443, -32.2124519, 32.2124519
1: -7.8409715, 28.8711681, -7.8409715, 28.8711681, -36.7121353, 36.7121353
2: -8.3597012, 28.9183235, -8.3597012, 28.9183235, -37.2780190, 37.2780190
3: -12.3940144, 29.6448441, -12.3940144, 29.6448441, -42.0388565, 42.0388565
4: -13.1956530, 28.4903889, -13.1956530, 28.4903889, -41.6860428, 41.6860428

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B2_A2_B1

### Relational analysis result of IS_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5612516, upper bound: 43.5804565
time: 0.48 seconds

## Relational analysis of IS_B1_A1_B2_A2_B2

### Relational analysis result of IS_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5586812, upper bound: 43.5586812
time: 0.49 seconds

## BFS IS instance: IS_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -8.6322088, 32.3104172, -5.2551436, 20.1315994, -28.7638092, 37.5655594
1: -10.1549702, 36.7054749, -6.0467672, 22.8425674, -32.9975319, 42.7522430
2: -10.6975584, 36.7274017, -6.4991312, 22.7556610, -33.4532204, 43.2265320
3: -15.9301043, 37.9201851, -9.6570997, 23.3396664, -39.2697716, 47.5772858
4: -16.8876266, 36.1860275, -10.4297295, 22.2815781, -39.1692047, 46.6157532

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A2_B1_A1_B1

### Relational analysis result of IS_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5604672, upper bound: 43.5721456
time: 0.71 seconds

## Relational analysis of IS_B1_A2_B1_A1_B2

### Relational analysis result of IS_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5557536, upper bound: 43.5263719
time: 0.48 seconds

## BFS IS instance: IS_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -8.8174801, 33.0339127, -5.2551436, 20.1315994, -28.9490757, 38.2890549
1: -10.3915501, 37.5448494, -6.0467672, 22.8425674, -33.2341118, 43.5916176
2: -10.9315701, 37.5578690, -6.4991312, 22.7556610, -33.6872330, 44.0569992
3: -16.3067951, 38.8084564, -9.6570997, 23.3396664, -39.6464577, 48.4655533
4: -17.2786331, 36.9893837, -10.4297295, 22.2815781, -39.5602112, 47.4191132

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A2_B1_A2_B1

### Relational analysis result of IS_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5223952, upper bound: 43.5670800
time: 0.49 seconds

## Relational analysis of IS_B1_A2_B1_A2_B2

### Relational analysis result of IS_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5176817, upper bound: 43.5213063
time: 0.49 seconds

## BFS IS instance: IS_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -8.6322088, 32.3104172, -6.7437096, 25.4687443, -34.1009483, 39.0541191
1: -10.1549702, 36.7054749, -7.8409715, 28.8711681, -39.0261307, 44.5464478
2: -10.6975584, 36.7274017, -8.3597012, 28.9183235, -39.6158791, 45.0870972
3: -15.9301043, 37.9201851, -12.3940144, 29.6448441, -45.5749435, 50.3142014
4: -16.8876266, 36.1860275, -13.1956530, 28.4903889, -45.3780136, 49.3816795

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B2_A1_B1

### Relational analysis result of IS_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4350507, upper bound: 43.5078658
time: 0.95 seconds

## Relational analysis of IS_B1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_B2_A1_B1

### Relational analysis result of IS_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5442409, upper bound: 43.5553418
time: 0.51 seconds

## Relational analysis of IS_B1_A2_B2_A1_B2

### Relational analysis result of IS_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5442409, upper bound: 43.5553418
time: 0.46 seconds

## BFS IS instance: IS_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -8.8174801, 33.0339127, -6.7437096, 25.4687443, -34.2862244, 39.7776146
1: -10.3915501, 37.5448494, -7.8409715, 28.8711681, -39.2627106, 45.3858223
2: -10.9315701, 37.5578690, -8.3597012, 28.9183235, -39.8498917, 45.9175644
3: -16.3067951, 38.8084564, -12.3940144, 29.6448441, -45.9516296, 51.2024689
4: -17.2786331, 36.9893837, -13.1956530, 28.4903889, -45.7690201, 50.1850357

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B2_A2_B1

### Relational analysis result of IS_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4351547, upper bound: 43.5057539
time: 0.70 seconds

## Relational analysis of IS_B1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B2_A2_B1

### Relational analysis result of IS_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5442409, upper bound: 43.5553418
time: 0.52 seconds

## Relational analysis of IS_B1_A2_B2_A2_B2

### Relational analysis result of IS_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5434515, upper bound: 43.5513966
time: 0.59 seconds

## BFS IS instance: IS_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -5.2551436, 20.1315994, -8.6322088, 32.3104172, -37.5655594, 28.7638092
1: -6.0467672, 22.8425674, -10.1549702, 36.7054749, -42.7522430, 32.9975319
2: -6.4991312, 22.7556610, -10.6975584, 36.7274017, -43.2265320, 33.4532204
3: -9.6570997, 23.3396664, -15.9301043, 37.9201851, -47.5772858, 39.2697716
4: -10.4297295, 22.2815781, -16.8876266, 36.1860275, -46.6157570, 39.1692047

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 17

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B2_A1_B1_A1_A1

### Relational analysis result of IS_B2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5721456, upper bound: 43.5604672
time: 0.55 seconds

## Relational analysis of IS_B2_A1_B1_A1_A2

### Relational analysis result of IS_B2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5263719, upper bound: 43.5557536
time: 0.73 seconds

## BFS IS instance: IS_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -6.7437096, 25.4687443, -8.6322088, 32.3104172, -39.0541191, 34.1009483
1: -7.8409715, 28.8711681, -10.1549702, 36.7054749, -44.5464478, 39.0261345
2: -8.3597012, 28.9183235, -10.6975584, 36.7274017, -45.0870972, 39.6158791
3: -12.3940144, 29.6448441, -15.9301043, 37.9201851, -50.3142014, 45.5749435
4: -13.1956530, 28.4903889, -16.8876266, 36.1860275, -49.3816795, 45.3780136

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B2_A1_B1_A2_A1

### Relational analysis result of IS_B2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5078658, upper bound: 43.4350507
time: 0.60 seconds

## Relational analysis of IS_B2_A1_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_B1_A2_A1

### Relational analysis result of IS_B2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5586702, upper bound: 43.5692031
time: 0.53 seconds

## Relational analysis of IS_B2_A1_B1_A2_A2

### Relational analysis result of IS_B2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5586702, upper bound: 43.5692031
time: 0.76 seconds

## BFS IS instance: IS_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -5.2551436, 20.1315994, -8.8174801, 33.0339127, -38.2890549, 28.9490757
1: -6.0467672, 22.8425674, -10.3915501, 37.5448494, -43.5916176, 33.2341118
2: -6.4991312, 22.7556610, -10.9315701, 37.5578690, -44.0569992, 33.6872330
3: -9.6570997, 23.3396664, -16.3067951, 38.8084564, -48.4655533, 39.6464577
4: -10.4297295, 22.2815781, -17.2786331, 36.9893837, -47.4191132, 39.5602112

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B2_A1_B2_A1_A1

### Relational analysis result of IS_B2_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5670800, upper bound: 43.5223952
time: 0.51 seconds

## Relational analysis of IS_B2_A1_B2_A1_A2

### Relational analysis result of IS_B2_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5213063, upper bound: 43.5176817
time: 0.76 seconds

## BFS IS instance: IS_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -6.7437096, 25.4687443, -8.8174801, 33.0339127, -39.7776146, 34.2862244
1: -7.8409715, 28.8711681, -10.3915501, 37.5448494, -45.3858223, 39.2627106
2: -8.3597012, 28.9183235, -10.9315701, 37.5578690, -45.9175644, 39.8498917
3: -12.3940144, 29.6448441, -16.3067951, 38.8084564, -51.2024689, 45.9516296
4: -13.1956530, 28.4903889, -17.2786331, 36.9893837, -50.1850357, 45.7690201

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B2_A1_B2_A2_A1

### Relational analysis result of IS_B2_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5057539, upper bound: 43.4351547
time: 0.63 seconds

## Relational analysis of IS_B2_A1_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_A1_B2_A2_A1

### Relational analysis result of IS_B2_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5553418, upper bound: 43.5442409
time: 0.47 seconds

## Relational analysis of IS_B2_A1_B2_A2_A2

### Relational analysis result of IS_B2_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5513966, upper bound: 43.5434515
time: 0.53 seconds

## BFS IS instance: IS_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -8.6322088, 32.3104172, -8.8156204, 32.9617119, -41.5939140, 41.1260338
1: -10.1549702, 36.7054749, -10.3809900, 37.4477654, -47.6027336, 47.0864639
2: -10.6975584, 36.7274017, -10.9258604, 37.4781151, -48.1756706, 47.6532516
3: -15.9301043, 37.9201851, -16.2741280, 38.6946754, -54.6247787, 54.1943130
4: -16.8876266, 36.1860275, -17.2310925, 36.9354782, -53.8231049, 53.4171143

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A2_A1_B1_B1

### Relational analysis result of IS_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4317113, upper bound: 43.4934255
time: 0.56 seconds

## Relational analysis of IS_B2_A2_A1_B1_B2

### Relational analysis result of IS_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4317113, upper bound: 43.4934255
time: 0.56 seconds

## BFS IS instance: IS_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -8.6301594, 32.3027611, -9.3646259, 35.1735840, -43.8037415, 41.6673775
1: -10.1524372, 36.6966629, -11.0751286, 40.0641594, -50.1854668, 47.7717896
2: -10.6950235, 36.7186508, -11.6250591, 39.9605408, -50.6555634, 48.3437119
3: -15.9262104, 37.9110336, -17.3567867, 41.4715424, -57.3512726, 55.2678223
4: -16.8836193, 36.1774902, -18.4965115, 39.2655792, -56.1491890, 54.6739960

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_A2_A1_B2_B1

### Relational analysis result of IS_B2_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3538075, upper bound: 43.3432026
time: 0.66 seconds

## Relational analysis of IS_B2_A2_A1_B2_B2

### Relational analysis result of IS_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3818490, upper bound: 43.3840848
time: 0.68 seconds

## BFS IS instance: IS_B2_A2_A2_A1

### Backsubstitution after applying IS history:
0: -8.5439262, 32.0631142, -8.8551903, 33.0941162, -41.6380424, 40.9183044
1: -10.0602074, 36.4324837, -10.4291992, 37.5969582, -47.6571655, 46.8616791
2: -10.5933409, 36.4442482, -10.9747181, 37.6314697, -48.2248116, 47.4189682
3: -15.7987118, 37.6562881, -16.3464375, 38.8504906, -54.6492004, 54.0027237
4: -16.7670708, 35.8889656, -17.2999840, 37.0951767, -53.8622475, 53.1889496

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B2_A2_A2_A1_B1

### Relational analysis result of IS_B2_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4275308, upper bound: 43.4514684
time: 0.94 seconds

## Relational analysis of IS_B2_A2_A2_A1_B2

### Relational analysis result of IS_B2_A2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3798398, upper bound: 43.3575275
time: 0.82 seconds

## BFS IS instance: IS_B2_A2_A2_A2

### Backsubstitution after applying IS history:
0: -9.1115875, 33.9125404, -8.8551903, 33.0941162, -42.2057037, 42.7677307
1: -10.7727814, 38.5158424, -10.4291992, 37.5969582, -48.3697395, 48.9450378
2: -11.2928276, 38.6079216, -10.9747181, 37.6314697, -48.9242973, 49.5826416
3: -16.8676147, 39.8401985, -16.3464375, 38.8504906, -55.7181015, 56.1866341
4: -17.7444897, 38.1677361, -17.2999840, 37.0951767, -54.8396683, 55.4677200

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B2_A2_A2_A2_A1

### Relational analysis result of IS_B2_A2_A2_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3760023, upper bound: 43.3610870
time: 0.80 seconds

## Relational analysis of IS_B2_A2_A2_A2_A2

### Relational analysis result of IS_B2_A2_A2_A2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3718156, upper bound: 43.3674222
time: 0.79 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.70 seconds
IS_B1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -43.5794141, upper bound: 43.5370581
IS_B1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -43.5325401, upper bound: 43.5325401
IS_B1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -43.5794141, upper bound: 43.5370581
IS_B1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -43.5325401, upper bound: 43.5367519
IS_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -43.5370581, upper bound: 43.5794141
IS_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -43.5325401, upper bound: 43.5767383
IS_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -43.5612516, upper bound: 43.5804565
IS_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -43.5586812, upper bound: 43.5586812
IS_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -43.5604672, upper bound: 43.5721456
IS_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -43.5557536, upper bound: 43.5263719
IS_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -43.5223952, upper bound: 43.5670800
IS_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -43.5176817, upper bound: 43.5213063
IS_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -43.5442409, upper bound: 43.5553418
IS_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -43.5442409, upper bound: 43.5553418
IS_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -43.5442409, upper bound: 43.5553418
IS_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -43.5434515, upper bound: 43.5513966
IS_B2_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -43.5721456, upper bound: 43.5604672
IS_B2_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -43.5263719, upper bound: 43.5557536
IS_B2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -43.5586702, upper bound: 43.5692031
IS_B2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -43.5586702, upper bound: 43.5692031
IS_B2_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -43.5670800, upper bound: 43.5223952
IS_B2_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -43.5213063, upper bound: 43.5176817
IS_B2_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -43.5553418, upper bound: 43.5442409
IS_B2_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -43.5513966, upper bound: 43.5434515
IS_B2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -43.4317113, upper bound: 43.4934255
IS_B2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -43.4317113, upper bound: 43.4934255
IS_B2_A2_A1_B2_B1, status: Status.VERIFIED, split count: 5, time: 4.70
Output dim: 3, lower bound: -43.3538075, upper bound: 43.3432026
IS_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -43.3818490, upper bound: 43.3840848
IS_B2_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -43.4275308, upper bound: 43.4514684
IS_B2_A2_A2_A1_B2, status: Status.VERIFIED, split count: 5, time: 4.70
Output dim: 3, lower bound: -43.3798398, upper bound: 43.3575275
IS_B2_A2_A2_A2_A1, status: Status.VERIFIED, split count: 5, time: 4.70
Output dim: 3, lower bound: -43.3760023, upper bound: 43.3610870
IS_B2_A2_A2_A2_A2, status: Status.VERIFIED, split count: 5, time: 4.70
Output dim: 3, lower bound: -43.3718156, upper bound: 43.3674222

## BFS IS instance: IS_B1_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -4.9090037, 18.9296074, -5.2551436, 20.1315994, -25.0405998, 24.1847477
1: -5.6224365, 21.5094490, -6.0467672, 22.8425674, -28.4650040, 27.5562134
2: -6.0714874, 21.3573895, -6.4991312, 22.7556610, -28.8271484, 27.8565197
3: -9.0102825, 21.9570026, -9.6570997, 23.3396664, -32.3499489, 31.6141014
4: -9.8312683, 20.8039608, -10.4297295, 22.2815781, -32.1128387, 31.2336884

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B1_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B1_A1_A1_B1

### Relational analysis result of IS_B1_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5325401, upper bound: 43.5325401
time: 0.55 seconds

## Relational analysis of IS_B1_A1_B1_A1_A1_B2

### Relational analysis result of IS_B1_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5325401, upper bound: 43.5325401
time: 0.51 seconds

## BFS IS instance: IS_B1_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -5.4008245, 20.7929802, -5.2551436, 20.1315994, -25.5324249, 26.0481224
1: -6.2052450, 23.6457863, -6.0467672, 22.8425674, -29.0478134, 29.6925545
2: -6.6841125, 23.4871082, -6.4991312, 22.7556610, -29.4397736, 29.9862385
3: -9.9212875, 24.1726971, -9.6570997, 23.3396664, -33.2609558, 33.8297958
4: -10.8100157, 22.8875217, -10.4297295, 22.2815781, -33.0915947, 33.3172455

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B1_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B1_A1_A2_B1

### Relational analysis result of IS_B1_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5325401, upper bound: 43.5325401
time: 0.53 seconds

## Relational analysis of IS_B1_A1_B1_A1_A2_B2

### Relational analysis result of IS_B1_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5325401, upper bound: 43.5325401
time: 0.55 seconds

## BFS IS instance: IS_B1_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -6.3014240, 23.9654980, -5.2551436, 20.1315994, -26.4330235, 29.2206402
1: -7.3049097, 27.2115173, -6.0467672, 22.8425674, -30.1474743, 33.2582817
2: -7.8123007, 27.1727753, -6.4991312, 22.7556610, -30.5679588, 33.6719055
3: -11.5818949, 27.9037056, -9.6570997, 23.3396664, -34.9215622, 37.5608025
4: -12.4305439, 26.6641655, -10.4297295, 22.2815781, -34.7121201, 37.0938835

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B1_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B1_A2_A1_A1

### Relational analysis result of IS_B1_A1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5597501, upper bound: 43.4726629
time: 0.52 seconds

## Relational analysis of IS_B1_A1_B1_A2_A1_A2

### Relational analysis result of IS_B1_A1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5491357, upper bound: 43.4708694
time: 0.51 seconds

## BFS IS instance: IS_B1_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -6.9152741, 26.2212715, -5.2551436, 20.1315994, -27.0468731, 31.4764118
1: -8.0347805, 29.7921047, -6.0467672, 22.8425674, -30.8773460, 35.8388710
2: -8.5723772, 29.7500801, -6.4991312, 22.7556610, -31.3280373, 36.2492104
3: -12.7084513, 30.5911541, -9.6570997, 23.3396664, -36.0481148, 40.2482491
4: -13.6220407, 29.1891994, -10.4297295, 22.2815781, -35.9036140, 39.6189232

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B1_A2_A2_A1

### Relational analysis result of IS_B1_A1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5581367, upper bound: 43.4724648
time: 0.71 seconds

## Relational analysis of IS_B1_A1_B1_A2_A2_A2

### Relational analysis result of IS_B1_A1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5451905, upper bound: 43.4700800
time: 0.55 seconds

## BFS IS instance: IS_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -5.2551436, 20.1315994, -6.3014240, 23.9654980, -29.2206402, 26.4330215
1: -6.0467672, 22.8425674, -7.3049097, 27.2115173, -33.2582817, 30.1474743
2: -6.4991312, 22.7556610, -7.8123007, 27.1727753, -33.6719055, 30.5679588
3: -9.6570997, 23.3396664, -11.5818949, 27.9037056, -37.5608063, 34.9215622
4: -10.4297295, 22.2815781, -12.4305439, 26.6641655, -37.0938835, 34.7121201

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B2_A1_B1_B1

### Relational analysis result of IS_B1_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4726629, upper bound: 43.5597501
time: 0.53 seconds

## Relational analysis of IS_B1_A1_B2_A1_B1_B2

### Relational analysis result of IS_B1_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4708694, upper bound: 43.5491357
time: 0.55 seconds

## BFS IS instance: IS_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -5.2551436, 20.1315994, -6.9152741, 26.2212715, -31.4764137, 27.0468731
1: -6.0467672, 22.8425674, -8.0347805, 29.7921047, -35.8388710, 30.8773479
2: -6.4991312, 22.7556610, -8.5723772, 29.7500801, -36.2492104, 31.3280373
3: -9.6570997, 23.3396664, -12.7084513, 30.5911541, -40.2482491, 36.0481148
4: -10.4297295, 22.2815781, -13.6220407, 29.1891994, -39.6189232, 35.9036140

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B2_A1_B2_B1

### Relational analysis result of IS_B1_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4724648, upper bound: 43.5581367
time: 0.49 seconds

## Relational analysis of IS_B1_A1_B2_A1_B2_B2

### Relational analysis result of IS_B1_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4700800, upper bound: 43.5451905
time: 0.84 seconds

## BFS IS instance: IS_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -6.7437096, 25.4687443, -6.5283041, 24.7212219, -31.4649315, 31.9970474
1: -7.8409715, 28.8711681, -7.5771909, 28.0262680, -35.8672409, 36.4483604
2: -8.3597012, 28.9183235, -8.0923138, 28.0525475, -36.4122467, 37.0106354
3: -12.3940144, 29.6448441, -11.9928169, 28.7598476, -41.1538620, 41.6376572
4: -13.1956530, 28.4903889, -12.8025274, 27.6119270, -40.8075714, 41.2929153

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5591247, upper bound: 43.5770626
time: 0.57 seconds

## Relational analysis of IS_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5569962, upper bound: 43.5769161
time: 0.65 seconds

## BFS IS instance: IS_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -6.7437096, 25.4687443, -6.7093639, 25.4152355, -32.1589432, 32.1781082
1: -7.8409715, 28.8711681, -7.8063669, 28.8139172, -36.6548882, 36.6775246
2: -8.3597012, 28.9183235, -8.3207026, 28.8522396, -37.2119370, 37.2390251
3: -12.3940144, 29.6448441, -12.3585405, 29.6142330, -42.0082474, 42.0033836
4: -13.1956530, 28.4903889, -13.1818914, 28.3913708, -41.5870247, 41.6722794

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5491357, upper bound: 43.5539200
time: 0.62 seconds

## Relational analysis of IS_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5539466, upper bound: 43.5539200
time: 0.77 seconds

## BFS IS instance: IS_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -8.6322088, 32.3104172, -5.0394983, 19.3376427, -27.9698524, 37.3499107
1: -10.1549702, 36.7054749, -5.7875409, 21.9354172, -32.0903816, 42.4930115
2: -10.6975584, 36.7274017, -6.2317314, 21.8455486, -32.5431061, 42.9591331
3: -15.9301043, 37.9201851, -9.2545710, 22.4035969, -38.3336983, 47.1747551
4: -16.8876266, 36.1860275, -10.0192795, 21.3928051, -38.2804337, 46.2053032

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_A1_B1_B1

### Relational analysis result of IS_B1_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5602780, upper bound: 43.5720115
time: 0.97 seconds

## Relational analysis of IS_B1_A2_B1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_B1_A1_B1_B1

### Relational analysis result of IS_B1_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5604672, upper bound: 43.5721456
time: 0.49 seconds

## Relational analysis of IS_B1_A2_B1_A1_B1_B2

### Relational analysis result of IS_B1_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5604672, upper bound: 43.5721456
time: 0.73 seconds

## BFS IS instance: IS_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -8.6322088, 32.3104172, -5.5686293, 21.0664139, -29.6986237, 37.8790474
1: -10.1549702, 36.7054749, -6.4469433, 23.8585854, -34.0135460, 43.1524200
2: -10.6975584, 36.7274017, -6.8837023, 23.8715839, -34.5691376, 43.6110992
3: -15.9301043, 37.9201851, -10.2405462, 24.4116497, -40.3417549, 48.1607323
4: -16.8876266, 36.1860275, -10.9216919, 23.5361233, -40.4237480, 47.1077194

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_A1_B2_B1

### Relational analysis result of IS_B1_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5555644, upper bound: 43.5259161
time: 0.55 seconds

## Relational analysis of IS_B1_A2_B1_A1_B2_B2

### Relational analysis result of IS_B1_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5472828, upper bound: 43.4704082
time: 0.53 seconds

## BFS IS instance: IS_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.8174801, 33.0339127, -5.0394983, 19.3376427, -28.1551189, 38.0734024
1: -10.3915501, 37.5448494, -5.7875409, 21.9354172, -32.3269577, 43.3323860
2: -10.9315701, 37.5578690, -6.2317314, 21.8455486, -32.7771187, 43.7896004
3: -16.3067951, 38.8084564, -9.2545710, 22.4035969, -38.7103844, 48.0630264
4: -17.2786331, 36.9893837, -10.0192795, 21.3928051, -38.6714401, 47.0086632

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_A2_B1_B1

### Relational analysis result of IS_B1_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5223952, upper bound: 43.5669459
time: 0.52 seconds

## Relational analysis of IS_B1_A2_B1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_B1_A2_B1_B1

### Relational analysis result of IS_B1_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5223952, upper bound: 43.5670800
time: 0.51 seconds

## Relational analysis of IS_B1_A2_B1_A2_B1_B2

### Relational analysis result of IS_B1_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5223952, upper bound: 43.5670800
time: 0.50 seconds

## BFS IS instance: IS_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -8.8174801, 33.0339127, -5.5686293, 21.0664139, -29.8838902, 38.6025429
1: -10.3915501, 37.5448494, -6.4469433, 23.8585854, -34.2501297, 43.9917908
2: -10.9315701, 37.5578690, -6.8837023, 23.8715839, -34.8031502, 44.4415703
3: -16.3067951, 38.8084564, -10.2405462, 24.4116497, -40.7184448, 49.0490036
4: -17.2786331, 36.9893837, -10.9216919, 23.5361233, -40.8147583, 47.9110756

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_A2_B2_B1

### Relational analysis result of IS_B1_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5176817, upper bound: 43.5208505
time: 0.56 seconds

## Relational analysis of IS_B1_A2_B1_A2_B2_B2

### Relational analysis result of IS_B1_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5094192, upper bound: 43.4653523
time: 0.53 seconds

## BFS IS instance: IS_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -8.6322088, 32.3104172, -6.5283041, 24.7212219, -33.3534317, 38.8387222
1: -10.1549702, 36.7054749, -7.5771909, 28.0262680, -38.1812363, 44.2826653
2: -10.6975584, 36.7274017, -8.0923138, 28.0525475, -38.7501068, 44.8197174
3: -15.9301043, 37.9201851, -11.9928169, 28.7598476, -44.6899490, 49.9130020
4: -16.8876266, 36.1860275, -12.8025274, 27.6119270, -44.4995499, 48.9885559

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B2_A1_B1_B1

### Relational analysis result of IS_B1_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4350507, upper bound: 43.5078658
time: 0.70 seconds

## Relational analysis of IS_B1_A2_B2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B2_A1_B1_B1

### Relational analysis result of IS_B1_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5692031, upper bound: 43.5586702
time: 0.51 seconds

## Relational analysis of IS_B1_A2_B2_A1_B1_B2

### Relational analysis result of IS_B1_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5684137, upper bound: 43.5547250
time: 0.58 seconds

## BFS IS instance: IS_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -8.6322088, 32.3104172, -6.7093639, 25.4152355, -34.0474434, 39.0197792
1: -10.1549702, 36.7054749, -7.8063669, 28.8139172, -38.9688873, 44.5118370
2: -10.6975584, 36.7274017, -8.3207026, 28.8522396, -39.5497971, 45.0481033
3: -15.9301043, 37.9201851, -12.3585405, 29.6142330, -45.5443382, 50.2787247
4: -16.8876266, 36.1860275, -13.1818914, 28.3913708, -45.2789993, 49.3679161

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B2_A1_B2_B1

### Relational analysis result of IS_B1_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5692031, upper bound: 43.5586702
time: 0.74 seconds

## Relational analysis of IS_B1_A2_B2_A1_B2_B2

### Relational analysis result of IS_B1_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5684137, upper bound: 43.5547250
time: 0.66 seconds

## BFS IS instance: IS_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -8.8174801, 33.0339127, -6.3014240, 23.9654980, -32.7829781, 39.3353348
1: -10.3915501, 37.5448494, -7.3049097, 27.2115173, -37.6030579, 44.8497543
2: -10.9315701, 37.5578690, -7.8123007, 27.1727753, -38.1043472, 45.3701706
3: -16.3067951, 38.8084564, -11.5818949, 27.9037056, -44.2104950, 50.3903503
4: -17.2786331, 36.9893837, -12.4305439, 26.6641655, -43.9427948, 49.4199295

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A2_B2_A2_B1_B1

### Relational analysis result of IS_B1_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5189648, upper bound: 43.5531641
time: 0.67 seconds

## Relational analysis of IS_B1_A2_B2_A2_B1_B2

### Relational analysis result of IS_B1_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5165910, upper bound: 43.5189613
time: 0.53 seconds

## BFS IS instance: IS_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.8174801, 33.0339127, -6.9152741, 26.2212715, -35.0387497, 39.9491806
1: -10.3915501, 37.5448494, -8.0347805, 29.7921047, -40.1836472, 45.5796280
2: -10.9315701, 37.5578690, -8.5723772, 29.7500801, -40.6816483, 46.1302452
3: -16.3067951, 38.8084564, -12.7084513, 30.5911541, -46.8979416, 51.5169029
4: -17.2786331, 36.9893837, -13.6220407, 29.1891994, -46.4678307, 50.6114235

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_B2_A2_B2_B1

### Relational analysis result of IS_B1_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5434515, upper bound: 43.5513966
time: 0.63 seconds

## Relational analysis of IS_B1_A2_B2_A2_B2_B2

### Relational analysis result of IS_B1_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5434515, upper bound: 43.5513966
time: 0.65 seconds

## BFS IS instance: IS_B2_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -5.0394983, 19.3376427, -8.6322088, 32.3104172, -37.3499107, 27.9698524
1: -5.7875409, 21.9354172, -10.1549702, 36.7054749, -42.4930115, 32.0903816
2: -6.2317314, 21.8455486, -10.6975584, 36.7274017, -42.9591331, 32.5431061
3: -9.2545710, 22.4035969, -15.9301043, 37.9201851, -47.1747551, 38.3336983
4: -10.0192795, 21.3928051, -16.8876266, 36.1860275, -46.2053032, 38.2804337

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B2_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_A1_B1_A1_A1_A1

### Relational analysis result of IS_B2_A1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5720115, upper bound: 43.5602780
time: 0.53 seconds

## Relational analysis of IS_B2_A1_B1_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_B2_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_B1_A1_A1_A1

### Relational analysis result of IS_B2_A1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5721456, upper bound: 43.5604672
time: 0.53 seconds

## Relational analysis of IS_B2_A1_B1_A1_A1_A2

### Relational analysis result of IS_B2_A1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5721456, upper bound: 43.5604672
time: 0.56 seconds

## BFS IS instance: IS_B2_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -5.5686293, 21.0664139, -8.6322088, 32.3104172, -37.8790474, 29.6986237
1: -6.4469433, 23.8585854, -10.1549702, 36.7054749, -43.1524200, 34.0135460
2: -6.8837023, 23.8715839, -10.6975584, 36.7274017, -43.6110992, 34.5691376
3: -10.2405462, 24.4116497, -15.9301043, 37.9201851, -48.1607323, 40.3417549
4: -10.9216919, 23.5361233, -16.8876266, 36.1860275, -47.1077194, 40.4237480

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B2_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_A1_B1_A1_A2_A1

### Relational analysis result of IS_B2_A1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5259161, upper bound: 43.5555644
time: 0.56 seconds

## Relational analysis of IS_B2_A1_B1_A1_A2_A2

### Relational analysis result of IS_B2_A1_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4704082, upper bound: 43.5472828
time: 0.53 seconds

## BFS IS instance: IS_B2_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -6.5283041, 24.7212219, -8.6322088, 32.3104172, -38.8387222, 33.3534279
1: -7.5771909, 28.0262680, -10.1549702, 36.7054749, -44.2826653, 38.1812363
2: -8.0923138, 28.0525475, -10.6975584, 36.7274017, -44.8197174, 38.7501068
3: -11.9928169, 28.7598476, -15.9301043, 37.9201851, -49.9130020, 44.6899529
4: -12.8025274, 27.6119270, -16.8876266, 36.1860275, -48.9885559, 44.4995499

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B2_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B2_A1_B1_A2_A1_A1

### Relational analysis result of IS_B2_A1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5078658, upper bound: 43.4350507
time: 0.67 seconds

## Relational analysis of IS_B2_A1_B1_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_A1_B1_A2_A1_A1

### Relational analysis result of IS_B2_A1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5586702, upper bound: 43.5692031
time: 0.53 seconds

## Relational analysis of IS_B2_A1_B1_A2_A1_A2

### Relational analysis result of IS_B2_A1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5547250, upper bound: 43.5684137
time: 0.60 seconds

## BFS IS instance: IS_B2_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -6.7093639, 25.4152355, -8.6322088, 32.3104172, -39.0197792, 34.0474396
1: -7.8063669, 28.8139172, -10.1549702, 36.7054749, -44.5118370, 38.9688873
2: -8.3207026, 28.8522396, -10.6975584, 36.7274017, -45.0481033, 39.5497971
3: -12.3585405, 29.6142330, -15.9301043, 37.9201851, -50.2787247, 45.5443382
4: -13.1818914, 28.3913708, -16.8876266, 36.1860275, -49.3679161, 45.2789993

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B2_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_A1_B1_A2_A2_A1

### Relational analysis result of IS_B2_A1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5586702, upper bound: 43.5692031
time: 0.81 seconds

## Relational analysis of IS_B2_A1_B1_A2_A2_A2

### Relational analysis result of IS_B2_A1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5547250, upper bound: 43.5684137
time: 0.57 seconds

## BFS IS instance: IS_B2_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -5.0394983, 19.3376427, -8.8174801, 33.0339127, -38.0734024, 28.1551189
1: -5.7875409, 21.9354172, -10.3915501, 37.5448494, -43.3323860, 32.3269615
2: -6.2317314, 21.8455486, -10.9315701, 37.5578690, -43.7896004, 32.7771187
3: -9.2545710, 22.4035969, -16.3067951, 38.8084564, -48.0630264, 38.7103844
4: -10.0192795, 21.3928051, -17.2786331, 36.9893837, -47.0086632, 38.6714401

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B2_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_A1_B2_A1_A1_A1

### Relational analysis result of IS_B2_A1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5669459, upper bound: 43.5223952
time: 0.58 seconds

## Relational analysis of IS_B2_A1_B2_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_B2_A1_A1_A1

### Relational analysis result of IS_B2_A1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5670800, upper bound: 43.5223952
time: 0.51 seconds

## Relational analysis of IS_B2_A1_B2_A1_A1_A2

### Relational analysis result of IS_B2_A1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5670800, upper bound: 43.5223952
time: 0.54 seconds

## BFS IS instance: IS_B2_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -5.5686293, 21.0664139, -8.8174801, 33.0339127, -38.6025429, 29.8838902
1: -6.4469433, 23.8585854, -10.3915501, 37.5448494, -43.9917908, 34.2501297
2: -6.8837023, 23.8715839, -10.9315701, 37.5578690, -44.4415703, 34.8031502
3: -10.2405462, 24.4116497, -16.3067951, 38.8084564, -49.0490036, 40.7184448
4: -10.9216919, 23.5361233, -17.2786331, 36.9893837, -47.9110756, 40.8147583

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B2_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_A1_B2_A1_A2_A1

### Relational analysis result of IS_B2_A1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5208505, upper bound: 43.5176817
time: 0.50 seconds

## Relational analysis of IS_B2_A1_B2_A1_A2_A2

### Relational analysis result of IS_B2_A1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4653523, upper bound: 43.5094192
time: 0.52 seconds

## BFS IS instance: IS_B2_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -6.3014240, 23.9654980, -8.8174801, 33.0339127, -39.3353348, 32.7829781
1: -7.3049097, 27.2115173, -10.3915501, 37.5448494, -44.8497543, 37.6030579
2: -7.8123007, 27.1727753, -10.9315701, 37.5578690, -45.3701706, 38.1043472
3: -11.5818949, 27.9037056, -16.3067951, 38.8084564, -50.3903503, 44.2104988
4: -12.4305439, 26.6641655, -17.2786331, 36.9893837, -49.4199295, 43.9427986

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B2_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B2_A1_B2_A2_A1_A1

### Relational analysis result of IS_B2_A1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5531641, upper bound: 43.5189648
time: 0.51 seconds

## Relational analysis of IS_B2_A1_B2_A2_A1_A2

### Relational analysis result of IS_B2_A1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5189613, upper bound: 43.5165910
time: 0.52 seconds

## BFS IS instance: IS_B2_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -6.9152741, 26.2212715, -8.8174801, 33.0339127, -39.9491806, 35.0387497
1: -8.0347805, 29.7921047, -10.3915501, 37.5448494, -45.5796280, 40.1836510
2: -8.5723772, 29.7500801, -10.9315701, 37.5578690, -46.1302452, 40.6816483
3: -12.7084513, 30.5911541, -16.3067951, 38.8084564, -51.5169029, 46.8979416
4: -13.6220407, 29.1891994, -17.2786331, 36.9893837, -50.6114235, 46.4678307

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B2_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_A1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_B2_A2_A2_A1

### Relational analysis result of IS_B2_A1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5513966, upper bound: 43.5434515
time: 0.53 seconds

## Relational analysis of IS_B2_A1_B2_A2_A2_A2

### Relational analysis result of IS_B2_A1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5513966, upper bound: 43.5434515
time: 0.56 seconds

## BFS IS instance: IS_B2_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -8.6322088, 32.3104172, -8.5925446, 32.1775818, -40.8097839, 40.9029617
1: -10.1549702, 36.7054749, -10.1066513, 36.5556908, -46.7106590, 46.8121262
2: -10.6975584, 36.7274017, -10.6485958, 36.5736465, -47.2712021, 47.3759995
3: -15.9301043, 37.9201851, -15.8576345, 37.7638512, -53.6939545, 53.7778206
4: -16.8876266, 36.1860275, -16.8185234, 36.0260582, -52.9136848, 53.0045433

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_B2_A2_A1_B1_B1_A1

### Relational analysis result of IS_B2_A2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4196544, upper bound: 43.4523578
time: 0.71 seconds

## Relational analysis of IS_B2_A2_A1_B1_B1_A2

### Relational analysis result of IS_B2_A2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4303320, upper bound: 43.4760308
time: 0.71 seconds

## BFS IS instance: IS_B2_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -8.6322088, 32.3104172, -8.7776985, 32.9010162, -41.5332222, 41.0881119
1: -10.1549702, 36.7054749, -10.3430977, 37.3951607, -47.5501251, 47.0485725
2: -10.6975584, 36.7274017, -10.8825235, 37.4039421, -48.1014977, 47.6099205
3: -15.9301043, 37.9201851, -16.2341270, 38.6523247, -54.5824280, 54.1543121
4: -16.8876266, 36.1860275, -17.2096958, 36.8289986, -53.7166252, 53.3957100

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B2_A2_A1_B1_B2_A1

### Relational analysis result of IS_B2_A2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4277048, upper bound: 43.4514723
time: 0.77 seconds

## Relational analysis of IS_B2_A2_A1_B1_B2_A2

### Relational analysis result of IS_B2_A2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4256336, upper bound: 43.4678438
time: 0.67 seconds

## BFS IS instance: IS_B2_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -8.6301594, 32.3027611, -9.3598814, 35.1582031, -43.7883606, 41.6626434
1: -10.1524372, 36.6966629, -11.0692644, 40.0464897, -50.1679039, 47.7659264
2: -10.6950235, 36.7186508, -11.6192999, 39.9427032, -50.6377220, 48.3379517
3: -15.9262104, 37.9110336, -17.3481617, 41.4529648, -57.3327713, 55.2591934
4: -16.8836193, 36.1774902, -18.4878654, 39.2475166, -56.1311340, 54.6653519

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B2_A2_A1_B2_B2_A1

### Relational analysis result of IS_B2_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3818490, upper bound: 43.3840848
time: 1.06 seconds

## Relational analysis of IS_B2_A2_A1_B2_B2_A2

### Relational analysis result of IS_B2_A2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3818490, upper bound: 43.3840848
time: 0.63 seconds

## BFS IS instance: IS_B2_A2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -8.5439262, 32.0631142, -8.8156204, 32.9617119, -41.5056381, 40.8787308
1: -10.0602074, 36.4324837, -10.3809900, 37.4477654, -47.5079727, 46.8134727
2: -10.5933409, 36.4442482, -10.9258604, 37.4781151, -48.0714531, 47.3700981
3: -15.7987118, 37.6562881, -16.2741280, 38.6946754, -54.4933853, 53.9304161
4: -16.7670708, 35.8889656, -17.2310925, 36.9354782, -53.7025490, 53.1200562

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_A2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_A2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A2_A2_A1_B1_A1

### Relational analysis result of IS_B2_A2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3884527, upper bound: 43.4359418
time: 0.58 seconds

## Relational analysis of IS_B2_A2_A2_A1_B1_A2

### Relational analysis result of IS_B2_A2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4273030, upper bound: 43.4511965
time: 0.58 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 5.23 seconds
IS_B1_A1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -43.5325401, upper bound: 43.5325401
IS_B1_A1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -43.5325401, upper bound: 43.5325401
IS_B1_A1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -43.5325401, upper bound: 43.5325401
IS_B1_A1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -43.5325401, upper bound: 43.5325401
IS_B1_A1_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -43.5597501, upper bound: 43.4726629
IS_B1_A1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -43.5491357, upper bound: 43.4708694
IS_B1_A1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -43.5581367, upper bound: 43.4724648
IS_B1_A1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -43.5451905, upper bound: 43.4700800
IS_B1_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -43.4726629, upper bound: 43.5597501
IS_B1_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -43.4708694, upper bound: 43.5491357
IS_B1_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -43.4724648, upper bound: 43.5581367
IS_B1_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -43.4700800, upper bound: 43.5451905
IS_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -43.5591247, upper bound: 43.5770626
IS_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -43.5569962, upper bound: 43.5769161
IS_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -43.5491357, upper bound: 43.5539200
IS_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -43.5539466, upper bound: 43.5539200
IS_B1_A2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -43.5604672, upper bound: 43.5721456
IS_B1_A2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -43.5604672, upper bound: 43.5721456
IS_B1_A2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -43.5555644, upper bound: 43.5259161
IS_B1_A2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -43.5472828, upper bound: 43.4704082
IS_B1_A2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -43.5223952, upper bound: 43.5670800
IS_B1_A2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -43.5223952, upper bound: 43.5670800
IS_B1_A2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -43.5176817, upper bound: 43.5208505
IS_B1_A2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -43.5094192, upper bound: 43.4653523
IS_B1_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -43.5692031, upper bound: 43.5586702
IS_B1_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -43.5684137, upper bound: 43.5547250
IS_B1_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -43.5692031, upper bound: 43.5586702
IS_B1_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -43.5684137, upper bound: 43.5547250
IS_B1_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -43.5189648, upper bound: 43.5531641
IS_B1_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -43.5165910, upper bound: 43.5189613
IS_B1_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -43.5434515, upper bound: 43.5513966
IS_B1_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -43.5434515, upper bound: 43.5513966
IS_B2_A1_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -43.5721456, upper bound: 43.5604672
IS_B2_A1_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -43.5721456, upper bound: 43.5604672
IS_B2_A1_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -43.5259161, upper bound: 43.5555644
IS_B2_A1_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -43.4704082, upper bound: 43.5472828
IS_B2_A1_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -43.5586702, upper bound: 43.5692031
IS_B2_A1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -43.5547250, upper bound: 43.5684137
IS_B2_A1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -43.5586702, upper bound: 43.5692031
IS_B2_A1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -43.5547250, upper bound: 43.5684137
IS_B2_A1_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -43.5670800, upper bound: 43.5223952
IS_B2_A1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -43.5670800, upper bound: 43.5223952
IS_B2_A1_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -43.5208505, upper bound: 43.5176817
IS_B2_A1_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -43.4653523, upper bound: 43.5094192
IS_B2_A1_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -43.5531641, upper bound: 43.5189648
IS_B2_A1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -43.5189613, upper bound: 43.5165910
IS_B2_A1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -43.5513966, upper bound: 43.5434515
IS_B2_A1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -43.5513966, upper bound: 43.5434515
IS_B2_A2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -43.4196544, upper bound: 43.4523578
IS_B2_A2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -43.4303320, upper bound: 43.4760308
IS_B2_A2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -43.4277048, upper bound: 43.4514723
IS_B2_A2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -43.4256336, upper bound: 43.4678438
IS_B2_A2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -43.3818490, upper bound: 43.3840848
IS_B2_A2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -43.3818490, upper bound: 43.3840848
IS_B2_A2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -43.3884527, upper bound: 43.4359418
IS_B2_A2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 3, lower bound: -43.4273030, upper bound: 43.4511965

## BFS IS instance: IS_B1_A1_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -4.9090037, 18.9296074, -4.9090037, 18.9296074, -23.8386078, 23.8386078
1: -5.6224365, 21.5094490, -5.6224365, 21.5094490, -27.1318855, 27.1318855
2: -6.0714874, 21.3573895, -6.0714874, 21.3573895, -27.4288769, 27.4288769
3: -9.0102825, 21.9570026, -9.0102825, 21.9570026, -30.9672852, 30.9672852
4: -9.8312683, 20.8039608, -9.8312683, 20.8039608, -30.6352291, 30.6352291

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B1_A1_A1_B1_B1

### Relational analysis result of IS_B1_A1_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5704387, upper bound: 43.5364967
time: 0.51 seconds

## Relational analysis of IS_B1_A1_B1_A1_A1_B1_B2

### Relational analysis result of IS_B1_A1_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5629175, upper bound: 43.4742198
time: 0.50 seconds

## BFS IS instance: IS_B1_A1_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -4.9090037, 18.9296074, -5.4008245, 20.7929802, -25.7019825, 24.3304329
1: -5.6224365, 21.5094490, -6.2052450, 23.6457863, -29.2682228, 27.7146950
2: -6.0714874, 21.3573895, -6.6841125, 23.4871082, -29.5585957, 28.0415020
3: -9.0102825, 21.9570026, -9.9212875, 24.1726971, -33.1829796, 31.8782902
4: -9.8312683, 20.8039608, -10.8100157, 22.8875217, -32.7187881, 31.6139755

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A1_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B1_A1_A1_B2_B1

### Relational analysis result of IS_B1_A1_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5704387, upper bound: 43.5364967
time: 0.55 seconds

## Relational analysis of IS_B1_A1_B1_A1_A1_B2_B2

### Relational analysis result of IS_B1_A1_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5629175, upper bound: 43.4742198
time: 0.49 seconds

## BFS IS instance: IS_B1_A1_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -5.4008245, 20.7929802, -4.9090037, 18.9296074, -24.3304329, 25.7019825
1: -6.2052450, 23.6457863, -5.6224365, 21.5094490, -27.7146950, 29.2682228
2: -6.6841125, 23.4871082, -6.0714874, 21.3573895, -28.0415020, 29.5585957
3: -9.9212875, 24.1726971, -9.0102825, 21.9570026, -31.8782902, 33.1829796
4: -10.8100157, 22.8875217, -9.8312683, 20.8039608, -31.6139755, 32.7187843

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B1_A1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B1_A1_A2_B1_A1

### Relational analysis result of IS_B1_A1_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5235106, upper bound: 43.4688450
time: 0.51 seconds

## Relational analysis of IS_B1_A1_B1_A1_A2_B1_A2

### Relational analysis result of IS_B1_A1_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4613239, upper bound: 43.4613239
time: 0.52 seconds

## BFS IS instance: IS_B1_A1_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -5.4008245, 20.7929802, -5.4008245, 20.7929802, -26.1938057, 26.1938057
1: -6.2052450, 23.6457863, -6.2052450, 23.6457863, -29.8510323, 29.8510323
2: -6.6841125, 23.4871082, -6.6841125, 23.4871082, -30.1712208, 30.1712208
3: -9.9212875, 24.1726971, -9.9212875, 24.1726971, -34.0939865, 34.0939865
4: -10.8100157, 22.8875217, -10.8100157, 22.8875217, -33.6975365, 33.6975365

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B1_A1_A2_B2_A1

### Relational analysis result of IS_B1_A1_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5235106, upper bound: 43.4688450
time: 0.52 seconds

## Relational analysis of IS_B1_A1_B1_A1_A2_B2_A2

### Relational analysis result of IS_B1_A1_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4613239, upper bound: 43.4613239
time: 0.53 seconds

## BFS IS instance: IS_B1_A1_B1_A2_A1_A1

### Backsubstitution after applying IS history:
0: -6.0885549, 23.2257309, -5.2551436, 20.1315994, -26.2201519, 28.4808731
1: -7.0463910, 26.3764725, -6.0467672, 22.8425674, -29.8889580, 32.4232330
2: -7.5475416, 26.3148308, -6.4991312, 22.7556610, -30.3032017, 32.8139610
3: -11.1884851, 27.0328350, -9.6570997, 23.3396664, -34.5281525, 36.6899338
4: -12.0433311, 25.7938061, -10.4297295, 22.2815781, -34.3249016, 36.2235336

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B1_A1_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B1_A2_A1_A1_B1

### Relational analysis result of IS_B1_A1_B1_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5597501, upper bound: 43.4726629
time: 0.80 seconds

## Relational analysis of IS_B1_A1_B1_A2_A1_A1_B2

### Relational analysis result of IS_B1_A1_B1_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5597501, upper bound: 43.4726629
time: 0.54 seconds

## BFS IS instance: IS_B1_A1_B1_A2_A1_A2

### Backsubstitution after applying IS history:
0: -6.2608924, 23.8898964, -5.2551436, 20.1315994, -26.3924904, 29.1450386
1: -7.2641339, 27.1352291, -6.0467672, 22.8425674, -30.1067009, 33.1819954
2: -7.7652574, 27.0853271, -6.4991312, 22.7556610, -30.5209179, 33.5844574
3: -11.5361423, 27.8390503, -9.6570997, 23.3396664, -34.8758049, 37.4961472
4: -12.4052420, 26.5429840, -10.4297295, 22.2815781, -34.6868172, 36.9727020

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B1_A1_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B1_A2_A1_A2_B1

### Relational analysis result of IS_B1_A1_B1_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5491357, upper bound: 43.4708694
time: 0.54 seconds

## Relational analysis of IS_B1_A1_B1_A2_A1_A2_B2

### Relational analysis result of IS_B1_A1_B1_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5491357, upper bound: 43.4708694
time: 0.51 seconds

## BFS IS instance: IS_B1_A1_B1_A2_A2_A1

### Backsubstitution after applying IS history:
0: -6.6963181, 25.4623470, -5.2551436, 20.1315994, -26.8279171, 30.7174911
1: -7.7666783, 28.9356041, -6.0467672, 22.8425674, -30.6092415, 34.9823723
2: -8.3003368, 28.8721256, -6.4991312, 22.7556610, -31.0559959, 35.3712502
3: -12.3004551, 29.6954193, -9.6570997, 23.3396664, -35.6401215, 39.3525200
4: -13.2228851, 28.2979012, -10.4297295, 22.2815781, -35.5044594, 38.7276268

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B1_A1_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B1_A2_A2_A1_B1

### Relational analysis result of IS_B1_A1_B1_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5581367, upper bound: 43.4724648
time: 0.74 seconds

## Relational analysis of IS_B1_A1_B1_A2_A2_A1_B2

### Relational analysis result of IS_B1_A1_B1_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5581367, upper bound: 43.4724648
time: 0.68 seconds

## BFS IS instance: IS_B1_A1_B1_A2_A2_A2

### Backsubstitution after applying IS history:
0: -6.8828583, 26.1740513, -5.2551436, 20.1315994, -27.0144539, 31.4291935
1: -8.0028114, 29.7413483, -6.0467672, 22.8425674, -30.8453770, 35.7881165
2: -8.5354614, 29.6899586, -6.4991312, 22.7556610, -31.2911224, 36.1890907
3: -12.6767759, 30.5600548, -9.6570997, 23.3396664, -36.0164413, 40.2171555
4: -13.6112204, 29.0972939, -10.4297295, 22.2815781, -35.8927994, 39.5270195

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B1_A1_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B1_A2_A2_A2_B1

### Relational analysis result of IS_B1_A1_B1_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5451905, upper bound: 43.4700800
time: 0.57 seconds

## Relational analysis of IS_B1_A1_B1_A2_A2_A2_B2

### Relational analysis result of IS_B1_A1_B1_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5451905, upper bound: 43.4700800
time: 0.61 seconds

## BFS IS instance: IS_B1_A1_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -5.2551436, 20.1315994, -6.0885549, 23.2257309, -28.4808731, 26.2201519
1: -6.0467672, 22.8425674, -7.0463910, 26.3764725, -32.4232368, 29.8889580
2: -6.4991312, 22.7556610, -7.5475416, 26.3148308, -32.8139534, 30.3031998
3: -9.6570997, 23.3396664, -11.1884851, 27.0328350, -36.6899338, 34.5281525
4: -10.4297295, 22.2815781, -12.0433311, 25.7938061, -36.2235336, 34.3249016

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A1_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B2_A1_B1_B1_A1

### Relational analysis result of IS_B1_A1_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4726629, upper bound: 43.5597501
time: 0.49 seconds

## Relational analysis of IS_B1_A1_B2_A1_B1_B1_A2

### Relational analysis result of IS_B1_A1_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4726629, upper bound: 43.5597501
time: 0.56 seconds

## BFS IS instance: IS_B1_A1_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -5.2551436, 20.1315994, -6.2608924, 23.8898964, -29.1450386, 26.3924904
1: -6.0467672, 22.8425674, -7.2641339, 27.1352291, -33.1819954, 30.1067009
2: -6.4991312, 22.7556610, -7.7652574, 27.0853271, -33.5844574, 30.5209179
3: -9.6570997, 23.3396664, -11.5361423, 27.8390503, -37.4961472, 34.8758049
4: -10.4297295, 22.2815781, -12.4052420, 26.5429840, -36.9727020, 34.6868210

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A1_B2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B2_A1_B1_B2_A1

### Relational analysis result of IS_B1_A1_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4708694, upper bound: 43.5491357
time: 0.55 seconds

## Relational analysis of IS_B1_A1_B2_A1_B1_B2_A2

### Relational analysis result of IS_B1_A1_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4708694, upper bound: 43.5491357
time: 0.53 seconds

## BFS IS instance: IS_B1_A1_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -5.2551436, 20.1315994, -6.6963181, 25.4623470, -30.7174911, 26.8279171
1: -6.0467672, 22.8425674, -7.7666783, 28.9356041, -34.9823723, 30.6092415
2: -6.4991312, 22.7556610, -8.3003368, 28.8721256, -35.3712502, 31.0559959
3: -9.6570997, 23.3396664, -12.3004551, 29.6954193, -39.3525200, 35.6401215
4: -10.4297295, 22.2815781, -13.2228851, 28.2979012, -38.7276268, 35.5044594

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A1_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B2_A1_B2_B1_A1

### Relational analysis result of IS_B1_A1_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4724648, upper bound: 43.5581367
time: 0.50 seconds

## Relational analysis of IS_B1_A1_B2_A1_B2_B1_A2

### Relational analysis result of IS_B1_A1_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4724648, upper bound: 43.5581367
time: 0.52 seconds

## BFS IS instance: IS_B1_A1_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -5.2551436, 20.1315994, -6.8828583, 26.1740513, -31.4291954, 27.0144539
1: -6.0467672, 22.8425674, -8.0028114, 29.7413483, -35.7881165, 30.8453770
2: -6.4991312, 22.7556610, -8.5354614, 29.6899586, -36.1890907, 31.2911224
3: -9.6570997, 23.3396664, -12.6767759, 30.5600548, -40.2171555, 36.0164413
4: -10.4297295, 22.2815781, -13.6112204, 29.0972939, -39.5270195, 35.8927994

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A1_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B2_A1_B2_B2_A1

### Relational analysis result of IS_B1_A1_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4700800, upper bound: 43.5451905
time: 0.57 seconds

## Relational analysis of IS_B1_A1_B2_A1_B2_B2_A2

### Relational analysis result of IS_B1_A1_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4700800, upper bound: 43.5451905
time: 0.71 seconds

## BFS IS instance: IS_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6.3014240, 23.9654980, -6.5283041, 24.7212219, -31.0226460, 30.4938011
1: -7.3049097, 27.2115173, -7.5771909, 28.0262680, -35.3311729, 34.7887077
2: -7.8123007, 27.1727753, -8.0923138, 28.0525475, -35.8648491, 35.2650909
3: -11.5818949, 27.9037056, -11.9928169, 28.7598476, -40.3417358, 39.8965225
4: -12.4305439, 26.6641655, -12.8025274, 27.6119270, -40.0424690, 39.4666901

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5569962, upper bound: 43.5769161
time: 0.58 seconds

## Relational analysis of IS_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5569962, upper bound: 43.5769161
time: 0.56 seconds

## BFS IS instance: IS_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -6.9152741, 26.2212715, -6.5283041, 24.7212219, -31.6364956, 32.7495766
1: -8.0347805, 29.7921047, -7.5771909, 28.0262680, -36.0610504, 37.3692970
2: -8.5723772, 29.7500801, -8.0923138, 28.0525475, -36.6249237, 37.8423920
3: -12.7084513, 30.5911541, -11.9928169, 28.7598476, -41.4682922, 42.5839691
4: -13.6220407, 29.1891994, -12.8025274, 27.6119270, -41.2339554, 41.9917259

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5569962, upper bound: 43.5769161
time: 0.64 seconds

## Relational analysis of IS_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5569962, upper bound: 43.5769161
time: 0.65 seconds

## BFS IS instance: IS_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6.3014240, 23.9654980, -6.7093639, 25.4152355, -31.7166576, 30.6748619
1: -7.3049097, 27.2115173, -7.8063669, 28.8139172, -36.1188240, 35.0178719
2: -7.8123007, 27.1727753, -8.3207026, 28.8522396, -36.6645393, 35.4934769
3: -11.5818949, 27.9037056, -12.3585405, 29.6142330, -41.1961288, 40.2622452
4: -12.4305439, 26.6641655, -13.1818914, 28.3913708, -40.8219147, 39.8460503

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B2_A2_B2_A1_A1

### Relational analysis result of IS_B1_A1_B2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5549699, upper bound: 43.5539200
time: 0.61 seconds

## Relational analysis of IS_B1_A1_B2_A2_B2_A1_A2

### Relational analysis result of IS_B1_A1_B2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5549699, upper bound: 43.5539200
time: 0.96 seconds

## BFS IS instance: IS_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6.9152741, 26.2212715, -6.7093639, 25.4152355, -32.3305092, 32.9306335
1: -8.0347805, 29.7921047, -7.8063669, 28.8139172, -36.8486977, 37.5984688
2: -8.5723772, 29.7500801, -8.3207026, 28.8522396, -37.4246178, 38.0707817
3: -12.7084513, 30.5911541, -12.3585405, 29.6142330, -42.3226852, 42.9496956
4: -13.6220407, 29.1891994, -13.1818914, 28.3913708, -42.0134125, 42.3710861

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B2_A2_B2_A2_A1

### Relational analysis result of IS_B1_A1_B2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5539466, upper bound: 43.5539200
time: 0.75 seconds

## Relational analysis of IS_B1_A1_B2_A2_B2_A2_A2

### Relational analysis result of IS_B1_A1_B2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5539466, upper bound: 43.5539200
time: 0.63 seconds

## BFS IS instance: IS_B1_A2_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -8.6322088, 32.3104172, -4.8395281, 18.6394978, -27.2717056, 37.1499443
1: -10.1549702, 36.7054749, -5.5451407, 21.1490383, -31.3040047, 42.2506142
2: -10.6975584, 36.7274017, -5.9836831, 21.0349770, -31.7325344, 42.7110863
3: -15.9301043, 37.9201851, -8.8850317, 21.5857143, -37.5158195, 46.8052177
4: -16.8876266, 36.1860275, -9.6547794, 20.5678463, -37.4554749, 45.8408012

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_B1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_B1_A2_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_A1_B1_B1_B1

### Relational analysis result of IS_B1_A2_B1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5602780, upper bound: 43.5720115
time: 0.83 seconds

## Relational analysis of IS_B1_A2_B1_A1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A2_B1_A1_B1_B1_A1

### Relational analysis result of IS_B1_A2_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5604672, upper bound: 43.5721456
time: 0.52 seconds

## Relational analysis of IS_B1_A2_B1_A1_B1_B1_A2

### Relational analysis result of IS_B1_A2_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5604672, upper bound: 43.5721456
time: 0.54 seconds

## BFS IS instance: IS_B1_A2_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -8.6322088, 32.3104172, -5.0165229, 19.3200130, -27.9522209, 37.3269348
1: -10.1549702, 36.7054749, -5.7698326, 21.9297009, -32.0846634, 42.4753075
2: -10.6975584, 36.7274017, -6.2069774, 21.8248901, -32.5224457, 42.9343796
3: -15.9301043, 37.9201851, -9.2441177, 22.4114609, -38.3415642, 47.1643028
4: -16.8876266, 36.1860275, -10.0225515, 21.3482018, -38.2358246, 46.2085800

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_B1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_B1_A2_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_A1_B1_B2_B1

### Relational analysis result of IS_B1_A2_B1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5602780, upper bound: 43.5720115
time: 0.52 seconds

## Relational analysis of IS_B1_A2_B1_A1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A2_B1_A1_B1_B2_A1

### Relational analysis result of IS_B1_A2_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5604672, upper bound: 43.5721456
time: 0.57 seconds

## Relational analysis of IS_B1_A2_B1_A1_B1_B2_A2

### Relational analysis result of IS_B1_A2_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5604672, upper bound: 43.5721456
time: 0.76 seconds

## BFS IS instance: IS_B1_A2_B1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -8.6322088, 32.3104172, -5.2180510, 19.8573036, -28.4895134, 37.5284653
1: -10.1549702, 36.7054749, -6.0158553, 22.5195217, -32.6744919, 42.7213287
2: -10.6975584, 36.7274017, -6.4501691, 22.4596786, -33.1572380, 43.1775703
3: -15.9301043, 37.9201851, -9.5833883, 23.0195580, -38.9496613, 47.5035706
4: -16.8876266, 36.1860275, -10.3194571, 22.0301399, -38.9177628, 46.5054855

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_B1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_B1_A2_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_B1_A1_B2_B1_B1

### Relational analysis result of IS_B1_A2_B1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5555644, upper bound: 43.5259161
time: 0.50 seconds

## Relational analysis of IS_B1_A2_B1_A1_B2_B1_B2

### Relational analysis result of IS_B1_A2_B1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5555644, upper bound: 43.5259161
time: 0.55 seconds

## BFS IS instance: IS_B1_A2_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -8.6322088, 32.3104172, -5.6673207, 21.5628738, -30.1950836, 37.9777374
1: -10.1549702, 36.7054749, -6.5436316, 24.4766960, -34.6316681, 43.2491074
2: -10.6975584, 36.7274017, -7.0108037, 24.4052582, -35.1028175, 43.7382050
3: -15.9301043, 37.9201851, -10.4097338, 25.0520611, -40.9821663, 48.3299179
4: -16.8876266, 36.1860275, -11.2162266, 23.9139538, -40.8015785, 47.4022446

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_B1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_B1_A1_B2_B2_B1

### Relational analysis result of IS_B1_A2_B1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5472828, upper bound: 43.4704082
time: 0.53 seconds

## Relational analysis of IS_B1_A2_B1_A1_B2_B2_B2

### Relational analysis result of IS_B1_A2_B1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5472828, upper bound: 43.4704082
time: 0.55 seconds

## BFS IS instance: IS_B1_A2_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -8.8174801, 33.0339127, -4.8395281, 18.6394978, -27.4569740, 37.8734398
1: -10.3915501, 37.5448494, -5.5451407, 21.1490383, -31.5405846, 43.0899887
2: -10.9315701, 37.5578690, -5.9836831, 21.0349770, -31.9665470, 43.5415535
3: -16.3067951, 38.8084564, -8.8850317, 21.5857143, -37.8925095, 47.6934891
4: -17.2786331, 36.9893837, -9.6547794, 20.5678463, -37.8464813, 46.6441650

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_B1_A2_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_A2_B1_B1_B1

### Relational analysis result of IS_B1_A2_B1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5223952, upper bound: 43.5669459
time: 0.67 seconds

## Relational analysis of IS_B1_A2_B1_A2_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A2_B1_A2_B1_B1_A1

### Relational analysis result of IS_B1_A2_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5223952, upper bound: 43.5670800
time: 0.62 seconds

## Relational analysis of IS_B1_A2_B1_A2_B1_B1_A2

### Relational analysis result of IS_B1_A2_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5223952, upper bound: 43.5670800
time: 0.53 seconds

## BFS IS instance: IS_B1_A2_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -8.8174801, 33.0339127, -5.0165229, 19.3200130, -28.1374893, 38.0504303
1: -10.3915501, 37.5448494, -5.7698326, 21.9297009, -32.3212433, 43.3146820
2: -10.9315701, 37.5578690, -6.2069774, 21.8248901, -32.7564621, 43.7648468
3: -16.3067951, 38.8084564, -9.2441177, 22.4114609, -38.7182541, 48.0525742
4: -17.2786331, 36.9893837, -10.0225515, 21.3482018, -38.6268311, 47.0119362

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_A2_B1_B2_B1

### Relational analysis result of IS_B1_A2_B1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5223952, upper bound: 43.5669459
time: 0.55 seconds

## Relational analysis of IS_B1_A2_B1_A2_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_B1_A2_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A2_B1_A2_B1_B2_A1

### Relational analysis result of IS_B1_A2_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5223952, upper bound: 43.5670800
time: 0.56 seconds

## Relational analysis of IS_B1_A2_B1_A2_B1_B2_A2

### Relational analysis result of IS_B1_A2_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5223952, upper bound: 43.5670800
time: 0.54 seconds

## BFS IS instance: IS_B1_A2_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -8.8174801, 33.0339127, -5.2180510, 19.8573036, -28.6747799, 38.2519608
1: -10.3915501, 37.5448494, -6.0158553, 22.5195217, -32.9110718, 43.5607033
2: -10.9315701, 37.5578690, -6.4501691, 22.4596786, -33.3912506, 44.0080376
3: -16.3067951, 38.8084564, -9.5833883, 23.0195580, -39.3263550, 48.3918419
4: -17.2786331, 36.9893837, -10.3194571, 22.0301399, -39.3087692, 47.3088417

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_B1_A2_B2_B1_B1

### Relational analysis result of IS_B1_A2_B1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5176817, upper bound: 43.5208505
time: 0.54 seconds

## Relational analysis of IS_B1_A2_B1_A2_B2_B1_B2

### Relational analysis result of IS_B1_A2_B1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5176817, upper bound: 43.5208505
time: 0.54 seconds

## BFS IS instance: IS_B1_A2_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -8.8174801, 33.0339127, -5.6673207, 21.5628738, -30.3803501, 38.7012329
1: -10.3915501, 37.5448494, -6.5436316, 24.4766960, -34.8682480, 44.0884819
2: -10.9315701, 37.5578690, -7.0108037, 24.4052582, -35.3368301, 44.5686722
3: -16.3067951, 38.8084564, -10.4097338, 25.0520611, -41.3588562, 49.2181892
4: -17.2786331, 36.9893837, -11.2162266, 23.9139538, -41.1925888, 48.2056122

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_B1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A2_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_B1_A2_B2_B2_B1

### Relational analysis result of IS_B1_A2_B1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5094192, upper bound: 43.4653523
time: 0.63 seconds

## Relational analysis of IS_B1_A2_B1_A2_B2_B2_B2

### Relational analysis result of IS_B1_A2_B1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5094192, upper bound: 43.4653523
time: 0.52 seconds

## BFS IS instance: IS_B1_A2_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -8.6322088, 32.3104172, -6.0885549, 23.2257309, -31.8579407, 38.3989639
1: -10.1549702, 36.7054749, -7.0463910, 26.3764725, -36.5314331, 43.7518654
2: -10.6975584, 36.7274017, -7.5475416, 26.3148308, -37.0123863, 44.2749405
3: -15.9301043, 37.9201851, -11.1884851, 27.0328350, -42.9629402, 49.1086693
4: -16.8876266, 36.1860275, -12.0433311, 25.7938061, -42.6814346, 48.2293510

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_B2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A2_B2_A1_B1_B1_B1

### Relational analysis result of IS_B1_A2_B2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5593660, upper bound: 43.5792777
time: 0.56 seconds

## Relational analysis of IS_B1_A2_B2_A1_B1_B1_B2

### Relational analysis result of IS_B1_A2_B2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5579472, upper bound: 43.5585932
time: 0.54 seconds

## BFS IS instance: IS_B1_A2_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -8.6322088, 32.3104172, -6.6963181, 25.4623470, -34.0945511, 39.0067329
1: -10.1549702, 36.7054749, -7.7666783, 28.9356041, -39.0905724, 44.4721527
2: -10.6975584, 36.7274017, -8.3003368, 28.8721256, -39.5696754, 45.0277405
3: -15.9301043, 37.9201851, -12.3004551, 29.6954193, -45.6255226, 50.2206421
4: -16.8876266, 36.1860275, -13.2228851, 28.2979012, -45.1855278, 49.4089088

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_B2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A2_B2_A1_B1_B2_B1

### Relational analysis result of IS_B1_A2_B2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5590796, upper bound: 43.5763264
time: 0.55 seconds

## Relational analysis of IS_B1_A2_B2_A1_B1_B2_B2

### Relational analysis result of IS_B1_A2_B2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5579165, upper bound: 43.5585180
time: 0.58 seconds

## BFS IS instance: IS_B1_A2_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -8.6322088, 32.3104172, -6.2608924, 23.8898964, -32.5221062, 38.5713043
1: -10.1549702, 36.7054749, -7.2641339, 27.1352291, -37.2901993, 43.9696083
2: -10.6975584, 36.7274017, -7.7652574, 27.0853271, -37.7828827, 44.4926605
3: -15.9301043, 37.9201851, -11.5361423, 27.8390503, -43.7691536, 49.4563217
4: -16.8876266, 36.1860275, -12.4052420, 26.5429840, -43.4306068, 48.5912704

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A2_B2_A1_B2_B1_B1

### Relational analysis result of IS_B1_A2_B2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5568475, upper bound: 43.5582296
time: 0.74 seconds

## Relational analysis of IS_B1_A2_B2_A1_B2_B1_B2

### Relational analysis result of IS_B1_A2_B2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5544738, upper bound: 43.5240269
time: 0.73 seconds

## BFS IS instance: IS_B1_A2_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -8.6322088, 32.3104172, -6.8828583, 26.1740513, -34.8062515, 39.1932716
1: -10.1549702, 36.7054749, -8.0028114, 29.7413483, -39.8963165, 44.7082863
2: -10.6975584, 36.7274017, -8.5354614, 29.6899586, -40.3875160, 45.2628632
3: -15.9301043, 37.9201851, -12.6767759, 30.5600548, -46.4901581, 50.5969620
4: -16.8876266, 36.1860275, -13.6112204, 29.0972939, -45.9849205, 49.7972488

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_B2_A1_B2_B2_B1

### Relational analysis result of IS_B1_A2_B2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5604260, upper bound: 43.5525333
time: 0.56 seconds

## Relational analysis of IS_B1_A2_B2_A1_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A2_B2_A1_B2_B2_B1

### Relational analysis result of IS_B1_A2_B2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5560581, upper bound: 43.5542844
time: 0.58 seconds

## Relational analysis of IS_B1_A2_B2_A1_B2_B2_B2

### Relational analysis result of IS_B1_A2_B2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5542586, upper bound: 43.5234224
time: 0.56 seconds

## BFS IS instance: IS_B1_A2_B2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -8.8174801, 33.0339127, -6.0426178, 23.0335770, -31.8510571, 39.0765305
1: -10.3915501, 37.5448494, -6.9947534, 26.1434937, -36.5350418, 44.5396042
2: -10.9315701, 37.5578690, -7.4914846, 26.1120586, -37.0436249, 45.0493546
3: -16.3067951, 38.8084564, -11.1059456, 26.8007526, -43.1075478, 49.9144020
4: -17.2786331, 36.9893837, -11.9460468, 25.6156979, -42.8943291, 48.9354324

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_B2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_B2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_B2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_B2_A2_B1_B1_B1

### Relational analysis result of IS_B1_A2_B2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5189648, upper bound: 43.5531641
time: 0.62 seconds

## Relational analysis of IS_B1_A2_B2_A2_B1_B1_B2

### Relational analysis result of IS_B1_A2_B2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5189648, upper bound: 43.5531641
time: 0.69 seconds

## BFS IS instance: IS_B1_A2_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -8.8174801, 33.0339127, -6.5792227, 24.7873821, -33.6048622, 39.6131325
1: -10.3915501, 37.5448494, -7.6677504, 28.1005020, -38.4920502, 45.2126007
2: -10.9315701, 37.5578690, -8.1529112, 28.1633129, -39.0948830, 45.7107811
3: -16.3067951, 38.8084564, -12.1156969, 28.8535728, -45.1603661, 50.9241486
4: -17.2786331, 36.9893837, -12.8663216, 27.7725067, -45.0511398, 49.8557014

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=46.890357971191406
rel_dist={3: [-43.60075692157075, 43.60075692157075]}

## Binary search (step 1) starts
Candidate diff: 0.0312500


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5742888, upper bound: 43.5900861
time: 0.53 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
time: 0.57 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.28 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 1.28
Output dim: 3, lower bound: -43.5742888, upper bound: 43.5900861
IS_B2, status: Status.UNKNOWN, split count: 1, time: 1.28
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -7.6606002, 28.3044319, -6.4284191, 24.1388493, -31.7994499, 34.7328491
1: -8.9601765, 31.9744511, -7.4645872, 27.3031864, -36.2633629, 39.4390373
2: -9.4893942, 32.1866379, -7.9618182, 27.3991623, -36.8885536, 40.1484489
3: -14.0368652, 32.8534927, -11.7890568, 27.9809723, -42.0178375, 44.6425476
4: -14.5853920, 32.1437492, -12.4559155, 27.1424770, -41.7278671, 44.5996590

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
time: 0.84 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
time: 0.54 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -7.6146083, 28.1353722, -8.8551903, 33.0941162, -40.7087250, 36.9905624
1: -8.9045057, 31.7804298, -10.4291992, 37.5969582, -46.5014648, 42.2096176
2: -9.4335175, 31.9929218, -10.9747181, 37.6314697, -47.0649872, 42.9676399
3: -13.9499683, 32.6496849, -16.3464375, 38.8504906, -52.8004417, 48.9961243
4: -14.4945126, 31.9539337, -17.2999840, 37.0951767, -51.5896873, 49.2539177

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
time: 0.80 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
time: 0.75 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.68 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 3.68
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 3.68
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 3.68
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 3.68
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -6.4284191, 24.1388493, -6.4284191, 24.1388493, -30.5672684, 30.5672684
1: -7.4645872, 27.3031864, -7.4645872, 27.3031864, -34.7677727, 34.7677727
2: -7.9618182, 27.3991623, -7.9618182, 27.3991623, -35.3609810, 35.3609810
3: -11.7890568, 27.9809723, -11.7890568, 27.9809723, -39.7700233, 39.7700272
4: -12.4559155, 27.1424770, -12.4559155, 27.1424770, -39.5983925, 39.5983925

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B1_A1_B1

### Relational analysis result of IS_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5742888, upper bound: 43.5900861
time: 0.73 seconds

## Relational analysis of IS_B1_A1_B2

### Relational analysis result of IS_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5726843, upper bound: 43.5814188
time: 0.54 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -8.8551903, 33.0941162, -6.4284191, 24.1388493, -32.9940414, 39.5225296
1: -10.4291992, 37.5969582, -7.4645872, 27.3031864, -37.7323837, 45.0615425
2: -10.9747181, 37.6314697, -7.9618182, 27.3991623, -38.3738785, 45.5932884
3: -16.3464375, 38.8504906, -11.7890568, 27.9809723, -44.3274078, 50.6395416
4: -17.2999840, 37.0951767, -12.4559155, 27.1424770, -44.4424591, 49.5510902

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B1

### Relational analysis result of IS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5742888, upper bound: 43.5900861
time: 0.54 seconds

## Relational analysis of IS_B1_A2_B2

### Relational analysis result of IS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5726843, upper bound: 43.5814188
time: 0.52 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -6.4284191, 24.1388493, -8.8551903, 33.0941162, -39.5225296, 32.9940414
1: -7.4645872, 27.3031864, -10.4291992, 37.5969582, -45.0615425, 37.7323837
2: -7.9618182, 27.3991623, -10.9747181, 37.6314697, -45.5932884, 38.3738785
3: -11.7890568, 27.9809723, -16.3464375, 38.8504906, -50.6395416, 44.3274078
4: -12.4559155, 27.1424770, -17.2999840, 37.0951767, -49.5510902, 44.4424591

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_B1

### Relational analysis result of IS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5442084, upper bound: 43.5652404
time: 0.53 seconds

## Relational analysis of IS_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5409015, upper bound: 43.5409015
time: 0.47 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -8.8551903, 33.0941162, -8.8551903, 33.0941162, -41.9493065, 41.9493065
1: -10.4291992, 37.5969582, -10.4291992, 37.5969582, -48.0261536, 48.0261536
2: -10.9747181, 37.6314697, -10.9747181, 37.6314697, -48.6061859, 48.6061859
3: -16.3464375, 38.8504906, -16.3464375, 38.8504906, -55.1969299, 55.1969299
4: -17.2999840, 37.0951767, -17.2999840, 37.0951767, -54.3951607, 54.3951607

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_A2_A1

### Relational analysis result of IS_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5652404, upper bound: 43.5442084
time: 0.49 seconds

## Relational analysis of IS_B2_A2_A2

### Relational analysis result of IS_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5409015, upper bound: 43.5409015
time: 0.69 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.07 seconds
IS_B1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 4.07
Output dim: 3, lower bound: -43.5742888, upper bound: 43.5900861
IS_B1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 4.07
Output dim: 3, lower bound: -43.5726843, upper bound: 43.5814188
IS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 4.07
Output dim: 3, lower bound: -43.5742888, upper bound: 43.5900861
IS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 4.07
Output dim: 3, lower bound: -43.5726843, upper bound: 43.5814188
IS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 4.07
Output dim: 3, lower bound: -43.5442084, upper bound: 43.5652404
IS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 4.07
Output dim: 3, lower bound: -43.5409015, upper bound: 43.5409015
IS_B2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 4.07
Output dim: 3, lower bound: -43.5652404, upper bound: 43.5442084
IS_B2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 4.07
Output dim: 3, lower bound: -43.5409015, upper bound: 43.5409015

## BFS IS instance: IS_B1_A1_B1

### Backsubstitution after applying IS history:
0: -6.4284191, 24.1388493, -5.2551436, 20.1315994, -26.5600185, 29.3939915
1: -7.4645872, 27.3031864, -6.0467672, 22.8425674, -30.3071537, 33.3499527
2: -7.9618182, 27.3991623, -6.4991312, 22.7556610, -30.7174797, 33.8982925
3: -11.7890568, 27.9809723, -9.6570997, 23.3396664, -35.1287231, 37.6380692
4: -12.4559155, 27.1424770, -10.4297295, 22.2815781, -34.7374878, 37.5722046

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A1_B1_A1

### Relational analysis result of IS_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5839320, upper bound: 43.5839320
time: 0.55 seconds

## Relational analysis of IS_B1_A1_B1_A2

### Relational analysis result of IS_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5839320, upper bound: 43.5839320
time: 0.53 seconds

## BFS IS instance: IS_B1_A1_B2

### Backsubstitution after applying IS history:
0: -6.3839698, 23.9801636, -6.7437096, 25.4687443, -31.8527126, 30.7238731
1: -7.4101686, 27.1250076, -7.8409715, 28.8711681, -36.2813301, 34.9659767
2: -7.9080462, 27.2169266, -8.3597012, 28.9183235, -36.8263664, 35.5766220
3: -11.7067308, 27.7952938, -12.3940144, 29.6448441, -41.3515739, 40.1893082
4: -12.3762817, 26.9616508, -13.1956530, 28.4903889, -40.8666687, 40.1572990

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A1_B2_A1

### Relational analysis result of IS_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5839320, upper bound: 43.5839320
time: 0.53 seconds

## Relational analysis of IS_B1_A1_B2_A2

### Relational analysis result of IS_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5839320, upper bound: 43.5839320
time: 0.52 seconds

## BFS IS instance: IS_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.8551903, 33.0941162, -5.2551436, 20.1315994, -28.9867897, 38.3492584
1: -10.4291992, 37.5969582, -6.0467672, 22.8425674, -33.2717590, 43.6437263
2: -10.9747181, 37.6314697, -6.4991312, 22.7556610, -33.7303772, 44.1306000
3: -16.3464375, 38.8504906, -9.6570997, 23.3396664, -39.6861038, 48.5075874
4: -17.2999840, 37.0951767, -10.4297295, 22.2815781, -39.5815620, 47.5249062

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_B1_A1

### Relational analysis result of IS_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5712842, upper bound: 43.5679481
time: 0.54 seconds

## Relational analysis of IS_B1_A2_B1_A2

### Relational analysis result of IS_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5463916, upper bound: 43.5645889
time: 0.53 seconds

## BFS IS instance: IS_B1_A2_B2

### Backsubstitution after applying IS history:
0: -8.8142185, 32.9435463, -6.7437096, 25.4687443, -34.2829628, 39.6872520
1: -10.3782749, 37.4255791, -7.8409715, 28.8711681, -39.2494431, 45.2665520
2: -10.9250174, 37.4597244, -8.3597012, 28.9183235, -39.8433342, 45.8194237
3: -16.2692432, 38.6729546, -12.3940144, 29.6448441, -45.9140701, 51.0669708
4: -17.2244434, 36.9254417, -13.1956530, 28.4903889, -45.7148323, 50.1210938

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_B2_A1

### Relational analysis result of IS_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5689864, upper bound: 43.5586687
time: 0.45 seconds

## Relational analysis of IS_B1_A2_B2_A2

### Relational analysis result of IS_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5442409, upper bound: 43.5553418
time: 0.49 seconds

## BFS IS instance: IS_B2_A1_B1

### Backsubstitution after applying IS history:
0: -6.4284191, 24.1388493, -8.6322088, 32.3104172, -38.7388306, 32.7710571
1: -7.4645872, 27.3031864, -10.1549702, 36.7054749, -44.1700592, 37.4581566
2: -7.9618182, 27.3991623, -10.6975584, 36.7274017, -44.6892204, 38.0967178
3: -11.7890568, 27.9809723, -15.9301043, 37.9201851, -49.7092400, 43.9110756
4: -12.4559155, 27.1424770, -16.8876266, 36.1860275, -48.6419373, 44.0301056

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_A1_B1_A1

### Relational analysis result of IS_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5679481, upper bound: 43.5712842
time: 0.50 seconds

## Relational analysis of IS_B2_A1_B1_A2

### Relational analysis result of IS_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5586687, upper bound: 43.5689864
time: 0.59 seconds

## BFS IS instance: IS_B2_A1_B2

### Backsubstitution after applying IS history:
0: -6.3767309, 23.9696274, -8.8174801, 33.0339127, -39.4106369, 32.7871094
1: -7.4012656, 27.1143246, -10.3915501, 37.5448494, -44.9461136, 37.5058632
2: -7.8979626, 27.2030087, -10.9315701, 37.5578690, -45.4558296, 38.1345787
3: -11.6956806, 27.7829742, -16.3067951, 38.8084564, -50.5041351, 44.0897675
4: -12.3670588, 26.9406662, -17.2786331, 36.9893837, -49.3564415, 44.2192993

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_A1_B2_A1

### Relational analysis result of IS_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5645889, upper bound: 43.5463916
time: 0.92 seconds

## Relational analysis of IS_B2_A1_B2_A2

### Relational analysis result of IS_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5553418, upper bound: 43.5442409
time: 0.49 seconds

## BFS IS instance: IS_B2_A2_A1

### Backsubstitution after applying IS history:
0: -8.6322088, 32.3104172, -8.8551903, 33.0941162, -41.7263222, 41.1656075
1: -10.1549702, 36.7054749, -10.4291992, 37.5969582, -47.7519264, 47.1346703
2: -10.6975584, 36.7274017, -10.9747181, 37.6314697, -48.3290291, 47.7021179
3: -15.9301043, 37.9201851, -16.3464375, 38.8504906, -54.7805939, 54.2666245
4: -16.8876266, 36.1860275, -17.2999840, 37.0951767, -53.9828033, 53.4860115

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A2_A1_B1

### Relational analysis result of IS_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5409015, upper bound: 43.5409015
time: 0.53 seconds

## Relational analysis of IS_B2_A2_A1_B2

### Relational analysis result of IS_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5409015, upper bound: 43.5409015
time: 0.49 seconds

## BFS IS instance: IS_B2_A2_A2

### Backsubstitution after applying IS history:
0: -8.8174801, 33.0339127, -8.7998943, 32.9144211, -41.7319031, 41.8338089
1: -10.3915501, 37.5448494, -10.3614931, 37.3947411, -47.7862892, 47.9063416
2: -10.9315701, 37.5578690, -10.9064531, 37.4224205, -48.3539886, 48.4643173
3: -16.3067951, 38.8084564, -16.2461872, 38.6390495, -54.9458427, 55.0546417
4: -17.2786331, 36.9893837, -17.2043457, 36.8796883, -54.1583214, 54.1937294

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B2_A2_A2_A1

### Relational analysis result of IS_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5387238, upper bound: 43.5154913
time: 0.54 seconds

## Relational analysis of IS_B2_A2_A2_A2

### Relational analysis result of IS_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476
time: 0.54 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.07 seconds
IS_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 4.07
Output dim: 3, lower bound: -43.5839320, upper bound: 43.5839320
IS_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 4.07
Output dim: 3, lower bound: -43.5839320, upper bound: 43.5839320
IS_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.07
Output dim: 3, lower bound: -43.5839320, upper bound: 43.5839320
IS_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.07
Output dim: 3, lower bound: -43.5839320, upper bound: 43.5839320
IS_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 4.07
Output dim: 3, lower bound: -43.5712842, upper bound: 43.5679481
IS_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 4.07
Output dim: 3, lower bound: -43.5463916, upper bound: 43.5645889
IS_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.07
Output dim: 3, lower bound: -43.5689864, upper bound: 43.5586687
IS_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.07
Output dim: 3, lower bound: -43.5442409, upper bound: 43.5553418
IS_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 4.07
Output dim: 3, lower bound: -43.5679481, upper bound: 43.5712842
IS_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 4.07
Output dim: 3, lower bound: -43.5586687, upper bound: 43.5689864
IS_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.07
Output dim: 3, lower bound: -43.5645889, upper bound: 43.5463916
IS_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.07
Output dim: 3, lower bound: -43.5553418, upper bound: 43.5442409
IS_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.07
Output dim: 3, lower bound: -43.5409015, upper bound: 43.5409015
IS_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.07
Output dim: 3, lower bound: -43.5409015, upper bound: 43.5409015
IS_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 4.07
Output dim: 3, lower bound: -43.5387238, upper bound: 43.5154913
IS_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 4.07
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476

## BFS IS instance: IS_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -5.2551436, 20.1315994, -5.2551436, 20.1315994, -25.3867397, 25.3867416
1: -6.0467672, 22.8425674, -6.0467672, 22.8425674, -28.8893318, 28.8893318
2: -6.4991312, 22.7556610, -6.4991312, 22.7556610, -29.2547913, 29.2547913
3: -9.6570997, 23.3396664, -9.6570997, 23.3396664, -32.9967651, 32.9967651
4: -10.4297295, 22.2815781, -10.4297295, 22.2815781, -32.7113037, 32.7113037

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B1_A1_A1

### Relational analysis result of IS_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5750000, upper bound: 43.5368862
time: 0.50 seconds

## Relational analysis of IS_B1_A1_B1_A1_A2

### Relational analysis result of IS_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5325401, upper bound: 43.5325401
time: 0.52 seconds

## BFS IS instance: IS_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -6.7437096, 25.4687443, -5.2551436, 20.1315994, -26.8753090, 30.7238846
1: -7.8409715, 28.8711681, -6.0467672, 22.8425674, -30.6835365, 34.9179344
2: -8.3597012, 28.9183235, -6.4991312, 22.7556610, -31.1153622, 35.4174538
3: -12.3940144, 29.6448441, -9.6570997, 23.3396664, -35.7336807, 39.3019371
4: -13.1956530, 28.4903889, -10.4297295, 22.2815781, -35.4772301, 38.9201202

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B1_A2_A1

### Relational analysis result of IS_B1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5750000, upper bound: 43.5368862
time: 0.49 seconds

## Relational analysis of IS_B1_A1_B1_A2_A2

### Relational analysis result of IS_B1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5325401, upper bound: 43.5366103
time: 0.51 seconds

## BFS IS instance: IS_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -5.2551436, 20.1315994, -6.7437096, 25.4687443, -30.7238846, 26.8753090
1: -6.0467672, 22.8425674, -7.8409715, 28.8711681, -34.9179344, 30.6835365
2: -6.4991312, 22.7556610, -8.3597012, 28.9183235, -35.4174538, 31.1153622
3: -9.6570997, 23.3396664, -12.3940144, 29.6448441, -39.3019371, 35.7336807
4: -10.4297295, 22.2815781, -13.1956530, 28.4903889, -38.9201202, 35.4772301

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B2_A1_B1

### Relational analysis result of IS_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5368862, upper bound: 43.5750000
time: 0.54 seconds

## Relational analysis of IS_B1_A1_B2_A1_B2

### Relational analysis result of IS_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5325401, upper bound: 43.5735626
time: 0.50 seconds

## BFS IS instance: IS_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -6.7388091, 25.4494114, -6.7437096, 25.4687443, -32.2075462, 32.1931190
1: -7.8323278, 28.8483620, -7.8409715, 28.8711681, -36.7034912, 36.6893234
2: -8.3534727, 28.8956718, -8.3597012, 28.9183235, -37.2717972, 37.2553711
3: -12.3802204, 29.6187992, -12.3940144, 29.6448441, -42.0250549, 42.0128136
4: -13.1852350, 28.4671116, -13.1956530, 28.4903889, -41.6756248, 41.6627655

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B2_A2_B1

### Relational analysis result of IS_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5611379, upper bound: 43.5782773
time: 0.52 seconds

## Relational analysis of IS_B1_A1_B2_A2_B2

### Relational analysis result of IS_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5586812, upper bound: 43.5586812
time: 0.53 seconds

## BFS IS instance: IS_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -8.6322088, 32.3104172, -5.2551436, 20.1315994, -28.7638092, 37.5655594
1: -10.1549702, 36.7054749, -6.0467672, 22.8425674, -32.9975319, 42.7522430
2: -10.6975584, 36.7274017, -6.4991312, 22.7556610, -33.4532204, 43.2265320
3: -15.9301043, 37.9201851, -9.6570997, 23.3396664, -39.2697716, 47.5772858
4: -16.8876266, 36.1860275, -10.4297295, 22.2815781, -39.1692047, 46.6157532

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A2_B1_A1_B1

### Relational analysis result of IS_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5590796, upper bound: 43.5674811
time: 0.54 seconds

## Relational analysis of IS_B1_A2_B1_A1_B2

### Relational analysis result of IS_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5547796, upper bound: 43.5259964
time: 0.79 seconds

## BFS IS instance: IS_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -8.8174801, 33.0339127, -5.2088509, 19.9812698, -28.7987461, 38.2427635
1: -10.3915501, 37.5448494, -5.9901991, 22.6750488, -33.0665894, 43.5350418
2: -10.9315701, 37.5578690, -6.4421506, 22.5805073, -33.5120773, 44.0000191
3: -16.3067951, 38.8084564, -9.5738401, 23.1661835, -39.4729767, 48.3822975
4: -17.2786331, 36.9893837, -10.3513479, 22.0983219, -39.3769531, 47.3407326

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A2_B1_A2_B1

### Relational analysis result of IS_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5210533, upper bound: 43.5612360
time: 0.53 seconds

## Relational analysis of IS_B1_A2_B1_A2_B2

### Relational analysis result of IS_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5173906, upper bound: 43.5213025
time: 0.69 seconds

## BFS IS instance: IS_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -8.5930605, 32.1664772, -6.7437096, 25.4687443, -34.0618019, 38.9101830
1: -10.1064119, 36.5416260, -7.8409715, 28.8711681, -38.9775734, 44.3825951
2: -10.6501856, 36.5633698, -8.3597012, 28.9183235, -39.5685081, 44.9230652
3: -15.8564844, 37.7506409, -12.3940144, 29.6448441, -45.5013237, 50.1446533
4: -16.8155651, 36.0240479, -13.1956530, 28.4903889, -45.3059540, 49.2196999

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_B2_A1_B1

### Relational analysis result of IS_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5442409, upper bound: 43.5553418
time: 0.54 seconds

## Relational analysis of IS_B1_A2_B2_A1_B2

### Relational analysis result of IS_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5442409, upper bound: 43.5553418
time: 0.50 seconds

## BFS IS instance: IS_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -8.7759876, 32.8803558, -6.6918359, 25.3002129, -34.0761986, 39.5721931
1: -10.3398943, 37.3700981, -7.7774014, 28.6834145, -39.0233078, 45.1474991
2: -10.8813286, 37.3827362, -8.2959385, 28.7224560, -39.6037827, 45.6786728
3: -16.2284031, 38.6279373, -12.3002157, 29.4475803, -45.6759834, 50.9281502
4: -17.2025299, 36.8155746, -13.1070518, 28.2879868, -45.4905128, 49.9226265

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B2_A2_B1

### Relational analysis result of IS_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5442409, upper bound: 43.5553418
time: 0.49 seconds

## Relational analysis of IS_B1_A2_B2_A2_B2

### Relational analysis result of IS_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5434515, upper bound: 43.5513966
time: 0.58 seconds

## BFS IS instance: IS_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -5.2551436, 20.1315994, -8.6322088, 32.3104172, -37.5655594, 28.7638092
1: -6.0467672, 22.8425674, -10.1549702, 36.7054749, -42.7522430, 32.9975319
2: -6.4991312, 22.7556610, -10.6975584, 36.7274017, -43.2265320, 33.4532204
3: -9.6570997, 23.3396664, -15.9301043, 37.9201851, -47.5772858, 39.2697716
4: -10.4297295, 22.2815781, -16.8876266, 36.1860275, -46.6157570, 39.1692047

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B2_A1_B1_A1_A1

### Relational analysis result of IS_B2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5674811, upper bound: 43.5590796
time: 0.81 seconds

## Relational analysis of IS_B2_A1_B1_A1_A2

### Relational analysis result of IS_B2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5259964, upper bound: 43.5547796
time: 0.55 seconds

## BFS IS instance: IS_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -6.7437096, 25.4687443, -8.5930605, 32.1664772, -38.9101830, 34.0618019
1: -7.8409715, 28.8711681, -10.1064119, 36.5416260, -44.3825951, 38.9775734
2: -8.3597012, 28.9183235, -10.6501856, 36.5633698, -44.9230652, 39.5685081
3: -12.3940144, 29.6448441, -15.8564844, 37.7506409, -50.1446533, 45.5013237
4: -13.1956530, 28.4903889, -16.8155651, 36.0240479, -49.2196999, 45.3059540

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_B1_A2_A1

### Relational analysis result of IS_B2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5586687, upper bound: 43.5689864
time: 0.51 seconds

## Relational analysis of IS_B2_A1_B1_A2_A2

### Relational analysis result of IS_B2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5586687, upper bound: 43.5689864
time: 1.03 seconds

## BFS IS instance: IS_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -5.2088509, 19.9812698, -8.8174801, 33.0339127, -38.2427635, 28.7987461
1: -5.9901991, 22.6750488, -10.3915501, 37.5448494, -43.5350418, 33.0665894
2: -6.4421506, 22.5805073, -10.9315701, 37.5578690, -44.0000191, 33.5120773
3: -9.5738401, 23.1661835, -16.3067951, 38.8084564, -48.3822975, 39.4729767
4: -10.3513479, 22.0983219, -17.2786331, 36.9893837, -47.3407326, 39.3769531

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B2_A1_B2_A1_A1

### Relational analysis result of IS_B2_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5612360, upper bound: 43.5210533
time: 0.52 seconds

## Relational analysis of IS_B2_A1_B2_A1_A2

### Relational analysis result of IS_B2_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5213025, upper bound: 43.5173906
time: 0.51 seconds

## BFS IS instance: IS_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -6.6878004, 25.2843075, -8.7759876, 32.8803558, -39.5681572, 34.0602913
1: -7.7702889, 28.6646500, -10.3398943, 37.3700981, -45.1403885, 39.0045433
2: -8.2908144, 28.7038212, -10.8813286, 37.3827362, -45.6735497, 39.5851517
3: -12.2888670, 29.4261456, -16.2284031, 38.6279373, -50.9168053, 45.6545486
4: -13.0984793, 28.2688313, -17.2025299, 36.8155746, -49.9140549, 45.4713593

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_A1_B2_A2_A1

### Relational analysis result of IS_B2_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5553418, upper bound: 43.5442409
time: 0.57 seconds

## Relational analysis of IS_B2_A1_B2_A2_A2

### Relational analysis result of IS_B2_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4663064, upper bound: 43.5434515
time: 0.71 seconds

## BFS IS instance: IS_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -8.6322088, 32.3104172, -8.6322088, 32.3104172, -40.9426193, 40.9426193
1: -10.1549702, 36.7054749, -10.1549702, 36.7054749, -46.8604431, 46.8604431
2: -10.6975584, 36.7274017, -10.6975584, 36.7274017, -47.4249573, 47.4249573
3: -15.9301043, 37.9201851, -15.9301043, 37.9201851, -53.8502884, 53.8502884
4: -16.8876266, 36.1860275, -16.8876266, 36.1860275, -53.0736542, 53.0736542

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B2_A2_A1_B1_A1

### Relational analysis result of IS_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5486156, upper bound: 43.5181094
time: 0.59 seconds

## Relational analysis of IS_B2_A2_A1_B1_A2

### Relational analysis result of IS_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5463511, upper bound: 43.5181077
time: 0.79 seconds

## BFS IS instance: IS_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -8.6322088, 32.3104172, -8.8174801, 33.0339127, -41.6661148, 41.1278992
1: -10.1549702, 36.7054749, -10.3915501, 37.5448494, -47.6998177, 47.0970230
2: -10.6975584, 36.7274017, -10.9315701, 37.5578690, -48.2554245, 47.6589737
3: -15.9301043, 37.9201851, -16.3067951, 38.8084564, -54.7385597, 54.2269783
4: -16.8876266, 36.1860275, -17.2786331, 36.9893837, -53.8770103, 53.4646606

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B2_A2_A1_B2_A1

### Relational analysis result of IS_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5486156, upper bound: 43.5181094
time: 1.02 seconds

## Relational analysis of IS_B2_A2_A1_B2_A2

### Relational analysis result of IS_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5463511, upper bound: 43.5181077
time: 0.62 seconds

## BFS IS instance: IS_B2_A2_A2_A1

### Backsubstitution after applying IS history:
0: -8.5439262, 32.0631142, -8.7998943, 32.9144211, -41.4583473, 40.8630066
1: -10.0602074, 36.4324837, -10.3614931, 37.3947411, -47.4549484, 46.7939758
2: -10.5933409, 36.4442482, -10.9064531, 37.4224205, -48.0157623, 47.3506966
3: -15.7987118, 37.6562881, -16.2461872, 38.6390495, -54.4377594, 53.9024734
4: -16.7670708, 35.8889656, -17.2043457, 36.8796883, -53.6467590, 53.0933113

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A2_A2_A1_A1

### Relational analysis result of IS_B2_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4066308, upper bound: 43.4650302
time: 0.59 seconds

## Relational analysis of IS_B2_A2_A2_A1_A2

### Relational analysis result of IS_B2_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5385524, upper bound: 43.5153401
time: 0.64 seconds

## BFS IS instance: IS_B2_A2_A2_A2

### Backsubstitution after applying IS history:
0: -9.1115875, 33.9125404, -8.7687101, 32.8094215, -41.9210091, 42.6812515
1: -10.7727814, 38.5158424, -10.3233290, 37.2768059, -48.0495872, 48.8391724
2: -11.2928276, 38.6079216, -10.8677750, 37.3009148, -48.5937424, 49.4756966
3: -16.8676147, 39.8401985, -16.1885128, 38.5135956, -55.3812103, 56.0287094
4: -17.7444897, 38.1677361, -17.1475468, 36.7557449, -54.5002365, 55.3152847

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B2_A2_A2_A2_B1

### Relational analysis result of IS_B2_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476
time: 0.54 seconds

## Relational analysis of IS_B2_A2_A2_A2_B2

### Relational analysis result of IS_B2_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476
time: 0.58 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 5.12 seconds
IS_B1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -43.5750000, upper bound: 43.5368862
IS_B1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -43.5325401, upper bound: 43.5325401
IS_B1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -43.5750000, upper bound: 43.5368862
IS_B1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -43.5325401, upper bound: 43.5366103
IS_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -43.5368862, upper bound: 43.5750000
IS_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -43.5325401, upper bound: 43.5735626
IS_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -43.5611379, upper bound: 43.5782773
IS_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -43.5586812, upper bound: 43.5586812
IS_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -43.5590796, upper bound: 43.5674811
IS_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -43.5547796, upper bound: 43.5259964
IS_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -43.5210533, upper bound: 43.5612360
IS_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -43.5173906, upper bound: 43.5213025
IS_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -43.5442409, upper bound: 43.5553418
IS_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -43.5442409, upper bound: 43.5553418
IS_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -43.5442409, upper bound: 43.5553418
IS_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -43.5434515, upper bound: 43.5513966
IS_B2_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -43.5674811, upper bound: 43.5590796
IS_B2_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -43.5259964, upper bound: 43.5547796
IS_B2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -43.5586687, upper bound: 43.5689864
IS_B2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -43.5586687, upper bound: 43.5689864
IS_B2_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -43.5612360, upper bound: 43.5210533
IS_B2_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -43.5213025, upper bound: 43.5173906
IS_B2_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -43.5553418, upper bound: 43.5442409
IS_B2_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -43.4663064, upper bound: 43.5434515
IS_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -43.5486156, upper bound: 43.5181094
IS_B2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -43.5463511, upper bound: 43.5181077
IS_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -43.5486156, upper bound: 43.5181094
IS_B2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -43.5463511, upper bound: 43.5181077
IS_B2_A2_A2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -43.4066308, upper bound: 43.4650302
IS_B2_A2_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -43.5385524, upper bound: 43.5153401
IS_B2_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476
IS_B2_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476

## BFS IS instance: IS_B1_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -4.9090037, 18.9296074, -5.2551436, 20.1315994, -25.0405998, 24.1847477
1: -5.6224365, 21.5094490, -6.0467672, 22.8425674, -28.4650040, 27.5562134
2: -6.0714874, 21.3573895, -6.4991312, 22.7556610, -28.8271484, 27.8565197
3: -9.0102825, 21.9570026, -9.6570997, 23.3396664, -32.3499489, 31.6141014
4: -9.8312683, 20.8039608, -10.4297295, 22.2815781, -32.1128387, 31.2336884

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B1_A1_A1_B1

### Relational analysis result of IS_B1_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5325401, upper bound: 43.5325401
time: 0.50 seconds

## Relational analysis of IS_B1_A1_B1_A1_A1_B2

### Relational analysis result of IS_B1_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5325401, upper bound: 43.5325401
time: 0.52 seconds

## BFS IS instance: IS_B1_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -5.4008245, 20.7929802, -5.2252574, 20.0268478, -25.4276733, 26.0182381
1: -6.2052450, 23.6457863, -6.0095611, 22.7262707, -28.9315147, 29.6553478
2: -6.6841125, 23.4871082, -6.4622755, 22.6329479, -29.3170605, 29.9493828
3: -9.9212875, 24.1726971, -9.6003199, 23.2192421, -33.1405296, 33.7730179
4: -10.8100157, 22.8875217, -10.3769379, 22.1532593, -32.9632759, 33.2644501

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B1_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B1_A1_A2_B1

### Relational analysis result of IS_B1_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4670481, upper bound: 43.5060097
time: 0.51 seconds

## Relational analysis of IS_B1_A1_B1_A1_A2_B2

### Relational analysis result of IS_B1_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4611298, upper bound: 43.4611298
time: 0.56 seconds

## BFS IS instance: IS_B1_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -6.3014240, 23.9654980, -5.2551436, 20.1315994, -26.4330235, 29.2206402
1: -7.3049097, 27.2115173, -6.0467672, 22.8425674, -30.1474743, 33.2582817
2: -7.8123007, 27.1727753, -6.4991312, 22.7556610, -30.5679588, 33.6719055
3: -11.5818949, 27.9037056, -9.6570997, 23.3396664, -34.9215622, 37.5608025
4: -12.4305439, 26.6641655, -10.4297295, 22.2815781, -34.7121201, 37.0938835

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B1_A2_A1_A1

### Relational analysis result of IS_B1_A1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5239338, upper bound: 43.4695699
time: 0.51 seconds

## Relational analysis of IS_B1_A1_B1_A2_A1_A2

### Relational analysis result of IS_B1_A1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5278449, upper bound: 43.4697512
time: 0.54 seconds

## BFS IS instance: IS_B1_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -6.9152741, 26.2212715, -5.2252574, 20.0268478, -26.9421215, 31.4465275
1: -8.0347805, 29.7921047, -6.0095611, 22.7262707, -30.7610493, 35.8016663
2: -8.5723772, 29.7500801, -6.4622755, 22.6329479, -31.2053242, 36.2123528
3: -12.7084513, 30.5911541, -9.6003199, 23.2192421, -35.9276924, 40.1914711
4: -13.6220407, 29.1891994, -10.3769379, 22.1532593, -35.7752953, 39.5661316

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B1_A2_A2_A1

### Relational analysis result of IS_B1_A1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5237448, upper bound: 43.4692334
time: 0.69 seconds

## Relational analysis of IS_B1_A1_B1_A2_A2_A2

### Relational analysis result of IS_B1_A1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5260855, upper bound: 43.4692354
time: 0.66 seconds

## BFS IS instance: IS_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -5.2551436, 20.1315994, -6.3014240, 23.9654980, -29.2206402, 26.4330215
1: -6.0467672, 22.8425674, -7.3049097, 27.2115173, -33.2582817, 30.1474743
2: -6.4991312, 22.7556610, -7.8123007, 27.1727753, -33.6719055, 30.5679588
3: -9.6570997, 23.3396664, -11.5818949, 27.9037056, -37.5608063, 34.9215622
4: -10.4297295, 22.2815781, -12.4305439, 26.6641655, -37.0938835, 34.7121201

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B2_A1_B1_B1

### Relational analysis result of IS_B1_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4695699, upper bound: 43.5239338
time: 0.49 seconds

## Relational analysis of IS_B1_A1_B2_A1_B1_B2

### Relational analysis result of IS_B1_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4695699, upper bound: 43.5278449
time: 0.54 seconds

## BFS IS instance: IS_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -5.2252574, 20.0268478, -6.9152741, 26.2212715, -31.4465294, 26.9421215
1: -6.0095611, 22.7262707, -8.0347805, 29.7921047, -35.8016663, 30.7610493
2: -6.4622755, 22.6329479, -8.5723772, 29.7500801, -36.2123528, 31.2053242
3: -9.6003199, 23.2192421, -12.7084513, 30.5911541, -40.1914711, 35.9276886
4: -10.3769379, 22.1532593, -13.6220407, 29.1891994, -39.5661316, 35.7752914

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B2_A1_B2_B1

### Relational analysis result of IS_B1_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4670481, upper bound: 43.5237448
time: 0.88 seconds

## Relational analysis of IS_B1_A1_B2_A1_B2_B2

### Relational analysis result of IS_B1_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4692354, upper bound: 43.5260855
time: 0.56 seconds

## BFS IS instance: IS_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -6.7388091, 25.4494114, -6.5283041, 24.7212219, -31.4600296, 31.9777145
1: -7.8323278, 28.8483620, -7.5771909, 28.0262680, -35.8585968, 36.4255486
2: -8.3534727, 28.8956718, -8.0923138, 28.0525475, -36.4060211, 36.9879837
3: -12.3802204, 29.6187992, -11.9928169, 28.7598476, -41.1400566, 41.6116180
4: -13.1852350, 28.4671116, -12.8025274, 27.6119270, -40.7971497, 41.2696381

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5586812, upper bound: 43.5586812
time: 0.50 seconds

## Relational analysis of IS_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5586812, upper bound: 43.5586812
time: 0.55 seconds

## BFS IS instance: IS_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -6.6878004, 25.2843075, -6.7093639, 25.4152355, -32.1030350, 31.9936714
1: -7.7702889, 28.6646500, -7.8063669, 28.8139172, -36.5842056, 36.4710159
2: -8.2908144, 28.7038212, -8.3207026, 28.8522396, -37.1430550, 37.0245247
3: -12.2888670, 29.4261456, -12.3585405, 29.6142330, -41.9030991, 41.7846870
4: -13.0984793, 28.2688313, -13.1818914, 28.3913708, -41.4898491, 41.4507217

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5549699, upper bound: 43.5539200
time: 0.76 seconds

## Relational analysis of IS_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5539466, upper bound: 43.5539200
time: 0.80 seconds

## BFS IS instance: IS_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -8.6322088, 32.3104172, -5.0394983, 19.3376427, -27.9698524, 37.3499107
1: -10.1549702, 36.7054749, -5.7875409, 21.9354172, -32.0903816, 42.4930115
2: -10.6975584, 36.7274017, -6.2317314, 21.8455486, -32.5431061, 42.9591331
3: -15.9301043, 37.9201851, -9.2545710, 22.4035969, -38.3336983, 47.1747551
4: -16.8876266, 36.1860275, -10.0192795, 21.3928051, -38.2804337, 46.2053032

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_A1_B1_B1

### Relational analysis result of IS_B1_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5589374, upper bound: 43.5674811
time: 0.52 seconds

## Relational analysis of IS_B1_A2_B1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_B1_A1_B1_B1

### Relational analysis result of IS_B1_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5590796, upper bound: 43.5674811
time: 0.48 seconds

## Relational analysis of IS_B1_A2_B1_A1_B1_B2

### Relational analysis result of IS_B1_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5590796, upper bound: 43.5674811
time: 0.53 seconds

## BFS IS instance: IS_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -8.6015329, 32.2070732, -5.5686293, 21.0664139, -29.6679459, 37.7757034
1: -10.1174545, 36.5886230, -6.4469433, 23.8585854, -33.9760399, 43.0355682
2: -10.6595440, 36.6078415, -6.8837023, 23.8715839, -34.5311241, 43.4915428
3: -15.8734093, 37.7966385, -10.2405462, 24.4116497, -40.2850533, 48.0371857
4: -16.8316593, 36.0642014, -10.9216919, 23.5361233, -40.3677788, 46.9858932

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_A1_B2_B1

### Relational analysis result of IS_B1_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5547796, upper bound: 43.5255054
time: 0.58 seconds

## Relational analysis of IS_B1_A2_B1_A1_B2_B2

### Relational analysis result of IS_B1_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5183207, upper bound: 43.4668146
time: 0.73 seconds

## BFS IS instance: IS_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.8174801, 33.0339127, -4.9945722, 19.1920948, -28.0095692, 38.0284843
1: -10.3915501, 37.5448494, -5.7327819, 21.7732925, -32.1648369, 43.2776299
2: -10.9315701, 37.5578690, -6.1764441, 21.6759148, -32.6074829, 43.7343063
3: -16.3067951, 38.8084564, -9.1741276, 22.2358761, -38.5426712, 47.9825821
4: -17.2786331, 36.9893837, -9.9434233, 21.2148380, -38.4934692, 46.9328079

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_A2_B1_B1

### Relational analysis result of IS_B1_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5210533, upper bound: 43.5612360
time: 0.65 seconds

## Relational analysis of IS_B1_A2_B1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_B1_A2_B1_B1

### Relational analysis result of IS_B1_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5210533, upper bound: 43.5609907
time: 0.56 seconds

## Relational analysis of IS_B1_A2_B1_A2_B1_B2

### Relational analysis result of IS_B1_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5210533, upper bound: 43.5609907
time: 0.58 seconds

## BFS IS instance: IS_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -8.7858028, 32.9264526, -5.5207715, 20.9111004, -29.6969032, 38.4472198
1: -10.3527565, 37.4227066, -6.3881083, 23.6853485, -34.0381012, 43.8108139
2: -10.8923359, 37.4335938, -6.8246484, 23.6887436, -34.5810776, 44.2582436
3: -16.2480621, 38.6802444, -10.1537533, 24.2320938, -40.4801521, 48.8339920
4: -17.2209148, 36.8629074, -10.8407536, 23.3468246, -40.5677414, 47.7036591

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_A2_B2_B1

### Relational analysis result of IS_B1_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5173906, upper bound: 43.5208035
time: 0.51 seconds

## Relational analysis of IS_B1_A2_B1_A2_B2_B2

### Relational analysis result of IS_B1_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5094192, upper bound: 43.4653523
time: 0.60 seconds

## BFS IS instance: IS_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -8.5930605, 32.1664772, -6.5283041, 24.7212219, -33.3142815, 38.6947823
1: -10.1064119, 36.5416260, -7.5771909, 28.0262680, -38.1326790, 44.1188164
2: -10.6501856, 36.5633698, -8.0923138, 28.0525475, -38.7027321, 44.6556854
3: -15.8564844, 37.7506409, -11.9928169, 28.7598476, -44.6163254, 49.7434578
4: -16.8155651, 36.0240479, -12.8025274, 27.6119270, -44.4274826, 48.8265762

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B2_A1_B1_B1

### Relational analysis result of IS_B1_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5689864, upper bound: 43.5586687
time: 0.50 seconds

## Relational analysis of IS_B1_A2_B2_A1_B1_B2

### Relational analysis result of IS_B1_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5680844, upper bound: 43.5547191
time: 0.99 seconds

## BFS IS instance: IS_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -8.5930605, 32.1664772, -6.7093639, 25.4152355, -34.0082932, 38.8758392
1: -10.1064119, 36.5416260, -7.8063669, 28.8139172, -38.9203300, 44.3479843
2: -10.6501856, 36.5633698, -8.3207026, 28.8522396, -39.5024261, 44.8840714
3: -15.8564844, 37.7506409, -12.3585405, 29.6142330, -45.4707184, 50.1091805
4: -16.8155651, 36.0240479, -13.1818914, 28.3913708, -45.2069359, 49.2059364

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B2_A1_B2_B1

### Relational analysis result of IS_B1_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5689864, upper bound: 43.5586687
time: 0.52 seconds

## Relational analysis of IS_B1_A2_B2_A1_B2_B2

### Relational analysis result of IS_B1_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5680844, upper bound: 43.5547191
time: 0.56 seconds

## BFS IS instance: IS_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -8.7759876, 32.8803558, -6.2515597, 23.8048096, -32.5807953, 39.1319160
1: -10.3398943, 37.3700981, -7.2439523, 27.0316677, -37.3715591, 44.6140518
2: -10.8813286, 37.3827362, -7.7507277, 26.9848003, -37.8661270, 45.1334648
3: -16.2284031, 38.6279373, -11.4921074, 27.7163181, -43.9447136, 50.1200447
4: -17.2025299, 36.8155746, -12.3457909, 26.4698715, -43.6724014, 49.1613655

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_B2_A2_B1_B1

### Relational analysis result of IS_B1_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5442409, upper bound: 43.5553418
time: 0.54 seconds

## Relational analysis of IS_B1_A2_B2_A2_B1_B2

### Relational analysis result of IS_B1_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5442409, upper bound: 43.5553418
time: 0.50 seconds

## BFS IS instance: IS_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.7504930, 32.7889481, -6.8627620, 26.0500183, -34.8005104, 39.6517067
1: -10.3080959, 37.2676659, -7.9705338, 29.6015415, -39.9096375, 45.2382011
2: -10.8498917, 37.2764397, -8.5076466, 29.5513268, -40.4012146, 45.7840881
3: -16.1796608, 38.5216827, -12.6133423, 30.3921337, -46.5717773, 51.1350250
4: -17.1574078, 36.7058182, -13.5321617, 28.9838638, -46.1412621, 50.2379799

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_B2_A2_B2_B1

### Relational analysis result of IS_B1_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5198541, upper bound: 43.5513966
time: 0.87 seconds

## Relational analysis of IS_B1_A2_B2_A2_B2_B2

### Relational analysis result of IS_B1_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5434515, upper bound: 43.5513966
time: 0.59 seconds

## BFS IS instance: IS_B2_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -5.0394983, 19.3376427, -8.6322088, 32.3104172, -37.3499107, 27.9698524
1: -5.7875409, 21.9354172, -10.1549702, 36.7054749, -42.4930115, 32.0903816
2: -6.2317314, 21.8455486, -10.6975584, 36.7274017, -42.9591331, 32.5431061
3: -9.2545710, 22.4035969, -15.9301043, 37.9201851, -47.1747551, 38.3336983
4: -10.0192795, 21.3928051, -16.8876266, 36.1860275, -46.2053032, 38.2804337

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B2_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_A1_B1_A1_A1_A1

### Relational analysis result of IS_B2_A1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5674811, upper bound: 43.5589374
time: 0.70 seconds

## Relational analysis of IS_B2_A1_B1_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_B1_A1_A1_A1

### Relational analysis result of IS_B2_A1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5674811, upper bound: 43.5590796
time: 0.50 seconds

## Relational analysis of IS_B2_A1_B1_A1_A1_A2

### Relational analysis result of IS_B2_A1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5674811, upper bound: 43.5590796
time: 0.72 seconds

## BFS IS instance: IS_B2_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -5.5686293, 21.0664139, -8.6015329, 32.2070732, -37.7757034, 29.6679459
1: -6.4469433, 23.8585854, -10.1174545, 36.5886230, -43.0355682, 33.9760399
2: -6.8837023, 23.8715839, -10.6595440, 36.6078415, -43.4915390, 34.5311241
3: -10.2405462, 24.4116497, -15.8734093, 37.7966385, -48.0371857, 40.2850533
4: -10.9216919, 23.5361233, -16.8316593, 36.0642014, -46.9858932, 40.3677788

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B2_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_A1_B1_A1_A2_A1

### Relational analysis result of IS_B2_A1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5255054, upper bound: 43.5547796
time: 0.73 seconds

## Relational analysis of IS_B2_A1_B1_A1_A2_A2

### Relational analysis result of IS_B2_A1_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4668146, upper bound: 43.5183207
time: 0.78 seconds

## BFS IS instance: IS_B2_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -6.5283041, 24.7212219, -8.5930605, 32.1664772, -38.6947823, 33.3142815
1: -7.5771909, 28.0262680, -10.1064119, 36.5416260, -44.1188164, 38.1326790
2: -8.0923138, 28.0525475, -10.6501856, 36.5633698, -44.6556854, 38.7027321
3: -11.9928169, 28.7598476, -15.8564844, 37.7506409, -49.7434578, 44.6163292
4: -12.8025274, 27.6119270, -16.8155651, 36.0240479, -48.8265762, 44.4274826

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B2_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_A1_B1_A2_A1_A1

### Relational analysis result of IS_B2_A1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5586687, upper bound: 43.5689864
time: 0.67 seconds

## Relational analysis of IS_B2_A1_B1_A2_A1_A2

### Relational analysis result of IS_B2_A1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5547191, upper bound: 43.5680844
time: 0.61 seconds

## BFS IS instance: IS_B2_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -6.7093639, 25.4152355, -8.5930605, 32.1664772, -38.8758354, 34.0082932
1: -7.8063669, 28.8139172, -10.1064119, 36.5416260, -44.3479843, 38.9203300
2: -8.3207026, 28.8522396, -10.6501856, 36.5633698, -44.8840714, 39.5024261
3: -12.3585405, 29.6142330, -15.8564844, 37.7506409, -50.1091805, 45.4707184
4: -13.1818914, 28.3913708, -16.8155651, 36.0240479, -49.2059364, 45.2069359

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B2_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_A1_B1_A2_A2_A1

### Relational analysis result of IS_B2_A1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5586687, upper bound: 43.5689864
time: 0.51 seconds

## Relational analysis of IS_B2_A1_B1_A2_A2_A2

### Relational analysis result of IS_B2_A1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5547191, upper bound: 43.5680844
time: 0.58 seconds

## BFS IS instance: IS_B2_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -4.9945722, 19.1920948, -8.8174801, 33.0339127, -38.0284843, 28.0095711
1: -5.7327819, 21.7732925, -10.3915501, 37.5448494, -43.2776299, 32.1648407
2: -6.1764441, 21.6759148, -10.9315701, 37.5578690, -43.7343063, 32.6074829
3: -9.1741276, 22.2358761, -16.3067951, 38.8084564, -47.9825821, 38.5426712
4: -9.9434233, 21.2148380, -17.2786331, 36.9893837, -46.9328079, 38.4934692

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B2_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_A1_B2_A1_A1_A1

### Relational analysis result of IS_B2_A1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5612360, upper bound: 43.5210533
time: 0.52 seconds

## Relational analysis of IS_B2_A1_B2_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_B2_A1_A1_A1

### Relational analysis result of IS_B2_A1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5609907, upper bound: 43.5210533
time: 0.57 seconds

## Relational analysis of IS_B2_A1_B2_A1_A1_A2

### Relational analysis result of IS_B2_A1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5609907, upper bound: 43.5210533
time: 0.52 seconds

## BFS IS instance: IS_B2_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -5.5207715, 20.9111004, -8.7858028, 32.9264526, -38.4472198, 29.6969032
1: -6.3881083, 23.6853485, -10.3527565, 37.4227066, -43.8108139, 34.0381012
2: -6.8246484, 23.6887436, -10.8923359, 37.4335938, -44.2582436, 34.5810776
3: -10.1537533, 24.2320938, -16.2480621, 38.6802444, -48.8339958, 40.4801521
4: -10.8407536, 23.3468246, -17.2209148, 36.8629074, -47.7036591, 40.5677414

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B2_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_A1_B2_A1_A2_A1

### Relational analysis result of IS_B2_A1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5208035, upper bound: 43.5173906
time: 0.55 seconds

## Relational analysis of IS_B2_A1_B2_A1_A2_A2

### Relational analysis result of IS_B2_A1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4653523, upper bound: 43.5094192
time: 0.58 seconds

## BFS IS instance: IS_B2_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -6.2515597, 23.8048096, -8.7759876, 32.8803558, -39.1319160, 32.5807953
1: -7.2439523, 27.0316677, -10.3398943, 37.3700981, -44.6140518, 37.3715591
2: -7.7507277, 26.9848003, -10.8813286, 37.3827362, -45.1334648, 37.8661270
3: -11.4921074, 27.7163181, -16.2284031, 38.6279373, -50.1200447, 43.9447174
4: -12.3457909, 26.4698715, -17.2025299, 36.8155746, -49.1613655, 43.6724014

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B2_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_B2_A2_A1_A1

### Relational analysis result of IS_B2_A1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5553418, upper bound: 43.5442409
time: 0.50 seconds

## Relational analysis of IS_B2_A1_B2_A2_A1_A2

### Relational analysis result of IS_B2_A1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5553418, upper bound: 43.5442409
time: 0.48 seconds

## BFS IS instance: IS_B2_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -6.8540554, 26.0159321, -8.7504930, 32.7889481, -39.6430054, 34.7664223
1: -7.9552202, 29.5611782, -10.3080959, 37.2676659, -45.2228851, 39.8692741
2: -8.4965773, 29.5114670, -10.8498917, 37.2764397, -45.7730064, 40.3613586
3: -12.5889311, 30.3465195, -16.1796608, 38.5216827, -51.1106148, 46.5261612
4: -13.5135927, 28.9429054, -17.1574078, 36.7058182, -50.2194099, 46.1003036

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B2_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_B2_A2_A2_A1

### Relational analysis result of IS_B2_A1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5513966, upper bound: 43.5434515
time: 0.57 seconds

## Relational analysis of IS_B2_A1_B2_A2_A2_A2

### Relational analysis result of IS_B2_A1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5513966, upper bound: 43.5434515
time: 0.68 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -8.3524303, 31.3123322, -8.6322088, 32.3104172, -40.6628418, 39.9445305
1: -9.8154345, 35.5622025, -10.1549702, 36.7054749, -46.5209084, 45.7171707
2: -10.3523092, 35.5829659, -10.6975584, 36.7274017, -47.0797119, 46.2805214
3: -15.4090767, 36.7377586, -15.9301043, 37.9201851, -53.3292542, 52.6678619
4: -16.3657761, 35.0550346, -16.8876266, 36.1860275, -52.5517921, 51.9426613

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B2_A2_A1_B1_A1_B1

### Relational analysis result of IS_B2_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5565851, upper bound: 43.5565851
time: 0.60 seconds

## Relational analysis of IS_B2_A2_A1_B1_A1_B2

### Relational analysis result of IS_B2_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5565851, upper bound: 43.5565851
time: 0.78 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -8.9357872, 33.2270203, -8.6015329, 32.2070732, -41.1428604, 41.8285522
1: -10.5515137, 37.7184982, -10.1174545, 36.5886230, -47.1401367, 47.8359528
2: -11.0725422, 37.8242493, -10.6595440, 36.6078415, -47.6803741, 48.4837952
3: -16.5153427, 38.9987679, -15.8734093, 37.7966385, -54.3119736, 54.8721695
4: -17.3748760, 37.4123497, -16.8316593, 36.0642014, -53.4390793, 54.2440109

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_A2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_A2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_A2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B2_A2_A1_B1_A2_B1

### Relational analysis result of IS_B2_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5565851, upper bound: 43.5565851
time: 0.55 seconds

## Relational analysis of IS_B2_A2_A1_B1_A2_B2

### Relational analysis result of IS_B2_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5565851, upper bound: 43.5565851
time: 0.78 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -8.3524303, 31.3123322, -8.8174801, 33.0339127, -41.3863373, 40.1298141
1: -9.8154345, 35.5622025, -10.3915501, 37.5448494, -47.3602829, 45.9537506
2: -10.3523092, 35.5829659, -10.9315701, 37.5578690, -47.9101791, 46.5145340
3: -15.4090767, 36.7377586, -16.3067951, 38.8084564, -54.2175293, 53.0445557
4: -16.3657761, 35.0550346, -17.2786331, 36.9893837, -53.3551559, 52.3336678

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_A2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_A2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B2_A2_A1_B2_A1_B1

### Relational analysis result of IS_B2_A2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5463511, upper bound: 43.5181075
time: 0.54 seconds

## Relational analysis of IS_B2_A2_A1_B2_A1_B2

### Relational analysis result of IS_B2_A2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5463511, upper bound: 43.5181075
time: 0.56 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -8.9357872, 33.2270203, -8.7858028, 32.9264526, -41.8622360, 42.0128250
1: -10.5515137, 37.7184982, -10.3527565, 37.4227066, -47.9742203, 48.0712547
2: -11.0725422, 37.8242493, -10.8923359, 37.4335938, -48.5061264, 48.7165833
3: -16.5153427, 38.9987679, -16.2480621, 38.6802444, -55.1955757, 55.2468300
4: -17.3748760, 37.4123497, -17.2209148, 36.8629074, -54.2377853, 54.6332626

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_A2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_A2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B2_A2_A1_B2_A2_B1

### Relational analysis result of IS_B2_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5463511, upper bound: 43.5181077
time: 0.52 seconds

## Relational analysis of IS_B2_A2_A1_B2_A2_B2

### Relational analysis result of IS_B2_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5463511, upper bound: 43.5181077
time: 0.53 seconds

## BFS IS instance: IS_B2_A2_A2_A1_A1

### Backsubstitution after applying IS history:
0: -8.5025597, 31.8738346, -8.7978945, 32.9071312, -41.4096909, 40.6717300
1: -10.0117111, 36.1823883, -10.3590813, 37.3862305, -47.3979378, 46.5414696
2: -10.5372906, 36.2474289, -10.9039745, 37.4141464, -47.9514351, 47.1514053
3: -15.7192554, 37.3783875, -16.2424736, 38.6300850, -54.3493423, 53.6208611
4: -16.6247540, 35.7609291, -17.2002182, 36.8718681, -53.4966164, 52.9611473

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_A2_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B2_A2_A2_A1_A1_A1

### Relational analysis result of IS_B2_A2_A2_A1_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3268282, upper bound: 43.3124852
time: 0.77 seconds

## Relational analysis of IS_B2_A2_A2_A1_A1_A2

### Relational analysis result of IS_B2_A2_A2_A1_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3358028, upper bound: 43.3332483
time: 0.66 seconds

## BFS IS instance: IS_B2_A2_A2_A1_A2

### Backsubstitution after applying IS history:
0: -8.5394516, 32.0485039, -8.7998943, 32.9144211, -41.4538727, 40.8483963
1: -10.0546646, 36.4156799, -10.3614931, 37.3947411, -47.4494057, 46.7771721
2: -10.5878363, 36.4272728, -10.9064531, 37.4224205, -48.0102577, 47.3337212
3: -15.7905474, 37.6386337, -16.2461872, 38.6390495, -54.4295959, 53.8848190
4: -16.7589092, 35.8717308, -17.2043457, 36.8796883, -53.6385956, 53.0760765

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_A2_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_A2_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B2_A2_A2_A1_A2_B1

### Relational analysis result of IS_B2_A2_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5385524, upper bound: 43.5153401
time: 0.66 seconds

## Relational analysis of IS_B2_A2_A2_A1_A2_B2

### Relational analysis result of IS_B2_A2_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5385524, upper bound: 43.5153401
time: 0.59 seconds

## BFS IS instance: IS_B2_A2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -9.1115875, 33.9125404, -8.5244942, 31.9350224, -41.0466080, 42.4370346
1: -10.7727814, 38.5158424, -10.0274296, 36.2731361, -47.0459175, 48.5432739
2: -11.2928276, 38.6079216, -10.5661516, 36.2987938, -47.5916214, 49.1740723
3: -16.8676147, 39.8401985, -15.7337894, 37.4785500, -54.3461647, 55.5739861
4: -17.7444897, 38.1677361, -16.6914082, 35.7676544, -53.5121460, 54.8591461

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_A2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B2_A2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_A2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_A2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A2_A2_A2_B1_B1

### Relational analysis result of IS_B2_A2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476
time: 0.58 seconds

## Relational analysis of IS_B2_A2_A2_A2_B1_B2

### Relational analysis result of IS_B2_A2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476
time: 0.75 seconds

## BFS IS instance: IS_B2_A2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -9.1115875, 33.9125404, -9.1002617, 33.8217354, -42.9333229, 43.0127945
1: -10.7727814, 38.5158424, -10.7537775, 38.3973083, -49.1700897, 49.2696190
2: -11.2928276, 38.6079216, -11.2777548, 38.5081139, -49.8009415, 49.8856773
3: -16.8676147, 39.8401985, -16.8248043, 39.7067146, -56.5743256, 56.6650009
4: -17.7444897, 38.1677361, -17.6864700, 38.0939445, -55.8384323, 55.8542061

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_A2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_A2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A2_A2_A2_B2_B1

### Relational analysis result of IS_B2_A2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476
time: 0.53 seconds

## Relational analysis of IS_B2_A2_A2_A2_B2_B2

### Relational analysis result of IS_B2_A2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476
time: 0.56 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 5.22 seconds
IS_B1_A1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.5325401, upper bound: 43.5325401
IS_B1_A1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.5325401, upper bound: 43.5325401
IS_B1_A1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.4670481, upper bound: 43.5060097
IS_B1_A1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.4611298, upper bound: 43.4611298
IS_B1_A1_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.5239338, upper bound: 43.4695699
IS_B1_A1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.5278449, upper bound: 43.4697512
IS_B1_A1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.5237448, upper bound: 43.4692334
IS_B1_A1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.5260855, upper bound: 43.4692354
IS_B1_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.4695699, upper bound: 43.5239338
IS_B1_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.4695699, upper bound: 43.5278449
IS_B1_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.4670481, upper bound: 43.5237448
IS_B1_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.4692354, upper bound: 43.5260855
IS_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.5586812, upper bound: 43.5586812
IS_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.5586812, upper bound: 43.5586812
IS_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.5549699, upper bound: 43.5539200
IS_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.5539466, upper bound: 43.5539200
IS_B1_A2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.5590796, upper bound: 43.5674811
IS_B1_A2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.5590796, upper bound: 43.5674811
IS_B1_A2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.5547796, upper bound: 43.5255054
IS_B1_A2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.5183207, upper bound: 43.4668146
IS_B1_A2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.5210533, upper bound: 43.5609907
IS_B1_A2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.5210533, upper bound: 43.5609907
IS_B1_A2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.5173906, upper bound: 43.5208035
IS_B1_A2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.5094192, upper bound: 43.4653523
IS_B1_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.5689864, upper bound: 43.5586687
IS_B1_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.5680844, upper bound: 43.5547191
IS_B1_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.5689864, upper bound: 43.5586687
IS_B1_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.5680844, upper bound: 43.5547191
IS_B1_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.5442409, upper bound: 43.5553418
IS_B1_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.5442409, upper bound: 43.5553418
IS_B1_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.5198541, upper bound: 43.5513966
IS_B1_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.5434515, upper bound: 43.5513966
IS_B2_A1_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.5674811, upper bound: 43.5590796
IS_B2_A1_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.5674811, upper bound: 43.5590796
IS_B2_A1_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.5255054, upper bound: 43.5547796
IS_B2_A1_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.4668146, upper bound: 43.5183207
IS_B2_A1_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.5586687, upper bound: 43.5689864
IS_B2_A1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.5547191, upper bound: 43.5680844
IS_B2_A1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.5586687, upper bound: 43.5689864
IS_B2_A1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.5547191, upper bound: 43.5680844
IS_B2_A1_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.5609907, upper bound: 43.5210533
IS_B2_A1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.5609907, upper bound: 43.5210533
IS_B2_A1_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.5208035, upper bound: 43.5173906
IS_B2_A1_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.4653523, upper bound: 43.5094192
IS_B2_A1_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.5553418, upper bound: 43.5442409
IS_B2_A1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.5553418, upper bound: 43.5442409
IS_B2_A1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.5513966, upper bound: 43.5434515
IS_B2_A1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.5513966, upper bound: 43.5434515
IS_B2_A2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.5565851, upper bound: 43.5565851
IS_B2_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.5565851, upper bound: 43.5565851
IS_B2_A2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.5565851, upper bound: 43.5565851
IS_B2_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.5565851, upper bound: 43.5565851
IS_B2_A2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.5463511, upper bound: 43.5181075
IS_B2_A2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.5463511, upper bound: 43.5181075
IS_B2_A2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.5463511, upper bound: 43.5181077
IS_B2_A2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.5463511, upper bound: 43.5181077
IS_B2_A2_A2_A1_A1_A1, status: Status.VERIFIED, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.3268282, upper bound: 43.3124852
IS_B2_A2_A2_A1_A1_A2, status: Status.VERIFIED, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.3358028, upper bound: 43.3332483
IS_B2_A2_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.5385524, upper bound: 43.5153401
IS_B2_A2_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.5385524, upper bound: 43.5153401
IS_B2_A2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476
IS_B2_A2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476
IS_B2_A2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476
IS_B2_A2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.22
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476

## BFS IS instance: IS_B1_A1_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -4.9090037, 18.9296074, -4.9090037, 18.9296074, -23.8386078, 23.8386078
1: -5.6224365, 21.5094490, -5.6224365, 21.5094490, -27.1318855, 27.1318855
2: -6.0714874, 21.3573895, -6.0714874, 21.3573895, -27.4288769, 27.4288769
3: -9.0102825, 21.9570026, -9.0102825, 21.9570026, -30.9672852, 30.9672852
4: -9.8312683, 20.8039608, -9.8312683, 20.8039608, -30.6352291, 30.6352291

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B1_A1_A1_B1_B1

### Relational analysis result of IS_B1_A1_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5507912, upper bound: 43.5340445
time: 0.54 seconds

## Relational analysis of IS_B1_A1_B1_A1_A1_B1_B2

### Relational analysis result of IS_B1_A1_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5282079, upper bound: 43.4707168
time: 0.65 seconds

## BFS IS instance: IS_B1_A1_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -4.9090037, 18.9296074, -5.4008245, 20.7929802, -25.7019825, 24.3304329
1: -5.6224365, 21.5094490, -6.2052450, 23.6457863, -29.2682228, 27.7146950
2: -6.0714874, 21.3573895, -6.6841125, 23.4871082, -29.5585957, 28.0415020
3: -9.0102825, 21.9570026, -9.9212875, 24.1726971, -33.1829796, 31.8782902
4: -9.8312683, 20.8039608, -10.8100157, 22.8875217, -32.7187881, 31.6139755

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B1_A1_A1_B2_B1

### Relational analysis result of IS_B1_A1_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5507912, upper bound: 43.5340445
time: 0.54 seconds

## Relational analysis of IS_B1_A1_B1_A1_A1_B2_B2

### Relational analysis result of IS_B1_A1_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5282079, upper bound: 43.4707168
time: 0.55 seconds

## BFS IS instance: IS_B1_A1_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -5.4008245, 20.7929802, -5.0289822, 19.3434906, -24.7443161, 25.8219624
1: -6.2052450, 23.6457863, -5.7710342, 21.9570446, -28.1622887, 29.4168205
2: -6.6841125, 23.4871082, -6.2188196, 21.8396339, -28.5237465, 29.7059288
3: -9.9212875, 24.1726971, -9.2369642, 22.4180202, -32.3393097, 33.4096603
4: -10.8100157, 22.8875217, -10.0197945, 21.3444195, -32.1544304, 32.9073143

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B1_A1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B1_A1_A2_B1_A1

### Relational analysis result of IS_B1_A1_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4611298, upper bound: 43.4611298
time: 0.55 seconds

## Relational analysis of IS_B1_A1_B1_A1_A2_B1_A2

### Relational analysis result of IS_B1_A1_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4611298, upper bound: 43.4611298
time: 0.74 seconds

## BFS IS instance: IS_B1_A1_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -5.3541193, 20.6424465, -5.2039084, 20.0115147, -25.3656349, 25.8463535
1: -6.1482754, 23.4779243, -5.9933915, 22.7232742, -28.8715477, 29.4713097
2: -6.6270404, 23.3115005, -6.4392595, 22.6151752, -29.2422142, 29.7507553
3: -9.8374319, 23.9987335, -9.5919619, 23.2303028, -33.0677338, 33.5906944
4: -10.7313147, 22.7049065, -10.3826447, 22.1134319, -32.8447456, 33.0875511

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B1_A1_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B1_A1_A2_B2_B1

### Relational analysis result of IS_B1_A1_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4611298, upper bound: 43.4611298
time: 0.53 seconds

## Relational analysis of IS_B1_A1_B1_A1_A2_B2_B2

### Relational analysis result of IS_B1_A1_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4611298, upper bound: 43.4611298
time: 0.53 seconds

## BFS IS instance: IS_B1_A1_B1_A2_A1_A1

### Backsubstitution after applying IS history:
0: -6.0885549, 23.2257309, -5.2551436, 20.1315994, -26.2201519, 28.4808731
1: -7.0463910, 26.3764725, -6.0467672, 22.8425674, -29.8889580, 32.4232330
2: -7.5475416, 26.3148308, -6.4991312, 22.7556610, -30.3032017, 32.8139610
3: -11.1884851, 27.0328350, -9.6570997, 23.3396664, -34.5281525, 36.6899338
4: -12.0433311, 25.7938061, -10.4297295, 22.2815781, -34.3249016, 36.2235336

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B1_A2_A1_A1_B1

### Relational analysis result of IS_B1_A1_B1_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5239338, upper bound: 43.4695699
time: 0.56 seconds

## Relational analysis of IS_B1_A1_B1_A2_A1_A1_B2

### Relational analysis result of IS_B1_A1_B1_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5239338, upper bound: 43.4695699
time: 0.51 seconds

## BFS IS instance: IS_B1_A1_B1_A2_A1_A2

### Backsubstitution after applying IS history:
0: -6.2608924, 23.8898964, -5.2088509, 19.9812698, -26.2421627, 29.0987473
1: -7.2641339, 27.1352291, -5.9901991, 22.6750488, -29.9391823, 33.1254272
2: -7.7652574, 27.0853271, -6.4421506, 22.5805073, -30.3457642, 33.5274773
3: -11.5361423, 27.8390503, -9.5738401, 23.1661835, -34.7023201, 37.4128914
4: -12.4052420, 26.5429840, -10.3513479, 22.0983219, -34.5035629, 36.8943291

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B1_A1_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B1_A2_A1_A2_B1

### Relational analysis result of IS_B1_A1_B1_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5239338, upper bound: 43.4697512
time: 0.61 seconds

## Relational analysis of IS_B1_A1_B1_A2_A1_A2_B2

### Relational analysis result of IS_B1_A1_B1_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5239338, upper bound: 43.4697512
time: 0.61 seconds

## BFS IS instance: IS_B1_A1_B1_A2_A2_A1

### Backsubstitution after applying IS history:
0: -6.6963181, 25.4623470, -5.2252574, 20.0268478, -26.7231655, 30.6876049
1: -7.7666783, 28.9356041, -6.0095611, 22.7262707, -30.4929447, 34.9451637
2: -8.3003368, 28.8721256, -6.4622755, 22.6329479, -30.9332829, 35.3343925
3: -12.3004551, 29.6954193, -9.6003199, 23.2192421, -35.5196991, 39.2957382
4: -13.2228851, 28.2979012, -10.3769379, 22.1532593, -35.3761368, 38.6748314

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B1_A1_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B1_A2_A2_A1_B1

### Relational analysis result of IS_B1_A1_B1_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5237448, upper bound: 43.4692334
time: 0.59 seconds

## Relational analysis of IS_B1_A1_B1_A2_A2_A1_B2

### Relational analysis result of IS_B1_A1_B1_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5237448, upper bound: 43.4692334
time: 0.58 seconds

## BFS IS instance: IS_B1_A1_B1_A2_A2_A2

### Backsubstitution after applying IS history:
0: -6.8828583, 26.1740513, -5.1790419, 19.8766632, -26.7595196, 31.3530922
1: -8.0028114, 29.7413483, -5.9531384, 22.5589466, -30.5617542, 35.6944885
2: -8.5354614, 29.6899586, -6.4054241, 22.4579411, -30.9934006, 36.0953827
3: -12.6767759, 30.5600548, -9.5172176, 23.0459347, -35.7227097, 40.0772705
4: -13.6112204, 29.0972939, -10.2987232, 21.9701138, -35.5813332, 39.3960114

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B1_A1_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B1_A2_A2_A2_B1

### Relational analysis result of IS_B1_A1_B1_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5260855, upper bound: 43.4692354
time: 0.62 seconds

## Relational analysis of IS_B1_A1_B1_A2_A2_A2_B2

### Relational analysis result of IS_B1_A1_B1_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4611298, upper bound: 43.4692354
time: 0.97 seconds

## BFS IS instance: IS_B1_A1_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -5.2551436, 20.1315994, -6.0885549, 23.2257309, -28.4808731, 26.2201519
1: -6.0467672, 22.8425674, -7.0463910, 26.3764725, -32.4232368, 29.8889580
2: -6.4991312, 22.7556610, -7.5475416, 26.3148308, -32.8139534, 30.3031998
3: -9.6570997, 23.3396664, -11.1884851, 27.0328350, -36.6899338, 34.5281525
4: -10.4297295, 22.2815781, -12.0433311, 25.7938061, -36.2235336, 34.3249016

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B2_A1_B1_B1_A1

### Relational analysis result of IS_B1_A1_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4695699, upper bound: 43.5239338
time: 0.56 seconds

## Relational analysis of IS_B1_A1_B2_A1_B1_B1_A2

### Relational analysis result of IS_B1_A1_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4695699, upper bound: 43.5239338
time: 0.47 seconds

## BFS IS instance: IS_B1_A1_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -5.2088509, 19.9812698, -6.2608924, 23.8898964, -29.0987473, 26.2421627
1: -5.9901991, 22.6750488, -7.2641339, 27.1352291, -33.1254272, 29.9391823
2: -6.4421506, 22.5805073, -7.7652574, 27.0853271, -33.5274773, 30.3457642
3: -9.5738401, 23.1661835, -11.5361423, 27.8390503, -37.4128914, 34.7023201
4: -10.3513479, 22.0983219, -12.4052420, 26.5429840, -36.8943291, 34.5035629

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A1_B2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B2_A1_B1_B2_A1

### Relational analysis result of IS_B1_A1_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4697512, upper bound: 43.5278449
time: 0.68 seconds

## Relational analysis of IS_B1_A1_B2_A1_B1_B2_A2

### Relational analysis result of IS_B1_A1_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4697512, upper bound: 43.5278449
time: 0.55 seconds

## BFS IS instance: IS_B1_A1_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -5.2252574, 20.0268478, -6.6963181, 25.4623470, -30.6876030, 26.7231655
1: -6.0095611, 22.7262707, -7.7666783, 28.9356041, -34.9451637, 30.4929466
2: -6.4622755, 22.6329479, -8.3003368, 28.8721256, -35.3343925, 30.9332848
3: -9.6003199, 23.2192421, -12.3004551, 29.6954193, -39.2957382, 35.5196991
4: -10.3769379, 22.1532593, -13.2228851, 28.2979012, -38.6748314, 35.3761406

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A1_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B2_A1_B2_B1_A1

### Relational analysis result of IS_B1_A1_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4692334, upper bound: 43.5237448
time: 0.56 seconds

## Relational analysis of IS_B1_A1_B2_A1_B2_B1_A2

### Relational analysis result of IS_B1_A1_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4692334, upper bound: 43.5237448
time: 0.69 seconds

## BFS IS instance: IS_B1_A1_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -5.1790419, 19.8766632, -6.8828583, 26.1740513, -31.3530922, 26.7595196
1: -5.9531384, 22.5589466, -8.0028114, 29.7413483, -35.6944885, 30.5617542
2: -6.4054241, 22.4579411, -8.5354614, 29.6899586, -36.0953827, 30.9934006
3: -9.5172176, 23.0459347, -12.6767759, 30.5600548, -40.0772705, 35.7227097
4: -10.2987232, 21.9701138, -13.6112204, 29.0972939, -39.3960114, 35.5813332

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A1_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B2_A1_B2_B2_A1

### Relational analysis result of IS_B1_A1_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4692354, upper bound: 43.5260855
time: 0.96 seconds

## Relational analysis of IS_B1_A1_B2_A1_B2_B2_A2

### Relational analysis result of IS_B1_A1_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4611298, upper bound: 43.5260855
time: 0.60 seconds

## BFS IS instance: IS_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6.5275750, 24.7183266, -6.5283041, 24.7212219, -31.2487965, 31.2466278
1: -7.5759025, 28.0228786, -7.5771909, 28.0262680, -35.6021652, 35.6000710
2: -8.0913868, 28.0491600, -8.0923138, 28.0525475, -36.1439362, 36.1414719
3: -11.9907589, 28.7559509, -11.9928169, 28.7598476, -40.7506065, 40.7487679
4: -12.8009701, 27.6084270, -12.8025274, 27.6119270, -40.4128914, 40.4109535

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5571791, upper bound: 43.5771870
time: 0.88 seconds

## Relational analysis of IS_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5569027, upper bound: 43.5754099
time: 1.04 seconds

## BFS IS instance: IS_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -6.7019677, 25.3857117, -6.5283041, 24.7212219, -31.4231892, 31.9140167
1: -7.7935724, 28.7790051, -7.5771909, 28.0262680, -35.8198357, 36.3561974
2: -8.3112373, 28.8177509, -8.0923138, 28.0525475, -36.3637848, 36.9100647
3: -12.3380346, 29.5745430, -11.9928169, 28.7598476, -41.0978813, 41.5673599
4: -13.1659355, 28.3559570, -12.8025274, 27.6119270, -40.7778587, 41.1584854

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5571791, upper bound: 43.5771870
time: 0.61 seconds

## Relational analysis of IS_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5569027, upper bound: 43.5754099
time: 0.66 seconds

## BFS IS instance: IS_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6.2515597, 23.8048096, -6.7093639, 25.4152355, -31.6667957, 30.5141735
1: -7.2439523, 27.0316677, -7.8063669, 28.8139172, -36.0578690, 34.8380241
2: -7.7507277, 26.9848003, -8.3207026, 28.8522396, -36.6029663, 35.3055038
3: -11.4921074, 27.7163181, -12.3585405, 29.6142330, -41.1063385, 40.0748558
4: -12.3457909, 26.4698715, -13.1818914, 28.3913708, -40.7371597, 39.6517639

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5539466, upper bound: 43.5539200
time: 0.66 seconds

## Relational analysis of IS_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5539466, upper bound: 43.5539200
time: 0.61 seconds

## BFS IS instance: IS_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6.8540554, 26.0159321, -6.6871252, 25.3378582, -32.1919022, 32.7030563
1: -7.9552202, 29.5611782, -7.7787871, 28.7282238, -36.6834450, 37.3399658
2: -8.4965773, 29.5114670, -8.2934113, 28.7617912, -37.2583618, 37.8048782
3: -12.5889311, 30.3465195, -12.3162842, 29.5237465, -42.1126785, 42.6627960
4: -13.5135927, 28.9429054, -13.1422386, 28.2980289, -41.8116226, 42.0851402

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B2_A2_B2_A2_A1

### Relational analysis result of IS_B1_A1_B2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5539466, upper bound: 43.5539200
time: 0.81 seconds

## Relational analysis of IS_B1_A1_B2_A2_B2_A2_A2

### Relational analysis result of IS_B1_A1_B2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5539466, upper bound: 43.5539200
time: 0.65 seconds

## BFS IS instance: IS_B1_A2_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -8.6322088, 32.3104172, -4.8395281, 18.6394978, -27.2717056, 37.1499443
1: -10.1549702, 36.7054749, -5.5451407, 21.1490383, -31.3040047, 42.2506142
2: -10.6975584, 36.7274017, -5.9836831, 21.0349770, -31.7325344, 42.7110863
3: -15.9301043, 37.9201851, -8.8850317, 21.5857143, -37.5158195, 46.8052177
4: -16.8876266, 36.1860275, -9.6547794, 20.5678463, -37.4554749, 45.8408012

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_B1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_A1_B1_B1_B1

### Relational analysis result of IS_B1_A2_B1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5589374, upper bound: 43.5674811
time: 0.51 seconds

## Relational analysis of IS_B1_A2_B1_A1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_B1_A2_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A2_B1_A1_B1_B1_A1

### Relational analysis result of IS_B1_A2_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5590796, upper bound: 43.5674811
time: 0.58 seconds

## Relational analysis of IS_B1_A2_B1_A1_B1_B1_A2

### Relational analysis result of IS_B1_A2_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5590796, upper bound: 43.5674811
time: 0.55 seconds

## BFS IS instance: IS_B1_A2_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -8.6322088, 32.3104172, -5.0165229, 19.3200130, -27.9522209, 37.3269348
1: -10.1549702, 36.7054749, -5.7698326, 21.9297009, -32.0846634, 42.4753075
2: -10.6975584, 36.7274017, -6.2069774, 21.8248901, -32.5224457, 42.9343796
3: -15.9301043, 37.9201851, -9.2441177, 22.4114609, -38.3415642, 47.1643028
4: -16.8876266, 36.1860275, -10.0225515, 21.3482018, -38.2358246, 46.2085800

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_B1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_A1_B1_B2_B1

### Relational analysis result of IS_B1_A2_B1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5589374, upper bound: 43.5674811
time: 0.53 seconds

## Relational analysis of IS_B1_A2_B1_A1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_B1_A2_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A2_B1_A1_B1_B2_A1

### Relational analysis result of IS_B1_A2_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5590796, upper bound: 43.5674811
time: 0.69 seconds

## Relational analysis of IS_B1_A2_B1_A1_B1_B2_A2

### Relational analysis result of IS_B1_A2_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5590796, upper bound: 43.5674811
time: 0.54 seconds

## BFS IS instance: IS_B1_A2_B1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -8.6015329, 32.2070732, -5.2180510, 19.8573036, -28.4588356, 37.4251251
1: -10.1174545, 36.5886230, -6.0158553, 22.5195217, -32.6369781, 42.6044769
2: -10.6595440, 36.6078415, -6.4501691, 22.4596786, -33.1192245, 43.0580101
3: -15.8734093, 37.7966385, -9.5833883, 23.0195580, -38.8929672, 47.3800240
4: -16.8316593, 36.0642014, -10.3194571, 22.0301399, -38.8617935, 46.3836594

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_B1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_B1_A1_B2_B1_B1

### Relational analysis result of IS_B1_A2_B1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5547796, upper bound: 43.5255054
time: 0.76 seconds

## Relational analysis of IS_B1_A2_B1_A1_B2_B1_B2

### Relational analysis result of IS_B1_A2_B1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5547796, upper bound: 43.5255054
time: 0.59 seconds

## BFS IS instance: IS_B1_A2_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -8.5771322, 32.1201172, -5.6673207, 21.5628738, -30.1400051, 37.7874374
1: -10.0870800, 36.4921265, -6.5436316, 24.4766960, -34.5637741, 43.0357590
2: -10.6295090, 36.5066452, -7.0108037, 24.4052582, -35.0347672, 43.5174484
3: -15.8268700, 37.6954384, -10.4097338, 25.0520611, -40.8789291, 48.1051712
4: -16.7884083, 35.9595795, -11.2162266, 23.9139538, -40.7023582, 47.1758041

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_B1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_B1_A1_B2_B2_B1

### Relational analysis result of IS_B1_A2_B1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5183207, upper bound: 43.4668146
time: 0.57 seconds

## Relational analysis of IS_B1_A2_B1_A1_B2_B2_B2

### Relational analysis result of IS_B1_A2_B1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5183207, upper bound: 43.4668146
time: 0.75 seconds

## BFS IS instance: IS_B1_A2_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -8.8174801, 33.0339127, -4.8395281, 18.6394978, -27.4569740, 37.8734398
1: -10.3915501, 37.5448494, -5.5451407, 21.1490383, -31.5405846, 43.0899887
2: -10.9315701, 37.5578690, -5.9836831, 21.0349770, -31.9665470, 43.5415535
3: -16.3067951, 38.8084564, -8.8850317, 21.5857143, -37.8925095, 47.6934891
4: -17.2786331, 36.9893837, -9.6547794, 20.5678463, -37.8464813, 46.6441650

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_A2_B1_B1_B1

### Relational analysis result of IS_B1_A2_B1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5210533, upper bound: 43.5609907
time: 0.54 seconds

## Relational analysis of IS_B1_A2_B1_A2_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_B1_A2_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A2_B1_A2_B1_B1_A1

### Relational analysis result of IS_B1_A2_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5210533, upper bound: 43.5609907
time: 0.67 seconds

## Relational analysis of IS_B1_A2_B1_A2_B1_B1_A2

### Relational analysis result of IS_B1_A2_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5210533, upper bound: 43.5609907
time: 0.57 seconds

## BFS IS instance: IS_B1_A2_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -8.8174801, 33.0339127, -5.0165229, 19.3200130, -28.1374893, 38.0504303
1: -10.3915501, 37.5448494, -5.7698326, 21.9297009, -32.3212433, 43.3146820
2: -10.9315701, 37.5578690, -6.2069774, 21.8248901, -32.7564621, 43.7648468
3: -16.3067951, 38.8084564, -9.2441177, 22.4114609, -38.7182541, 48.0525742
4: -17.2786331, 36.9893837, -10.0225515, 21.3482018, -38.6268311, 47.0119362

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_A2_B1_B2_B1

### Relational analysis result of IS_B1_A2_B1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5210533, upper bound: 43.5609907
time: 0.63 seconds

## Relational analysis of IS_B1_A2_B1_A2_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A2_B1_A2_B1_B2_A1

### Relational analysis result of IS_B1_A2_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5210533, upper bound: 43.5609907
time: 0.59 seconds

## Relational analysis of IS_B1_A2_B1_A2_B1_B2_A2

### Relational analysis result of IS_B1_A2_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5210533, upper bound: 43.5609907
time: 0.58 seconds

## BFS IS instance: IS_B1_A2_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -8.7858028, 32.9264526, -5.1706009, 19.7029343, -28.4887371, 38.0970535
1: -10.3527565, 37.4227066, -5.9576077, 22.3476372, -32.7003899, 43.3803101
2: -10.8923359, 37.4335938, -6.3918052, 22.2797489, -33.1720848, 43.8253937
3: -16.2480621, 38.6802444, -9.4978619, 22.8413162, -39.0893669, 48.1781082
4: -17.2209148, 36.8629074, -10.2391148, 21.8423119, -39.0632248, 47.1020203

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_B1_A2_B2_B1_B1

### Relational analysis result of IS_B1_A2_B1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5173906, upper bound: 43.5208035
time: 0.53 seconds

## Relational analysis of IS_B1_A2_B1_A2_B2_B1_B2

### Relational analysis result of IS_B1_A2_B1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5173906, upper bound: 43.5208035
time: 0.51 seconds

## BFS IS instance: IS_B1_A2_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -8.7600079, 32.8338356, -5.6194525, 21.4083080, -30.1683140, 38.4532814
1: -10.3205700, 37.3189049, -6.4850497, 24.3042545, -34.6248245, 43.8039551
2: -10.8605299, 37.3259163, -6.9518995, 24.2250195, -35.0855408, 44.2778130
3: -16.1987171, 38.5725670, -10.3236294, 24.8734093, -41.0721283, 48.8961945
4: -17.1752586, 36.7517509, -11.1353645, 23.7268181, -40.9020691, 47.8871155

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_B1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A2_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_B1_A2_B2_B2_B1

### Relational analysis result of IS_B1_A2_B1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5094192, upper bound: 43.4653523
time: 0.64 seconds

## Relational analysis of IS_B1_A2_B1_A2_B2_B2_B2

### Relational analysis result of IS_B1_A2_B1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5094192, upper bound: 43.4653523
time: 0.66 seconds

## BFS IS instance: IS_B1_A2_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -8.5930605, 32.1664772, -6.0885549, 23.2257309, -31.8187904, 38.2550201
1: -10.1064119, 36.5416260, -7.0463910, 26.3764725, -36.4828720, 43.5880165
2: -10.6501856, 36.5633698, -7.5475416, 26.3148308, -36.9650116, 44.1109085
3: -15.8564844, 37.7506409, -11.1884851, 27.0328350, -42.8893166, 48.9391251
4: -16.8155651, 36.0240479, -12.0433311, 25.7938061, -42.6093712, 48.0673714

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_B2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A2_B2_A1_B1_B1_B1

### Relational analysis result of IS_B1_A2_B2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5593660, upper bound: 43.5792777
time: 1.03 seconds

## Relational analysis of IS_B1_A2_B2_A1_B1_B1_B2

### Relational analysis result of IS_B1_A2_B2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5579472, upper bound: 43.5585932
time: 0.55 seconds

## BFS IS instance: IS_B1_A2_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -8.5716076, 32.0911560, -6.6963181, 25.4623470, -34.0339508, 38.7874756
1: -10.0799026, 36.4573479, -7.7666783, 28.9356041, -39.0155067, 44.2240257
2: -10.6238031, 36.4755173, -8.3003368, 28.8721256, -39.4959221, 44.7758560
3: -15.8159027, 37.6629105, -12.3004551, 29.6954193, -45.5113220, 49.9633636
4: -16.7776718, 35.9328804, -13.2228851, 28.2979012, -45.0755730, 49.1557655

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_B2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A2_B2_A1_B1_B2_B1

### Relational analysis result of IS_B1_A2_B2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5590796, upper bound: 43.5763264
time: 0.82 seconds

## Relational analysis of IS_B1_A2_B2_A1_B1_B2_B2

### Relational analysis result of IS_B1_A2_B2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5579165, upper bound: 43.5585180
time: 0.62 seconds

## BFS IS instance: IS_B1_A2_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -8.5930605, 32.1664772, -6.2608924, 23.8898964, -32.4829559, 38.4273682
1: -10.1064119, 36.5416260, -7.2641339, 27.1352291, -37.2416420, 43.8057556
2: -10.6501856, 36.5633698, -7.7652574, 27.0853271, -37.7355118, 44.3286285
3: -15.8564844, 37.7506409, -11.5361423, 27.8390503, -43.6955338, 49.2867775
4: -16.8155651, 36.0240479, -12.4052420, 26.5429840, -43.3585434, 48.4292908

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0312500, mid=0.0312500, abs_max=46.890357971191406
rel_dist={3: [-43.60075692157075, 43.60075692157075]}

## Binary search (step 2) starts
Candidate diff: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5730299, upper bound: 43.5833343
time: 0.50 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701514, upper bound: 43.5701514
time: 0.50 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.20 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 1.20
Output dim: 3, lower bound: -43.5730299, upper bound: 43.5833343
IS_B2, status: Status.UNKNOWN, split count: 1, time: 1.20
Output dim: 3, lower bound: -43.5701514, upper bound: 43.5701514

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -7.5994468, 28.0988369, -6.4284191, 24.1388493, -31.7382946, 34.5272560
1: -8.8856230, 31.7429333, -7.4645872, 27.3031864, -36.1888084, 39.2075195
2: -9.4135551, 31.9497280, -7.9618182, 27.3991623, -36.8127174, 39.9115448
3: -13.9250774, 32.6126556, -11.7890568, 27.9809723, -41.9060516, 44.4017105
4: -14.4791918, 31.8966846, -12.4559155, 27.1424770, -41.6216698, 44.3525963

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701514, upper bound: 43.5701514
time: 0.54 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701514, upper bound: 43.5701514
time: 0.52 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -7.5638342, 27.9483128, -8.8551903, 33.0941162, -40.6579514, 36.8035049
1: -8.8431463, 31.5652332, -10.4291992, 37.5969582, -46.4401054, 41.9944267
2: -9.3718910, 31.7789803, -10.9747181, 37.6314697, -47.0033607, 42.7537003
3: -13.8541670, 32.4234009, -16.3464375, 38.8504906, -52.7046471, 48.7698364
4: -14.3933516, 31.7455139, -17.2999840, 37.0951767, -51.4885292, 49.0454941

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_B1

### Relational analysis result of IS_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4482981, upper bound: 43.4036863
time: 0.56 seconds

## Relational analysis of IS_B2_B2

### Relational analysis result of IS_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4482981, upper bound: 43.4036863
time: 0.64 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.37 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 3.37
Output dim: 3, lower bound: -43.5701514, upper bound: 43.5701514
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 3.37
Output dim: 3, lower bound: -43.5701514, upper bound: 43.5701514
IS_B2_B1, status: Status.UNKNOWN, split count: 2, time: 3.37
Output dim: 3, lower bound: -43.4482981, upper bound: 43.4036863
IS_B2_B2, status: Status.UNKNOWN, split count: 2, time: 3.37
Output dim: 3, lower bound: -43.4482981, upper bound: 43.4036863

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -6.4284191, 24.1388493, -6.4284191, 24.1388493, -30.5672684, 30.5672684
1: -7.4645872, 27.3031864, -7.4645872, 27.3031864, -34.7677727, 34.7677727
2: -7.9618182, 27.3991623, -7.9618182, 27.3991623, -35.3609810, 35.3609810
3: -11.7890568, 27.9809723, -11.7890568, 27.9809723, -39.7700233, 39.7700272
4: -12.4559155, 27.1424770, -12.4559155, 27.1424770, -39.5983925, 39.5983925

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B1_A1_B1

### Relational analysis result of IS_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5730299, upper bound: 43.5833343
time: 0.53 seconds

## Relational analysis of IS_B1_A1_B2

### Relational analysis result of IS_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5726613, upper bound: 43.5812720
time: 0.52 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -8.8551903, 33.0941162, -6.4284191, 24.1388493, -32.9940414, 39.5225296
1: -10.4291992, 37.5969582, -7.4645872, 27.3031864, -37.7323837, 45.0615425
2: -10.9747181, 37.6314697, -7.9618182, 27.3991623, -38.3738785, 45.5932884
3: -16.3464375, 38.8504906, -11.7890568, 27.9809723, -44.3274078, 50.6395416
4: -17.2999840, 37.0951767, -12.4559155, 27.1424770, -44.4424591, 49.5510902

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B1

### Relational analysis result of IS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5730299, upper bound: 43.5833343
time: 0.53 seconds

## Relational analysis of IS_B1_A2_B2

### Relational analysis result of IS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5726613, upper bound: 43.5812720
time: 0.50 seconds

## BFS IS instance: IS_B2_B1

### Backsubstitution after applying IS history:
0: -7.5563655, 27.9216976, -8.8108072, 32.8912697, -40.4476357, 36.7325058
1: -8.8341284, 31.5343533, -10.3775148, 37.3341560, -46.1682854, 41.9118690
2: -9.3627281, 31.7484436, -10.9147587, 37.4205513, -46.7832756, 42.6632004
3: -13.8404074, 32.3905563, -16.2609444, 38.5590134, -52.3994179, 48.6515007
4: -14.3781385, 31.7164936, -17.1529713, 36.9485512, -51.3266907, 48.8694649

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B2_B1_A1

### Relational analysis result of IS_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4298590, upper bound: 43.3873342
time: 0.78 seconds

## Relational analysis of IS_B2_B1_A2

### Relational analysis result of IS_B2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3388956, upper bound: 43.3308378
time: 0.66 seconds

## BFS IS instance: IS_B2_B2

### Backsubstitution after applying IS history:
0: -7.5638342, 27.9483128, -8.8502941, 33.0782204, -40.6420555, 36.7986031
1: -8.8431463, 31.5652332, -10.4231472, 37.5786552, -46.4217987, 41.9883804
2: -9.3718910, 31.7789803, -10.9686108, 37.6130257, -46.9849167, 42.7475891
3: -13.8541670, 32.4234009, -16.3375454, 38.8310013, -52.6851578, 48.7609406
4: -14.3933516, 31.7455139, -17.2906876, 37.0763474, -51.4696999, 49.0361900

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B2_B2_A1

### Relational analysis result of IS_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4748527, upper bound: 43.4325963
time: 0.58 seconds

## Relational analysis of IS_B2_B2_A2

### Relational analysis result of IS_B2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3773967, upper bound: 43.3773968
time: 0.63 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.74 seconds
IS_B1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 3.74
Output dim: 3, lower bound: -43.5730299, upper bound: 43.5833343
IS_B1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 3.74
Output dim: 3, lower bound: -43.5726613, upper bound: 43.5812720
IS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 3.74
Output dim: 3, lower bound: -43.5730299, upper bound: 43.5833343
IS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 3.74
Output dim: 3, lower bound: -43.5726613, upper bound: 43.5812720
IS_B2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.74
Output dim: 3, lower bound: -43.4298590, upper bound: 43.3873342
IS_B2_B1_A2, status: Status.VERIFIED, split count: 3, time: 3.74
Output dim: 3, lower bound: -43.3388956, upper bound: 43.3308378
IS_B2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.74
Output dim: 3, lower bound: -43.4748527, upper bound: 43.4325963
IS_B2_B2_A2, status: Status.VERIFIED, split count: 3, time: 3.74
Output dim: 3, lower bound: -43.3773967, upper bound: 43.3773968

## BFS IS instance: IS_B1_A1_B1

### Backsubstitution after applying IS history:
0: -5.9971590, 22.6623802, -5.2551436, 20.1315994, -26.1287575, 27.9175243
1: -6.9453621, 25.6627369, -6.0467672, 22.8425674, -29.7879276, 31.7095032
2: -7.4216046, 25.7031651, -6.4991312, 22.7556610, -30.1772652, 32.2022896
3: -11.0082836, 26.2700863, -9.6570997, 23.3396664, -34.3479500, 35.9271851
4: -11.7066336, 25.3688774, -10.4297295, 22.2815781, -33.9882088, 35.7986069

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A1_B1_A1

### Relational analysis result of IS_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5837457, upper bound: 43.5837457
time: 0.52 seconds

## Relational analysis of IS_B1_A1_B1_A2

### Relational analysis result of IS_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5837457, upper bound: 43.5837457
time: 0.51 seconds

## BFS IS instance: IS_B1_A1_B2

### Backsubstitution after applying IS history:
0: -6.3298082, 23.7885094, -6.7437096, 25.4687443, -31.7985535, 30.5322189
1: -7.3447208, 26.9102478, -7.8409715, 28.8711681, -36.2158813, 34.7512169
2: -7.8425822, 26.9971638, -8.3597012, 28.9183235, -36.7609024, 35.3568611
3: -11.6079922, 27.5714645, -12.3940144, 29.6448441, -41.2528267, 39.9654770
4: -12.2795286, 26.7434692, -13.1956530, 28.4903889, -40.7699165, 39.9391212

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B2_A1

### Relational analysis result of IS_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5834206, upper bound: 43.5811446
time: 0.58 seconds

## Relational analysis of IS_B1_A1_B2_A2

### Relational analysis result of IS_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5808229, upper bound: 43.5808229
time: 0.61 seconds

## BFS IS instance: IS_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.3963518, 31.5576992, -5.2551436, 20.1315994, -28.5279503, 36.8128433
1: -9.8746014, 35.8924103, -6.0467672, 22.8425674, -32.7171707, 41.9391785
2: -10.4066305, 35.8486519, -6.4991312, 22.7556610, -33.1622925, 42.3477821
3: -15.5131435, 37.0552979, -9.6570997, 23.3396664, -38.8528061, 46.7123947
4: -16.5106926, 35.2320938, -10.4297295, 22.2815781, -38.7922630, 45.6618195

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_B1_A1

### Relational analysis result of IS_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5635350, upper bound: 43.5612089
time: 0.49 seconds

## Relational analysis of IS_B1_A2_B1_A2

### Relational analysis result of IS_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5448007, upper bound: 43.5584100
time: 0.53 seconds

## BFS IS instance: IS_B1_A2_B2

### Backsubstitution after applying IS history:
0: -8.7644262, 32.7622604, -6.7437096, 25.4687443, -34.2331696, 39.5059624
1: -10.3167582, 37.2191505, -7.8409715, 28.8711681, -39.1879272, 45.0601234
2: -10.8648520, 37.2530975, -8.3597012, 28.9183235, -39.7831726, 45.6127930
3: -16.1762123, 38.4589233, -12.3940144, 29.6448441, -45.8210526, 50.8529358
4: -17.1328182, 36.7215614, -13.1956530, 28.4903889, -45.6232071, 49.9172134

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_B2_A1

### Relational analysis result of IS_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5622764, upper bound: 43.5580816
time: 0.50 seconds

## Relational analysis of IS_B1_A2_B2_A2

### Relational analysis result of IS_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5442409, upper bound: 43.5553388
time: 0.48 seconds

## BFS IS instance: IS_B2_B1_A1

### Backsubstitution after applying IS history:
0: -7.5138865, 27.7762203, -8.8108072, 32.8912697, -40.4051552, 36.5870285
1: -8.7819996, 31.3696842, -10.3775148, 37.3341560, -46.1161575, 41.7471962
2: -9.3101006, 31.5812798, -10.9147587, 37.4205513, -46.7306480, 42.4960403
3: -13.7619476, 32.2190323, -16.2609444, 38.5590134, -52.3209572, 48.4799767
4: -14.3028746, 31.5446301, -17.1529713, 36.9485512, -51.2514267, 48.6975975

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_B1_A1_A1

### Relational analysis result of IS_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4100490, upper bound: 43.3797757
time: 0.67 seconds

## Relational analysis of IS_B2_B1_A1_A2

### Relational analysis result of IS_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4298288, upper bound: 43.3872458
time: 0.63 seconds

## BFS IS instance: IS_B2_B2_A1

### Backsubstitution after applying IS history:
0: -7.5213609, 27.8029251, -8.8502941, 33.0782204, -40.5995827, 36.6532211
1: -8.7910242, 31.4006615, -10.4231472, 37.5786552, -46.3696785, 41.8238068
2: -9.3192673, 31.6119099, -10.9686108, 37.6130257, -46.9322929, 42.5805206
3: -13.7757196, 32.2519798, -16.3375454, 38.8310013, -52.6067123, 48.5895233
4: -14.3181105, 31.5737057, -17.2906876, 37.0763474, -51.3944588, 48.8643951

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_B2_A1_B1

### Relational analysis result of IS_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4730515, upper bound: 43.4308246
time: 0.66 seconds

## Relational analysis of IS_B2_B2_A1_B2

### Relational analysis result of IS_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4699983, upper bound: 43.4309311
time: 0.75 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.84 seconds
IS_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 4.84
Output dim: 3, lower bound: -43.5837457, upper bound: 43.5837457
IS_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 4.84
Output dim: 3, lower bound: -43.5837457, upper bound: 43.5837457
IS_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.84
Output dim: 3, lower bound: -43.5834206, upper bound: 43.5811446
IS_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.84
Output dim: 3, lower bound: -43.5808229, upper bound: 43.5808229
IS_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 4.84
Output dim: 3, lower bound: -43.5635350, upper bound: 43.5612089
IS_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 4.84
Output dim: 3, lower bound: -43.5448007, upper bound: 43.5584100
IS_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.84
Output dim: 3, lower bound: -43.5622764, upper bound: 43.5580816
IS_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.84
Output dim: 3, lower bound: -43.5442409, upper bound: 43.5553388
IS_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 4.84
Output dim: 3, lower bound: -43.4100490, upper bound: 43.3797757
IS_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 4.84
Output dim: 3, lower bound: -43.4298288, upper bound: 43.3872458
IS_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.84
Output dim: 3, lower bound: -43.4730515, upper bound: 43.4308246
IS_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.84
Output dim: 3, lower bound: -43.4699983, upper bound: 43.4309311

## BFS IS instance: IS_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -5.2551436, 20.1315994, -5.2551436, 20.1315994, -25.3867397, 25.3867416
1: -6.0467672, 22.8425674, -6.0467672, 22.8425674, -28.8893318, 28.8893318
2: -6.4991312, 22.7556610, -6.4991312, 22.7556610, -29.2547913, 29.2547913
3: -9.6570997, 23.3396664, -9.6570997, 23.3396664, -32.9967651, 32.9967651
4: -10.4297295, 22.2815781, -10.4297295, 22.2815781, -32.7113037, 32.7113037

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B1_A1_A1

### Relational analysis result of IS_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5602982, upper bound: 43.5354064
time: 0.49 seconds

## Relational analysis of IS_B1_A1_B1_A1_A2

### Relational analysis result of IS_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5320412, upper bound: 43.5320412
time: 0.56 seconds

## BFS IS instance: IS_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -6.7333407, 25.4273777, -5.2551436, 20.1315994, -26.8649368, 30.6825199
1: -7.8258891, 28.8233337, -6.0467672, 22.8425674, -30.6684570, 34.8701019
2: -8.3474169, 28.8704529, -6.4991312, 22.7556610, -31.1030769, 35.3695831
3: -12.3706512, 29.5924473, -9.6570997, 23.3396664, -35.7103195, 39.2495422
4: -13.1759529, 28.4422913, -10.4297295, 22.2815781, -35.4575272, 38.8720169

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B1_A2_A1

### Relational analysis result of IS_B1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5602982, upper bound: 43.5354064
time: 0.70 seconds

## Relational analysis of IS_B1_A1_B1_A2_A2

### Relational analysis result of IS_B1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5320412, upper bound: 43.5352627
time: 0.52 seconds

## BFS IS instance: IS_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -5.9077616, 22.3436146, -6.6097879, 25.0126324, -30.9203892, 28.9534016
1: -6.8350163, 25.3080196, -7.6782727, 28.3685532, -35.2035637, 32.9862862
2: -7.3177242, 25.3283272, -8.1945143, 28.3897915, -35.7075157, 33.5228424
3: -10.8386698, 25.9052677, -12.1477814, 29.1150246, -39.9536934, 38.0530472
4: -11.5513802, 24.9896393, -12.9643135, 27.9381161, -39.4894943, 37.9539528

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B2_A1_B1

### Relational analysis result of IS_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5602372, upper bound: 43.5697475
time: 0.59 seconds

## Relational analysis of IS_B1_A1_B2_A1_B2

### Relational analysis result of IS_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5578907, upper bound: 43.5547360
time: 0.59 seconds

## BFS IS instance: IS_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -6.4782391, 24.4436932, -6.6892424, 25.2815056, -31.7597446, 31.1329346
1: -7.5056667, 27.7153263, -7.7734771, 28.6642914, -36.1699600, 35.4888039
2: -8.0250196, 27.7291431, -8.2929592, 28.6993961, -36.7244110, 36.0221024
3: -11.8727446, 28.4061203, -12.2910070, 29.4259758, -41.2987137, 40.6971245
4: -12.6591158, 27.3279667, -13.0995502, 28.2627583, -40.9218712, 40.4275169

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B2_A2_B1

### Relational analysis result of IS_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5063387, upper bound: 43.5683536
time: 0.68 seconds

## Relational analysis of IS_B1_A1_B2_A2_B2

### Relational analysis result of IS_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4988436, upper bound: 43.5539461
time: 0.65 seconds

## BFS IS instance: IS_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -8.1724939, 30.7763424, -5.2044840, 19.9554691, -28.1279602, 35.9808273
1: -9.5991011, 35.0056190, -5.9851756, 22.6442890, -32.2433891, 40.9907913
2: -10.1295862, 34.9470978, -6.4362683, 22.5512848, -32.6808701, 41.3833656
3: -15.0948954, 36.1287003, -9.5634146, 23.1330853, -38.2279816, 45.6921158
4: -16.1004639, 34.3230057, -10.3376198, 22.0733128, -38.1737709, 44.6606255

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A2_B1_A1_B1

### Relational analysis result of IS_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5472787, upper bound: 43.5501204
time: 0.52 seconds

## Relational analysis of IS_B1_A2_B1_A1_B2

### Relational analysis result of IS_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5371676, upper bound: 43.5224118
time: 0.52 seconds

## BFS IS instance: IS_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -8.3579378, 31.4964809, -5.1323190, 19.7321587, -28.0900955, 36.6287994
1: -9.8361168, 35.8252487, -5.8967485, 22.3975353, -32.2336464, 41.7219963
2: -10.3633509, 35.7737656, -6.3479762, 22.2902794, -32.6536293, 42.1217422
3: -15.4720125, 37.0144997, -9.4361601, 22.8788242, -38.3508377, 46.4506607
4: -16.4907722, 35.1249466, -10.2217674, 21.7947598, -38.2855263, 45.3467140

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A2_B1_A2_B1

### Relational analysis result of IS_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5189224, upper bound: 43.5423014
time: 0.61 seconds

## Relational analysis of IS_B1_A2_B1_A2_B2

### Relational analysis result of IS_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5168084, upper bound: 43.5193028
time: 0.54 seconds

## BFS IS instance: IS_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -8.5432882, 31.9851837, -6.6882963, 25.2769985, -33.8202858, 38.6734810
1: -10.0449066, 36.3353119, -7.7731462, 28.6543427, -38.6992455, 44.1084557
2: -10.5902824, 36.3567581, -8.2909880, 28.6962204, -39.2865028, 44.6477432
3: -15.7634697, 37.5370903, -12.2910128, 29.4177837, -45.1812515, 49.8281021
4: -16.7245216, 35.8202515, -13.0947399, 28.2649918, -44.9895134, 48.9149933

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B2_A1_B1

### Relational analysis result of IS_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5622764, upper bound: 43.5580816
time: 0.73 seconds

## Relational analysis of IS_B1_A2_B2_A1_B2

### Relational analysis result of IS_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5608914, upper bound: 43.5541237
time: 0.73 seconds

## BFS IS instance: IS_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -8.7240391, 32.6896935, -6.6064835, 25.0223045, -33.7463455, 39.2961769
1: -10.2752285, 37.1534958, -7.6728826, 28.3738327, -38.6490517, 44.8263779
2: -10.8187466, 37.1651230, -8.1909695, 28.3996124, -39.2183533, 45.3560944
3: -16.1305790, 38.4040985, -12.1459389, 29.1224861, -45.2530632, 50.5500336
4: -17.1082592, 36.5997124, -12.9611845, 27.9546185, -45.0628700, 49.5608978

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B2_A2_B1

### Relational analysis result of IS_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5442409, upper bound: 43.5553388
time: 0.50 seconds

## Relational analysis of IS_B1_A2_B2_A2_B2

### Relational analysis result of IS_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5434515, upper bound: 43.5513943
time: 0.54 seconds

## BFS IS instance: IS_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -7.3080020, 27.0490398, -8.7534122, 32.6899948, -39.9979935, 35.8024406
1: -8.5298319, 30.5445614, -10.3068695, 37.1048012, -45.6346321, 40.8514328
2: -9.0536633, 30.7389679, -10.8432837, 37.1878510, -46.2415161, 41.5822411
3: -13.3772430, 31.3539124, -16.1538601, 38.3195305, -51.6967735, 47.5077744
4: -13.9186869, 30.7033119, -17.0465107, 36.7147522, -50.6334381, 47.7498245

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_B2_B1_A1_A1_A1

### Relational analysis result of IS_B2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4100277, upper bound: 43.3797757
time: 0.80 seconds

## Relational analysis of IS_B2_B1_A1_A1_A2

### Relational analysis result of IS_B2_B1_A1_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3677118, upper bound: 43.3619789
time: 0.77 seconds

## BFS IS instance: IS_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -7.4690423, 27.6785583, -8.6655560, 32.4175568, -39.8865967, 36.3441162
1: -8.7325840, 31.2721939, -10.1995068, 36.8008499, -45.5334320, 41.4717026
2: -9.2551584, 31.4667854, -10.7356243, 36.8696785, -46.1248322, 42.2024040
3: -13.7055960, 32.1299934, -15.9972591, 38.0020676, -51.7076645, 48.1272469
4: -14.2619524, 31.3978920, -16.9016151, 36.3822899, -50.6442375, 48.2995071

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B2_B1_A1_A2_B1

### Relational analysis result of IS_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4274932, upper bound: 43.3871679
time: 0.75 seconds

## Relational analysis of IS_B2_B1_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_B2_B1_A1_A2_A1

### Relational analysis result of IS_B2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4124803, upper bound: 43.3790408
time: 0.60 seconds

## Relational analysis of IS_B2_B1_A1_A2_A2

### Relational analysis result of IS_B2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4298288, upper bound: 43.3872458
time: 0.84 seconds

## BFS IS instance: IS_B2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -7.4681377, 27.6152573, -8.6274080, 32.2948112, -39.7629395, 36.2426643
1: -8.7258492, 31.1876087, -10.1490211, 36.6875229, -45.4133720, 41.3366318
2: -9.2530870, 31.3945141, -10.6915836, 36.7092934, -45.9623795, 42.0860977
3: -13.6762562, 32.0286407, -15.9213505, 37.9011421, -51.5773964, 47.9499893
4: -14.2188492, 31.3567123, -16.8786182, 36.1675415, -50.3863907, 48.2353210

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_B2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B2_B2_A1_B1_B1

### Relational analysis result of IS_B2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4504243, upper bound: 43.4269375
time: 0.72 seconds

## Relational analysis of IS_B2_B2_A1_B1_B2

### Relational analysis result of IS_B2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4631598, upper bound: 43.4248997
time: 0.60 seconds

## BFS IS instance: IS_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -7.3826513, 27.3448105, -8.8129749, 33.0192146, -40.4018669, 36.1577835
1: -8.6198082, 30.8852673, -10.3859720, 37.5279694, -46.1477737, 41.2712402
2: -9.1478195, 31.0799561, -10.9260187, 37.5408058, -46.6886215, 42.0059662
3: -13.5228806, 31.7135506, -16.2985821, 38.7907181, -52.3135948, 48.0121307
4: -14.0757923, 31.0303078, -17.2704239, 36.9720306, -51.0478210, 48.3007317

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B2_B2_A1_B2_B1

### Relational analysis result of IS_B2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4504254, upper bound: 43.4267476
time: 0.75 seconds

## Relational analysis of IS_B2_B2_A1_B2_B2

### Relational analysis result of IS_B2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4592822, upper bound: 43.4189035
time: 0.75 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 5.94 seconds
IS_B1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 5.94
Output dim: 3, lower bound: -43.5602982, upper bound: 43.5354064
IS_B1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 5.94
Output dim: 3, lower bound: -43.5320412, upper bound: 43.5320412
IS_B1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 5.94
Output dim: 3, lower bound: -43.5602982, upper bound: 43.5354064
IS_B1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 5.94
Output dim: 3, lower bound: -43.5320412, upper bound: 43.5352627
IS_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 5.94
Output dim: 3, lower bound: -43.5602372, upper bound: 43.5697475
IS_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 5.94
Output dim: 3, lower bound: -43.5578907, upper bound: 43.5547360
IS_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.94
Output dim: 3, lower bound: -43.5063387, upper bound: 43.5683536
IS_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.94
Output dim: 3, lower bound: -43.4988436, upper bound: 43.5539461
IS_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 5.94
Output dim: 3, lower bound: -43.5472787, upper bound: 43.5501204
IS_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 5.94
Output dim: 3, lower bound: -43.5371676, upper bound: 43.5224118
IS_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.94
Output dim: 3, lower bound: -43.5189224, upper bound: 43.5423014
IS_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.94
Output dim: 3, lower bound: -43.5168084, upper bound: 43.5193028
IS_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 5.94
Output dim: 3, lower bound: -43.5622764, upper bound: 43.5580816
IS_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 5.94
Output dim: 3, lower bound: -43.5608914, upper bound: 43.5541237
IS_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.94
Output dim: 3, lower bound: -43.5442409, upper bound: 43.5553388
IS_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.94
Output dim: 3, lower bound: -43.5434515, upper bound: 43.5513943
IS_B2_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 5.94
Output dim: 3, lower bound: -43.4100277, upper bound: 43.3797757
IS_B2_B1_A1_A1_A2, status: Status.VERIFIED, split count: 5, time: 5.94
Output dim: 3, lower bound: -43.3677118, upper bound: 43.3619789
IS_B2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 5.94
Output dim: 3, lower bound: -43.4124803, upper bound: 43.3790408
IS_B2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 5.94
Output dim: 3, lower bound: -43.4298288, upper bound: 43.3872458
IS_B2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 5.94
Output dim: 3, lower bound: -43.4504243, upper bound: 43.4269375
IS_B2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 5.94
Output dim: 3, lower bound: -43.4631598, upper bound: 43.4248997
IS_B2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 5.94
Output dim: 3, lower bound: -43.4504254, upper bound: 43.4267476
IS_B2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 5.94
Output dim: 3, lower bound: -43.4592822, upper bound: 43.4189035

## BFS IS instance: IS_B1_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -4.9090037, 18.9296074, -5.1542945, 19.7820091, -24.6910114, 24.0839024
1: -5.6224365, 21.5094490, -5.9232106, 22.4548607, -28.0772972, 27.4326591
2: -6.0714874, 21.3573895, -6.3742619, 22.3491707, -28.4206581, 27.7316513
3: -9.0102825, 21.9570026, -9.4689646, 22.9373608, -31.9476433, 31.4259682
4: -9.8312683, 20.8039608, -10.2557783, 21.8470917, -31.6783600, 31.0597382

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B1_A1_A1_B1

### Relational analysis result of IS_B1_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5089712, upper bound: 43.5175398
time: 0.77 seconds

## Relational analysis of IS_B1_A1_B1_A1_A1_B2

### Relational analysis result of IS_B1_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4987033, upper bound: 43.4661222
time: 0.57 seconds

## BFS IS instance: IS_B1_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -5.4008245, 20.7929802, -5.1827059, 19.8774414, -25.2782669, 25.9756851
1: -6.2052450, 23.6457863, -5.9566588, 22.5604248, -28.7656708, 29.6024437
2: -6.6841125, 23.4871082, -6.4098415, 22.4578743, -29.1419868, 29.8969498
3: -9.9212875, 24.1726971, -9.5194597, 23.0474834, -32.9687729, 33.6921577
4: -10.8100157, 22.8875217, -10.3017712, 21.9701290, -32.7801437, 33.1892929

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B1_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B1_A1_A2_B1

### Relational analysis result of IS_B1_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4636996, upper bound: 43.4835076
time: 0.54 seconds

## Relational analysis of IS_B1_A1_B1_A1_A2_B2

### Relational analysis result of IS_B1_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4602134, upper bound: 43.4602134
time: 0.53 seconds

## BFS IS instance: IS_B1_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -6.3007426, 23.9628239, -5.1542945, 19.7820091, -26.0827522, 29.1171188
1: -7.3037090, 27.2083588, -5.9232106, 22.4548607, -29.7585697, 33.1315689
2: -7.8114362, 27.1696320, -6.3742619, 22.3491707, -30.1606045, 33.5438843
3: -11.5799856, 27.9001198, -9.4689646, 22.9373608, -34.5173454, 37.3690834
4: -12.4291039, 26.6609402, -10.2557783, 21.8470917, -34.2761917, 36.9167175

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B1_A2_A1_A1

### Relational analysis result of IS_B1_A1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4954708, upper bound: 43.4651732
time: 0.59 seconds

## Relational analysis of IS_B1_A1_B1_A2_A1_A2

### Relational analysis result of IS_B1_A1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4987957, upper bound: 43.4656737
time: 0.53 seconds

## BFS IS instance: IS_B1_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -6.9010291, 26.1635189, -5.1827059, 19.8774414, -26.7784710, 31.3462257
1: -8.0122547, 29.7247810, -5.9566588, 22.5604248, -30.5726795, 35.6814384
2: -8.5548077, 29.6831627, -6.4098415, 22.4578743, -31.0126781, 36.0930023
3: -12.6730833, 30.5168724, -9.5194597, 23.0474834, -35.7205658, 40.0363312
4: -13.5932789, 29.1214905, -10.3017712, 21.9701290, -35.5634003, 39.4232597

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B1_A2_A2_A1

### Relational analysis result of IS_B1_A1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4967081, upper bound: 43.4651026
time: 0.62 seconds

## Relational analysis of IS_B1_A1_B1_A2_A2_A2

### Relational analysis result of IS_B1_A1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4988436, upper bound: 43.4655149
time: 0.65 seconds

## BFS IS instance: IS_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -5.8556719, 22.1618557, -6.3952656, 24.2670021, -30.1226730, 28.5571213
1: -6.7714615, 25.1029339, -7.4161768, 27.5266991, -34.2981529, 32.5191116
2: -7.2528958, 25.1175575, -7.9277906, 27.5264244, -34.7793198, 33.0453415
3: -10.7420044, 25.6914043, -11.7486191, 28.2345562, -38.9765587, 37.4400177
4: -11.4566231, 24.7761974, -12.5724287, 27.0626621, -38.5192795, 37.3486252

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5578907, upper bound: 43.5547360
time: 0.60 seconds

## Relational analysis of IS_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5578907, upper bound: 43.5547360
time: 0.62 seconds

## BFS IS instance: IS_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -5.7773042, 21.9188652, -6.5764704, 24.9620037, -30.7393074, 28.4953346
1: -6.6759973, 24.8326740, -7.6452065, 28.3145504, -34.9905472, 32.4778671
2: -7.1565709, 24.8324223, -8.1564093, 28.3270569, -35.4836197, 32.9888306
3: -10.6052732, 25.4123802, -12.1140919, 29.0850220, -39.6902924, 37.5264702
4: -11.3297005, 24.4791241, -12.9517593, 27.8431149, -39.1728058, 37.4308853

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5460020, upper bound: 43.5230990
time: 0.71 seconds

## Relational analysis of IS_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4987957, upper bound: 43.5209338
time: 0.62 seconds

## BFS IS instance: IS_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -6.4250679, 24.2584152, -6.4747691, 24.5368595, -30.9619274, 30.7331848
1: -7.4410319, 27.5062218, -7.5113759, 27.8232212, -35.2642479, 35.0175934
2: -7.9588208, 27.5148869, -8.0265694, 27.8370266, -35.7958488, 35.5414581
3: -11.7745409, 28.1878204, -11.8921070, 28.5450668, -40.3195992, 40.0799217
4: -12.5620794, 27.1107731, -12.7080669, 27.3883533, -39.9504318, 39.8188400

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4967081, upper bound: 43.5539461
time: 0.63 seconds

## Relational analysis of IS_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4967081, upper bound: 43.5539461
time: 0.66 seconds

## BFS IS instance: IS_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -6.3432517, 24.0028362, -6.6563573, 25.2314167, -31.5746689, 30.6591930
1: -7.3412609, 27.2230244, -7.7405338, 28.6108170, -35.9520721, 34.9635506
2: -7.8583002, 27.2166882, -8.2557087, 28.6371841, -36.4954796, 35.4723969
3: -11.6311893, 27.8948269, -12.2577505, 29.3995667, -41.0307541, 40.1525688
4: -12.4288597, 26.8004417, -13.0879383, 28.1688404, -40.5976944, 39.8883820

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5437599, upper bound: 43.5224424
time: 0.57 seconds

## Relational analysis of IS_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4988436, upper bound: 43.5206615
time: 0.56 seconds

## BFS IS instance: IS_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -8.1031284, 30.5289993, -4.9878254, 19.1570854, -27.2602139, 35.5168228
1: -9.5150394, 34.7205734, -5.7248712, 21.7320538, -31.2470932, 40.4454422
2: -10.0438499, 34.6635017, -6.1675997, 21.6361637, -31.6800137, 40.8311005
3: -14.9658184, 35.8350639, -9.1591549, 22.1921272, -37.1579437, 44.9942169
4: -15.9703531, 34.0427284, -9.9251528, 21.1796646, -37.1500053, 43.9678764

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_A1_B1_B1

### Relational analysis result of IS_B1_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5472787, upper bound: 43.5501204
time: 0.46 seconds

## Relational analysis of IS_B1_A2_B1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_B1_A1_B1_B1

### Relational analysis result of IS_B1_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5472787, upper bound: 43.5501204
time: 0.48 seconds

## Relational analysis of IS_B1_A2_B1_A1_B1_B2

### Relational analysis result of IS_B1_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5472787, upper bound: 43.5501204
time: 0.53 seconds

## BFS IS instance: IS_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -8.0324631, 30.3083324, -5.5183177, 20.8902836, -28.9227467, 35.8266487
1: -9.4280415, 34.4847527, -6.3855066, 23.6601238, -33.0881577, 40.8702583
2: -9.9565392, 34.4051971, -6.8211341, 23.6667480, -33.6232834, 41.2263298
3: -14.8363256, 35.5742226, -10.1468658, 24.2048683, -39.0411949, 45.7210884
4: -15.8469849, 33.7704697, -10.8301888, 23.3283501, -39.1753273, 44.6006546

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_A1_B2_B1

### Relational analysis result of IS_B1_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5371676, upper bound: 43.5222874
time: 0.57 seconds

## Relational analysis of IS_B1_A2_B1_A1_B2_B2

### Relational analysis result of IS_B1_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4958278, upper bound: 43.4630661
time: 0.62 seconds

## BFS IS instance: IS_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.2901173, 31.2559967, -4.9201732, 18.9510899, -27.2412014, 36.1761703
1: -9.7540836, 35.5480652, -5.6422067, 21.5048122, -31.2588940, 41.1902733
2: -10.2795000, 35.4980049, -6.0848842, 21.3950310, -31.6745281, 41.5828896
3: -15.3461361, 36.7288895, -9.0408974, 21.9581833, -37.3043213, 45.7697868
4: -16.3637924, 34.8522797, -9.8178349, 20.9202347, -37.2840271, 44.6701126

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_A2_B1_B1

### Relational analysis result of IS_B1_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5189224, upper bound: 43.5423014
time: 0.58 seconds

## Relational analysis of IS_B1_A2_B1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_B1_A2_B1_B1

### Relational analysis result of IS_B1_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5189159, upper bound: 43.5393824
time: 0.55 seconds

## Relational analysis of IS_B1_A2_B1_A2_B1_B2

### Relational analysis result of IS_B1_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5189159, upper bound: 43.5393824
time: 0.60 seconds

## BFS IS instance: IS_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -8.2245684, 31.0448208, -5.4418221, 20.6549301, -28.8794975, 36.4866409
1: -9.6728926, 35.3218307, -6.2911820, 23.3997002, -33.0725899, 41.6130104
2: -10.1983147, 35.2511978, -6.7275152, 23.3873272, -33.5856361, 41.9787140
3: -15.2245588, 36.4756432, -10.0107956, 23.9362125, -39.1607704, 46.4864349
4: -16.2473602, 34.5945969, -10.7071648, 23.0349045, -39.2822533, 45.3017616

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_A2_B2_B1

### Relational analysis result of IS_B1_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5168084, upper bound: 43.5191840
time: 0.64 seconds

## Relational analysis of IS_B1_A2_B1_A2_B2_B2

### Relational analysis result of IS_B1_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4971066, upper bound: 43.4631742
time: 0.75 seconds

## BFS IS instance: IS_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -8.4050350, 31.5180111, -6.2460589, 23.7731342, -32.1781693, 37.7640686
1: -9.8765526, 35.8179436, -7.2373176, 26.9942951, -36.8708420, 43.0552597
2: -10.4202394, 35.8138428, -7.7434101, 26.9496174, -37.3698578, 43.5572510
3: -15.5090389, 36.9930267, -11.4789524, 27.6765995, -43.1856384, 48.4719696
4: -16.4872856, 35.2513084, -12.3294935, 26.4377308, -42.9250183, 47.5808029

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A2_B2_A1_B1_B1

### Relational analysis result of IS_B1_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5462759, upper bound: 43.5484287
time: 0.51 seconds

## Relational analysis of IS_B1_A2_B2_A1_B1_B2

### Relational analysis result of IS_B1_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5361042, upper bound: 43.5214457
time: 0.51 seconds

## BFS IS instance: IS_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -8.4918842, 31.8048439, -6.8591313, 26.0269108, -34.5187912, 38.6639748
1: -9.9813375, 36.1348000, -7.9659610, 29.5727253, -39.5540619, 44.1007538
2: -10.5270882, 36.1463470, -8.5026236, 29.5251827, -40.0522614, 44.6489716
3: -15.6661997, 37.3270493, -12.6037149, 30.3615913, -46.0277786, 49.9307632
4: -16.6338463, 35.6019821, -13.5196590, 28.9608479, -45.5946960, 49.1216393

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A2_B2_A1_B2_B1

### Relational analysis result of IS_B1_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5454465, upper bound: 43.5456144
time: 0.57 seconds

## Relational analysis of IS_B1_A2_B2_A1_B2_B2

### Relational analysis result of IS_B1_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5354970, upper bound: 43.5208401
time: 0.52 seconds

## BFS IS instance: IS_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -8.5873795, 32.2296333, -6.1703930, 23.5431023, -32.1304817, 38.4000244
1: -10.1090441, 36.6385803, -7.1453052, 26.7390327, -36.8480682, 43.7838821
2: -10.6503849, 36.6303787, -7.6505866, 26.6787853, -37.3291702, 44.2809639
3: -15.8793974, 37.8682442, -11.3473091, 27.4126663, -43.2920647, 49.2155533
4: -16.8738613, 36.0378990, -12.2087479, 26.1536083, -43.0274658, 48.2466469

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A2_B2_A2_B1_B1

### Relational analysis result of IS_B1_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5185740, upper bound: 43.5410342
time: 0.56 seconds

## Relational analysis of IS_B1_A2_B2_A2_B1_B2

### Relational analysis result of IS_B1_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5165910, upper bound: 43.5184723
time: 0.61 seconds

## BFS IS instance: IS_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.6748075, 32.5160522, -6.7757201, 25.7666759, -34.4414825, 39.2917709
1: -10.2141819, 36.9595108, -7.8640661, 29.2862854, -39.5004578, 44.8235703
2: -10.7580252, 36.9624367, -8.4003229, 29.2225342, -39.9805565, 45.3627586
3: -16.0369873, 38.2021713, -12.4558735, 30.0627480, -46.0997238, 50.6580429
4: -17.0214367, 36.3888206, -13.3831587, 28.6441879, -45.6656265, 49.7719803

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A2_B2_A2_B2_B1

### Relational analysis result of IS_B1_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5180794, upper bound: 43.5387679
time: 0.67 seconds

## Relational analysis of IS_B1_A2_B2_A2_B2_B2

### Relational analysis result of IS_B1_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5163758, upper bound: 43.5179210
time: 0.85 seconds

## BFS IS instance: IS_B2_B1_A1_A1_A1

### Backsubstitution after applying IS history:
0: -7.2449203, 26.8429394, -8.7534122, 32.6899948, -39.9349136, 35.5963516
1: -8.4528065, 30.3117523, -10.3068695, 37.1048012, -45.5576096, 40.6186218
2: -8.9758472, 30.5009518, -10.8432837, 37.1878510, -46.1636887, 41.3442345
3: -13.2627583, 31.1109352, -16.1538601, 38.3195305, -51.5822868, 47.2647934
4: -13.8105469, 30.4544506, -17.0465107, 36.7147522, -50.5252991, 47.5009613

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_B2_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B2_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B2_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_B2_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_B2_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_B2_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B2_B1_A1_A1_A1_B1

### Relational analysis result of IS_B2_B1_A1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3773875, upper bound: 43.3682946
time: 0.68 seconds

## Relational analysis of IS_B2_B1_A1_A1_A1_B2

### Relational analysis result of IS_B2_B1_A1_A1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3773875, upper bound: 43.3797757
time: 0.61 seconds

## BFS IS instance: IS_B2_B1_A1_A2_A1

### Backsubstitution after applying IS history:
0: -7.3315086, 27.1909866, -8.6316118, 32.2978134, -39.6293221, 35.8225937
1: -8.5624590, 30.7166882, -10.1575146, 36.6643066, -45.2267647, 40.8741875
2: -9.0850277, 30.9044437, -10.6935091, 36.7318954, -45.8169212, 41.5979462
3: -13.4453793, 31.5479813, -15.9333324, 37.8594933, -51.3048668, 47.4813118
4: -14.0036793, 30.8388729, -16.8381939, 36.2440948, -50.2477684, 47.6770668

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_B1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_B2_B1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_B1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_B1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_B1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_B1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B2_B1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B2_B1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_B1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_B1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_B2_B1_A1_A2_A1_A1

### Relational analysis result of IS_B2_B1_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4124803, upper bound: 43.3790408
time: 0.63 seconds

## Relational analysis of IS_B2_B1_A1_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_B2_B1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_B2_B1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_B1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B2_B1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 27
type: B, layer: 3, pos: 27
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 2
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 13
type: A, layer: 3, pos: 42
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 42
type: A, layer: 3, pos: 41

Time for candidate selection: 7.92 seconds

### Candidate
type: A, layer: 3, pos: 27

## Relational analysis of IS_B2_B1_A1_A2_A1_A1

### Relational analysis result of IS_B2_B1_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4124803, upper bound: 43.3790408
time: 0.58 seconds

## Relational analysis of IS_B2_B1_A1_A2_A1_A2

### Relational analysis result of IS_B2_B1_A1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4110513, upper bound: 43.3781633
time: 0.57 seconds

## BFS IS instance: IS_B2_B1_A1_A2_A2

### Backsubstitution after applying IS history:
0: -7.3555131, 27.2373981, -8.5003548, 31.8407230, -39.1962318, 35.7377510
1: -8.5792885, 30.8038063, -9.9913034, 36.1615143, -44.7408028, 40.7951012
2: -9.1106091, 30.9517536, -10.5287762, 36.2015305, -45.3121414, 41.4805222
3: -13.4660368, 31.6694622, -15.6809425, 37.3211021, -50.7871361, 47.3504028
4: -14.0861320, 30.8596992, -16.5955925, 35.6993408, -49.7854729, 47.4552917

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_B1_A1_A2_A2_A1

### Relational analysis result of IS_B2_B1_A1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3315254, upper bound: 43.3434313
time: 0.94 seconds

## Relational analysis of IS_B2_B1_A1_A2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_B1_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_B2_B1_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_B1_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B2_B1_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_B1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B2_B1_A1_A2_A2_B1

### Relational analysis result of IS_B2_B1_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4274932, upper bound: 43.3871679
time: 0.55 seconds

## Relational analysis of IS_B2_B1_A1_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_B1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_B1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_B1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_B2_B1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_B2_B1_A1_A2_A2_A1

### Relational analysis result of IS_B2_B1_A1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4298288, upper bound: 43.3872458
time: 0.64 seconds

## Relational analysis of IS_B2_B1_A1_A2_A2_A2

### Relational analysis result of IS_B2_B1_A1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3930732, upper bound: 43.3717053
time: 0.68 seconds

## BFS IS instance: IS_B2_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -7.4013295, 27.3746719, -8.3476353, 31.2967281, -38.6980591, 35.7223053
1: -8.6445913, 30.9119186, -9.8094931, 35.5442276, -44.1888161, 40.7214127
2: -9.1706028, 31.1187935, -10.3463326, 35.5648613, -44.7354660, 41.4651260
3: -13.5514450, 31.7428379, -15.4003344, 36.7186165, -50.2700539, 47.1431732
4: -14.0929222, 31.0864506, -16.3566322, 35.0365715, -49.1294937, 47.4430847

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_B2_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_B2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_B2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_B2_A1_B1_B1_A1

### Relational analysis result of IS_B2_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4504243, upper bound: 43.4269375
time: 0.87 seconds

## Relational analysis of IS_B2_B2_A1_B1_B1_A2

### Relational analysis result of IS_B2_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4504243, upper bound: 43.4269375
time: 0.60 seconds

## BFS IS instance: IS_B2_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -7.3376789, 27.1688404, -8.9311800, 33.2119255, -40.5496025, 36.1000061
1: -8.5647011, 30.6798420, -10.5458260, 37.7010880, -46.2657890, 41.2256699
2: -9.0911617, 30.8794117, -11.0667562, 37.8067436, -46.8978996, 41.9461670
3: -13.4327955, 31.4952374, -16.5069580, 38.9802055, -52.4129982, 48.0021973
4: -13.9795437, 30.8359890, -17.3660297, 37.3945236, -51.3740654, 48.2020149

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_B2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_B2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_B2_B2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_B2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_B2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_B2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_B2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_B2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_B2_B2_A1_B1_B2_A1

### Relational analysis result of IS_B2_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4631408, upper bound: 43.4248997
time: 0.87 seconds

## Relational analysis of IS_B2_B2_A1_B1_B2_A2

### Relational analysis result of IS_B2_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4167397, upper bound: 43.4050184
time: 0.64 seconds

## BFS IS instance: IS_B2_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -7.3159246, 27.1046162, -8.5394516, 32.0485039, -39.3644295, 35.6440620
1: -8.5386639, 30.6100616, -10.0546646, 36.4156799, -44.9543457, 40.6647263
2: -9.0654726, 30.8046741, -10.5878363, 36.4272728, -45.4927406, 41.3925095
3: -13.3982935, 31.4283066, -15.7905474, 37.6386337, -51.0369263, 47.2188530
4: -13.9501410, 30.7604446, -16.7589092, 35.8717308, -49.8218727, 47.5193520

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_B2_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_B2_A1_B2_B1_A1

### Relational analysis result of IS_B2_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4504243, upper bound: 43.4267476
time: 0.60 seconds

## Relational analysis of IS_B2_B2_A1_B2_B1_A2

### Relational analysis result of IS_B2_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4504243, upper bound: 43.4267476
time: 0.63 seconds

## BFS IS instance: IS_B2_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -7.2523251, 26.8988991, -9.1068478, 33.8970909, -41.1494141, 36.0057449
1: -8.4590302, 30.3783398, -10.7669506, 38.4980316, -46.9570541, 41.1452904
2: -8.9860716, 30.5654984, -11.2868938, 38.5900154, -47.5760880, 41.8523865
3: -13.2799826, 31.1811676, -16.8590183, 39.8212090, -53.1011887, 48.0401840
4: -13.8371363, 30.5100403, -17.7354069, 38.1494560, -51.9865913, 48.2454453

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_B2_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_B2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_B2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_B2_A1_B2_B2_A1

### Relational analysis result of IS_B2_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4592822, upper bound: 43.4189035
time: 1.00 seconds

## Relational analysis of IS_B2_B2_A1_B2_B2_A2

### Relational analysis result of IS_B2_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4592822, upper bound: 43.4189035
time: 0.59 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 5.99 seconds
IS_B1_A1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 3, lower bound: -43.5089712, upper bound: 43.5175398
IS_B1_A1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 3, lower bound: -43.4987033, upper bound: 43.4661222
IS_B1_A1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 3, lower bound: -43.4636996, upper bound: 43.4835076
IS_B1_A1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 3, lower bound: -43.4602134, upper bound: 43.4602134
IS_B1_A1_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 3, lower bound: -43.4954708, upper bound: 43.4651732
IS_B1_A1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 3, lower bound: -43.4987957, upper bound: 43.4656737
IS_B1_A1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 3, lower bound: -43.4967081, upper bound: 43.4651026
IS_B1_A1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 3, lower bound: -43.4988436, upper bound: 43.4655149
IS_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 3, lower bound: -43.5578907, upper bound: 43.5547360
IS_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 3, lower bound: -43.5578907, upper bound: 43.5547360
IS_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 3, lower bound: -43.5460020, upper bound: 43.5230990
IS_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 3, lower bound: -43.4987957, upper bound: 43.5209338
IS_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 3, lower bound: -43.4967081, upper bound: 43.5539461
IS_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 3, lower bound: -43.4967081, upper bound: 43.5539461
IS_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 3, lower bound: -43.5437599, upper bound: 43.5224424
IS_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 3, lower bound: -43.4988436, upper bound: 43.5206615
IS_B1_A2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 3, lower bound: -43.5472787, upper bound: 43.5501204
IS_B1_A2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 3, lower bound: -43.5472787, upper bound: 43.5501204
IS_B1_A2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 3, lower bound: -43.5371676, upper bound: 43.5222874
IS_B1_A2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 3, lower bound: -43.4958278, upper bound: 43.4630661
IS_B1_A2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 3, lower bound: -43.5189159, upper bound: 43.5393824
IS_B1_A2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 3, lower bound: -43.5189159, upper bound: 43.5393824
IS_B1_A2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 3, lower bound: -43.5168084, upper bound: 43.5191840
IS_B1_A2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 3, lower bound: -43.4971066, upper bound: 43.4631742
IS_B1_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 3, lower bound: -43.5462759, upper bound: 43.5484287
IS_B1_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 3, lower bound: -43.5361042, upper bound: 43.5214457
IS_B1_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 3, lower bound: -43.5454465, upper bound: 43.5456144
IS_B1_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 3, lower bound: -43.5354970, upper bound: 43.5208401
IS_B1_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 3, lower bound: -43.5185740, upper bound: 43.5410342
IS_B1_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 3, lower bound: -43.5165910, upper bound: 43.5184723
IS_B1_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 3, lower bound: -43.5180794, upper bound: 43.5387679
IS_B1_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 3, lower bound: -43.5163758, upper bound: 43.5179210
IS_B2_B1_A1_A1_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.99
Output dim: 3, lower bound: -43.3773875, upper bound: 43.3682946
IS_B2_B1_A1_A1_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.99
Output dim: 3, lower bound: -43.3773875, upper bound: 43.3797757
IS_B2_B1_A1_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 3, lower bound: -43.4124803, upper bound: 43.3790408
IS_B2_B1_A1_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 3, lower bound: -43.4110513, upper bound: 43.3781633
IS_B2_B1_A1_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 3, lower bound: -43.4298288, upper bound: 43.3872458
IS_B2_B1_A1_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 3, lower bound: -43.3930732, upper bound: 43.3717053
IS_B2_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 3, lower bound: -43.4504243, upper bound: 43.4269375
IS_B2_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 3, lower bound: -43.4504243, upper bound: 43.4269375
IS_B2_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 3, lower bound: -43.4631408, upper bound: 43.4248997
IS_B2_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 3, lower bound: -43.4167397, upper bound: 43.4050184
IS_B2_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 3, lower bound: -43.4504243, upper bound: 43.4267476
IS_B2_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 3, lower bound: -43.4504243, upper bound: 43.4267476
IS_B2_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 3, lower bound: -43.4592822, upper bound: 43.4189035
IS_B2_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 3, lower bound: -43.4592822, upper bound: 43.4189035

## BFS IS instance: IS_B1_A1_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -4.8582063, 18.7531872, -4.9577832, 19.0985699, -23.9567757, 23.7109699
1: -5.5607405, 21.3109932, -5.6844931, 21.6856880, -27.2464275, 26.9954853
2: -6.0088134, 21.1523533, -6.1311479, 21.5555096, -27.5643234, 27.2834988
3: -8.9163227, 21.7504463, -9.1053019, 22.1364422, -31.0527649, 30.8557472
4: -9.7389994, 20.5956039, -9.8985271, 21.0375633, -30.7765617, 30.4941292

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B1_A1_A1_B1_B1

### Relational analysis result of IS_B1_A1_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5089712, upper bound: 43.5175398
time: 0.56 seconds

## Relational analysis of IS_B1_A1_B1_A1_A1_B1_B2

### Relational analysis result of IS_B1_A1_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5089712, upper bound: 43.5175398
time: 0.67 seconds

## BFS IS instance: IS_B1_A1_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -4.7858844, 18.5291538, -5.1313086, 19.7613087, -24.5471916, 23.6604614
1: -5.4722095, 21.0635319, -5.9050531, 22.4458580, -27.9180641, 26.9685822
2: -5.9208384, 20.8902512, -6.3495717, 22.3251266, -28.2459641, 27.2398167
3: -8.7889099, 21.4956665, -9.4575586, 22.9421806, -31.7310905, 30.9532242
4: -9.6225395, 20.3184776, -10.2586679, 21.8029175, -31.4254570, 30.5771446

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B1_A1_A1_B2_B1

### Relational analysis result of IS_B1_A1_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4987033, upper bound: 43.4661222
time: 0.85 seconds

## Relational analysis of IS_B1_A1_B1_A1_A1_B2_B2

### Relational analysis result of IS_B1_A1_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4987033, upper bound: 43.4661222
time: 0.56 seconds

## BFS IS instance: IS_B1_A1_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -5.3486099, 20.6111774, -4.9861073, 19.1929646, -24.5415688, 25.5972805
1: -6.1417828, 23.4408894, -5.7178378, 21.7899647, -27.9317474, 29.1587238
2: -6.6195807, 23.2761002, -6.1664224, 21.6632328, -28.2828121, 29.4425220
3: -9.8246441, 23.9595108, -9.1556301, 22.2452431, -32.0698853, 33.1151352
4: -10.7149191, 22.6733818, -9.9441013, 21.1597195, -31.8746357, 32.6174774

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B1_A1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B1_A1_A2_B1_A1

### Relational analysis result of IS_B1_A1_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4602134, upper bound: 43.4602134
time: 0.52 seconds

## Relational analysis of IS_B1_A1_B1_A1_A2_B1_A2

### Relational analysis result of IS_B1_A1_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4602134, upper bound: 43.4602134
time: 0.80 seconds

## BFS IS instance: IS_B1_A1_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -5.2765565, 20.3923912, -5.1641083, 19.8714752, -25.1480312, 25.5564976
1: -6.0537119, 23.1990738, -5.9439797, 22.5678883, -28.6215992, 29.1430492
2: -6.5322580, 23.0197754, -6.3905325, 22.4510746, -28.9833336, 29.4103069
3: -9.6982050, 23.7097397, -9.5164499, 23.0694866, -32.7676888, 33.2261887
4: -10.6005325, 22.4015694, -10.3124475, 21.9416656, -32.5421982, 32.7140160

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B1_A1_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B1_A1_A2_B2_A1

### Relational analysis result of IS_B1_A1_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4602134, upper bound: 43.4602134
time: 0.52 seconds

## Relational analysis of IS_B1_A1_B1_A1_A2_B2_A2

### Relational analysis result of IS_B1_A1_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4602134, upper bound: 43.4602134
time: 0.53 seconds

## BFS IS instance: IS_B1_A1_B1_A2_A1_A1

### Backsubstitution after applying IS history:
0: -6.0885549, 23.2257309, -5.1037683, 19.6061115, -25.6946640, 28.3294964
1: -7.0463910, 26.3764725, -5.8617659, 22.2568798, -29.3032703, 32.2382317
2: -7.5475416, 26.3148308, -6.3115911, 22.1449699, -29.6925125, 32.6264076
3: -11.1884851, 27.0328350, -9.3753757, 22.7311630, -33.9196472, 36.4082108
4: -12.0433311, 25.7938061, -10.1639118, 21.6388206, -33.6821442, 35.9577141

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_B1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B1_A2_A1_A1_B1

### Relational analysis result of IS_B1_A1_B1_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4954708, upper bound: 43.4651732
time: 0.60 seconds

## Relational analysis of IS_B1_A1_B1_A2_A1_A1_B2

### Relational analysis result of IS_B1_A1_B1_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4954708, upper bound: 43.4651732
time: 0.57 seconds

## BFS IS instance: IS_B1_A1_B1_A2_A1_A2

### Backsubstitution after applying IS history:
0: -6.2577157, 23.8770256, -5.0313768, 19.3818436, -25.6395588, 28.9084015
1: -7.2588172, 27.1199970, -5.7730703, 22.0090160, -29.2678337, 32.8930664
2: -7.7611303, 27.0703793, -6.2232547, 21.8828754, -29.6440029, 33.2936325
3: -11.5275402, 27.8219662, -9.2476997, 22.4759216, -34.0034523, 37.0696640
4: -12.3982773, 26.5276928, -10.0475492, 21.3590775, -33.7573547, 36.5752411

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_B1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B1_A2_A1_A2_B1

### Relational analysis result of IS_B1_A1_B1_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4987957, upper bound: 43.4656737
time: 0.51 seconds

## Relational analysis of IS_B1_A1_B1_A2_A1_A2_B2

### Relational analysis result of IS_B1_A1_B1_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4987957, upper bound: 43.4656737
time: 0.52 seconds

## BFS IS instance: IS_B1_A1_B1_A2_A2_A1

### Backsubstitution after applying IS history:
0: -6.6875443, 25.4257469, -5.1321077, 19.7012730, -26.3888168, 30.5578537
1: -7.7536058, 28.8930283, -5.8951459, 22.3621426, -30.1157494, 34.7881737
2: -8.2894773, 28.8299103, -6.3471155, 22.2533951, -30.5428734, 35.1770248
3: -12.2797890, 29.6489182, -9.4257946, 22.8409214, -35.1207123, 39.0747147
4: -13.2051535, 28.2554035, -10.2097702, 21.7616463, -34.9667969, 38.4651718

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_B1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B1_A2_A2_A1_B1

### Relational analysis result of IS_B1_A1_B1_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4967081, upper bound: 43.4651026
time: 0.66 seconds

## Relational analysis of IS_B1_A1_B1_A2_A2_A1_B2

### Relational analysis result of IS_B1_A1_B1_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4967081, upper bound: 43.4651026
time: 0.62 seconds

## BFS IS instance: IS_B1_A1_B1_A2_A2_A2

### Backsubstitution after applying IS history:
0: -6.8670044, 26.1093140, -5.0601077, 19.4787388, -26.3457413, 31.1694221
1: -7.9779143, 29.6654892, -5.8069854, 22.1162319, -30.0941448, 35.4724731
2: -8.5154953, 29.6150322, -6.2592626, 21.9933033, -30.5087986, 35.8742905
3: -12.6371927, 30.4754868, -9.2989740, 22.5875607, -35.2247543, 39.7744522
4: -13.5781269, 29.0213718, -10.0941887, 21.4839191, -35.0620384, 39.1155624

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_B1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B1_A2_A2_A2_B1

### Relational analysis result of IS_B1_A1_B1_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4602134, upper bound: 43.4655149
time: 0.64 seconds

## Relational analysis of IS_B1_A1_B1_A2_A2_A2_B2

### Relational analysis result of IS_B1_A1_B1_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4988436, upper bound: 43.4655149
time: 0.63 seconds

## BFS IS instance: IS_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -5.7061234, 21.6405411, -6.3952656, 24.2670021, -29.9731255, 28.0358067
1: -6.5904408, 24.5144615, -7.4161768, 27.5266991, -34.1171341, 31.9306374
2: -7.0664692, 24.5139065, -7.9277906, 27.5264244, -34.5928879, 32.4416962
3: -10.4670162, 25.0784016, -11.7486191, 28.2345562, -38.7015686, 36.8270187
4: -11.1840363, 24.1650333, -12.5724287, 27.0626621, -38.2466927, 36.7374573

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_B2_A1_B1_A1_A1

### Relational analysis result of IS_B1_A1_B2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5511690, upper bound: 43.5490628
time: 0.73 seconds

## Relational analysis of IS_B1_A1_B2_A1_B1_A1_A2

### Relational analysis result of IS_B1_A1_B2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5232617, upper bound: 43.5382451
time: 0.83 seconds

## BFS IS instance: IS_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -5.8667369, 22.2643013, -6.3952656, 24.2670021, -30.1337395, 28.6595669
1: -6.7931447, 25.2247486, -7.4161768, 27.5266991, -34.3198357, 32.6409264
2: -7.2691727, 25.2408981, -7.9277906, 27.5264244, -34.7955971, 33.1686783
3: -10.7906885, 25.8321896, -11.7486191, 28.2345562, -39.0252419, 37.5808105
4: -11.5194817, 24.8643398, -12.5724287, 27.0626621, -38.5821381, 37.4367638

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5235840, upper bound: 43.5444155
time: 0.54 seconds

## Relational analysis of IS_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5068513, upper bound: 43.5382451
time: 0.53 seconds

## BFS IS instance: IS_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -5.5266933, 21.0206604, -6.5069389, 24.7133541, -30.2400455, 27.5275993
1: -6.3781414, 23.8062801, -7.5613995, 28.0278912, -34.4060326, 31.3676796
2: -6.8453784, 23.8103371, -8.0701904, 28.0422134, -34.8875923, 31.8805275
3: -10.1488638, 24.3489952, -11.9855223, 28.7896614, -38.9385262, 36.3345032
4: -10.8588514, 23.4758854, -12.8211317, 27.5608673, -38.4197159, 36.2970161

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A1_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B2_A1_B2_A1_A1

### Relational analysis result of IS_B1_A1_B2_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5449356, upper bound: 43.5230990
time: 0.55 seconds

## Relational analysis of IS_B1_A1_B2_A1_B2_A1_A2

### Relational analysis result of IS_B1_A1_B2_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5449356, upper bound: 43.5230990
time: 0.68 seconds

## BFS IS instance: IS_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -6.0456309, 22.7139397, -6.4557896, 24.5449066, -30.5905380, 29.1697292
1: -7.0229769, 25.6920338, -7.4977713, 27.8498669, -34.8728409, 33.1898041
2: -7.4868565, 25.7929726, -8.0063629, 27.8452206, -35.3320770, 33.7993355
3: -11.1103468, 26.3227463, -11.8896313, 28.5880318, -39.6983795, 38.2123795
4: -11.7448854, 25.5597954, -12.7291269, 27.3548546, -39.0997353, 38.2889214

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A1_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B2_A1_B2_A2_A1

### Relational analysis result of IS_B1_A1_B2_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4954708, upper bound: 43.5209338
time: 0.55 seconds

## Relational analysis of IS_B1_A1_B2_A1_B2_A2_A2

### Relational analysis result of IS_B1_A1_B2_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4954708, upper bound: 43.5209338
time: 0.76 seconds

## BFS IS instance: IS_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6.2710218, 23.7205467, -6.4747691, 24.5368595, -30.8078804, 30.1953163
1: -7.2545557, 26.8992329, -7.5113759, 27.8232212, -35.0777740, 34.4106026
2: -7.7669778, 26.8930187, -8.0265694, 27.8370266, -35.6040039, 34.9195862
3: -11.4908857, 27.5549469, -11.8921070, 28.5450668, -40.0359421, 39.4470520
4: -12.2808580, 26.4816818, -12.7080669, 27.3883533, -39.6692123, 39.1897430

Time for backsubstitution: 2.05 seconds
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0156250, mid=0.0156250, abs_max=46.890357971191406
rel_dist={3: [-43.60075692157075, 43.60075692157075]}

## Binary Search with IS_dual Result
status: None
Maximum delta epsilon: None
execution time: 1133.94 seconds
