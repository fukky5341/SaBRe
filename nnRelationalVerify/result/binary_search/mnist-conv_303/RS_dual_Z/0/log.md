## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 0.96581779658
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-9.2881069, -6.2814474, -9.2881069, -6.2814474, -3.0066595, 3.0066595)
1: (-6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.4866414, 2.4866414)
2: (-8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.3348742, 2.3348742)
3: (-10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.6362443, 2.6362443)
4: (-5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270)
5: (-5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.4902666, 2.4902666)
6: (-13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620)
7: (3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031)
8: (-4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.9694605, 2.9694605)
9: (-2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185)

## BASE Result
execution time: IAR + LP analysis = 13.04 + 33.00 = 46.03 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -1.4649108, upper bound: 1.4649096


# Binary Search by BASE starts (time budget: 3553.97 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=1.7645337581634521
rel_dist={7: [-1.1515141861453313, 1.1515132652632292]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.VERIFIED, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=1.6208463907241821
rel_dist={7: [-0.8562086879320736, 0.8562091799144329]}

## Binary search (step 2) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=4, k_high=5, k_mid=4, eps_mid=0.0156250, abs_max=1.6687421798706055
rel_dist={7: [-0.9703767544746134, 0.9703745834961706]}

## Binary Search Result
Binary search time: 143.39 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.01171875


# Relational Split (RS_dual_Z) starts
Time budget: 3410.58 seconds

## Binary search (step 0) starts
Candidate k: 8, corresponding eps: 0.0312500


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6235
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6235

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2674312, upper bound: 1.2674333
time: 4.02 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2674312, upper bound: 1.2674333
time: 4.15 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 8.34 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 8.34
Output dim: 7, lower bound: -1.2674312, upper bound: 1.2674333
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 8.34
Output dim: 7, lower bound: -1.2674312, upper bound: 1.2674333

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.7294507, 2.7273581
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.4167013, 2.4146707
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.3126001, 2.3141291
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.3181357, 2.3183200
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.4474001, 2.4460049
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.6065326, 2.5977612
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 13.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2621105, upper bound: 1.2674205
time: 4.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2674184, upper bound: 1.2621126
time: 4.04 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.7273583, 2.7294509
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.4146705, 2.4167013
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.3141289, 2.3125997
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.3183198, 2.3181357
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.4460049, 2.4474001
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.5977607, 2.6065323
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 13.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2621105, upper bound: 1.2674205
time: 4.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2674184, upper bound: 1.2621127
time: 4.25 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 22.01 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 22.01
Output dim: 7, lower bound: -1.2621105, upper bound: 1.2674205
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 22.01
Output dim: 7, lower bound: -1.2674184, upper bound: 1.2621126
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 22.01
Output dim: 7, lower bound: -1.2621105, upper bound: 1.2674205
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 22.01
Output dim: 7, lower bound: -1.2674184, upper bound: 1.2621127

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.6882524, 2.7194133
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.4098911, 2.3792562
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.3113651, 2.3077757
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.3177743, 2.3164735
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.4186554, 2.4404685
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.6054144, 2.5920949
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 13.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2597095, upper bound: 1.2674144
time: 4.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2621044, upper bound: 1.2650162
time: 4.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.7215061, 2.6861601
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.3812866, 2.4078605
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.3062458, 2.3128946
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.3162894, 2.3179588
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.4418640, 2.4172604
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.6008663, 2.5966434
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 13.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2650140, upper bound: 1.2621065
time: 4.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2674123, upper bound: 1.2597117
time: 4.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.6861601, 2.7215061
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.4078603, 2.3812871
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.3128948, 2.3062463
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.3179593, 2.3162889
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.4172602, 2.4418638
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.5966434, 2.6008663
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 13.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 6181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2597095, upper bound: 1.2674144
time: 4.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2621044, upper bound: 1.2650145
time: 4.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.7194133, 2.6882527
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.3792562, 2.4098914
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.3077755, 2.3113654
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.3164735, 2.3177745
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.4404688, 2.4186556
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.5920949, 2.6054146
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 13.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2650140, upper bound: 1.2621065
time: 4.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2674123, upper bound: 1.2597117
time: 4.33 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 22.38 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.38
Output dim: 7, lower bound: -1.2597095, upper bound: 1.2674144
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.38
Output dim: 7, lower bound: -1.2621044, upper bound: 1.2650162
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.38
Output dim: 7, lower bound: -1.2650140, upper bound: 1.2621065
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.38
Output dim: 7, lower bound: -1.2674123, upper bound: 1.2597117
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.38
Output dim: 7, lower bound: -1.2597095, upper bound: 1.2674144
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.38
Output dim: 7, lower bound: -1.2621044, upper bound: 1.2650145
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.38
Output dim: 7, lower bound: -1.2650140, upper bound: 1.2621065
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.38
Output dim: 7, lower bound: -1.2674123, upper bound: 1.2597117

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.6809835, 2.7145658
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.4127598, 2.3797855
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.3012819, 2.3020387
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.2997732, 2.3044558
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.3748550, 2.4132271
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.5896831, 2.5646992
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 13.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 4627

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2580831, upper bound: 1.2674103
time: 4.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2597082, upper bound: 1.2657892
time: 4.23 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.6834049, 2.7121439
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.4104204, 2.3821251
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.3056283, 2.2976925
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.3057566, 2.2984719
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.3914108, 2.3966680
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.5780187, 2.5763636
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 13.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 4627

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2604793, upper bound: 1.2650143
time: 4.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2621027, upper bound: 1.2633898
time: 4.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.7142367, 2.6813123
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.3841558, 2.4083900
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.2961631, 2.3071575
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.2982874, 2.3059411
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.3980632, 2.3900156
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.5851350, 2.5692477
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 13.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 4627

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2633877, upper bound: 1.2621029
time: 4.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2650121, upper bound: 1.2604814
time: 4.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.7166586, 2.6788907
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.3818164, 2.4107294
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.3005095, 2.3028116
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.3042712, 2.2999575
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.4146223, 2.3734603
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.5734706, 2.5809119
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 13.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 4627

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2657872, upper bound: 1.2597104
time: 4.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2674099, upper bound: 1.2580853
time: 4.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.6788907, 2.7166584
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.4107294, 2.3818164
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.3028116, 2.3005092
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.2999573, 2.3042715
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.3734603, 2.4146223
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.5809121, 2.5734706
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 13.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 4627

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2580831, upper bound: 1.2674103
time: 4.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2597082, upper bound: 1.2657892
time: 4.22 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.6813126, 2.7142367
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.4083900, 2.3841558
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.3071575, 2.2961631
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.3059411, 2.2982876
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.3900156, 2.3980632
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.5692477, 2.5851347
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 13.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 4627

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2604793, upper bound: 1.2650143
time: 4.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2621027, upper bound: 1.2633898
time: 4.23 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.7121439, 2.6834049
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.3821249, 2.4104209
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.2976923, 2.3056283
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.2984719, 2.3057568
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.3966680, 2.3914108
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.5763636, 2.5780189
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 13.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 4627

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2633877, upper bound: 1.2621029
time: 4.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2650121, upper bound: 1.2604814
time: 4.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.7145658, 2.6809833
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.3797855, 2.4127603
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.3020387, 2.3012822
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.3044558, 2.2997730
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.4132271, 2.3748550
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.5646992, 2.5896831
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 13.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 4627

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2657872, upper bound: 1.2597104
time: 4.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2674099, upper bound: 1.2580853
time: 4.20 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 21.87 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.87
Output dim: 7, lower bound: -1.2580831, upper bound: 1.2674103
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.87
Output dim: 7, lower bound: -1.2597082, upper bound: 1.2657892
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.87
Output dim: 7, lower bound: -1.2604793, upper bound: 1.2650143
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.87
Output dim: 7, lower bound: -1.2621027, upper bound: 1.2633898
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.87
Output dim: 7, lower bound: -1.2633877, upper bound: 1.2621029
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.87
Output dim: 7, lower bound: -1.2650121, upper bound: 1.2604814
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.87
Output dim: 7, lower bound: -1.2657872, upper bound: 1.2597104
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.87
Output dim: 7, lower bound: -1.2674099, upper bound: 1.2580853
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.87
Output dim: 7, lower bound: -1.2580831, upper bound: 1.2674103
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.87
Output dim: 7, lower bound: -1.2597082, upper bound: 1.2657892
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.87
Output dim: 7, lower bound: -1.2604793, upper bound: 1.2650143
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.87
Output dim: 7, lower bound: -1.2621027, upper bound: 1.2633898
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.87
Output dim: 7, lower bound: -1.2633877, upper bound: 1.2621029
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.87
Output dim: 7, lower bound: -1.2650121, upper bound: 1.2604814
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.87
Output dim: 7, lower bound: -1.2657872, upper bound: 1.2597104
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.87
Output dim: 7, lower bound: -1.2674099, upper bound: 1.2580853

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.5664368, 2.6381259
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.4215035, 2.3815174
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.3022251, 2.3026819
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.2899888, 2.2897880
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.3455667, 2.4016237
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.5902882, 2.5655894
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 13.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 5857

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2528126, upper bound: 1.2674077
time: 4.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2580791, upper bound: 1.2621468
time: 4.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.6046152, 2.6000190
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.4144921, 2.3885415
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.3019252, 2.3029759
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.2851050, 2.2946754
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.3632693, 2.3839388
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.5905738, 2.5653045
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 13.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 5857

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2544381, upper bound: 1.2657849
time: 4.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2597041, upper bound: 1.2605253
time: 4.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.5688581, 2.6357343
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.4191694, 2.3838570
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.3065662, 2.2983360
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.2959740, 2.2838042
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.3621221, 2.3850741
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.5786242, 2.5772538
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 13.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5857

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2552161, upper bound: 1.2650103
time: 4.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2604750, upper bound: 1.2597422
time: 4.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.6070065, 2.5975971
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.4121528, 2.3908758
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.3062716, 2.2986348
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.2910893, 2.2886896
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.3798199, 2.3673792
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.5789089, 2.5769689
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 13.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5857

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2568402, upper bound: 1.2633842
time: 4.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2620984, upper bound: 1.2581171
time: 4.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.5996900, 2.6049137
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.3929067, 2.4101219
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.2971053, 2.3078010
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.2885048, 2.2912734
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.3687744, 2.3784246
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.5857401, 2.5701377
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 13.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5857

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2581167, upper bound: 1.2621005
time: 4.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2633837, upper bound: 1.2568405
time: 4.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.6378269, 2.5667655
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.3858876, 2.4171386
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.2968063, 2.3080959
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.2836201, 2.2961588
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.3864694, 2.3607268
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.5860252, 2.5698528
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 13.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5857

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2597416, upper bound: 1.2604771
time: 4.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2650081, upper bound: 1.2552167
time: 4.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.6021118, 2.6025224
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.3905721, 2.4124613
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.3014464, 2.3034549
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.2944911, 2.2852898
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.3853340, 2.3618741
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.5740757, 2.5818024
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 13.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5857

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2605231, upper bound: 1.2597046
time: 4.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2657829, upper bound: 1.2544386
time: 4.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.6402187, 2.5643439
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.3835483, 2.4194729
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.3011527, 2.3037546
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.2896035, 2.2901731
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.4030190, 2.3441715
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.5743604, 2.5815172
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 13.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5857

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2621465, upper bound: 1.2580794
time: 4.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2674056, upper bound: 1.2528132
time: 4.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.5643439, 2.6402187
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.4194732, 2.3835483
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.3037543, 2.3011527
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.2901728, 2.2896037
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.3441715, 2.4030190
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.5815172, 2.5743606
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 13.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 5857

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2528126, upper bound: 1.2674077
time: 4.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2580791, upper bound: 1.2621468
time: 4.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.6025224, 2.6021116
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.4124613, 2.3905723
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.3034549, 2.3014464
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.2852900, 2.2944911
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.3618741, 2.3853340
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.5818024, 2.5740759
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 13.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 5857

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2544381, upper bound: 1.2657850
time: 4.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2597041, upper bound: 1.2605253
time: 4.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.5667658, 2.6378272
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.4171386, 2.3858876
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.3080955, 2.2968063
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.2961590, 2.2836199
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.3607268, 2.3864694
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.5698528, 2.5860252
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 13.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5857

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2552161, upper bound: 1.2650103
time: 4.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2604750, upper bound: 1.2597420
time: 4.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.6049137, 2.5996900
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.4101219, 2.3929067
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.3078008, 2.2971053
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.2912734, 2.2885053
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.3784246, 2.3687744
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.5701380, 2.5857401
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 13.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5857

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2568402, upper bound: 1.2633843
time: 4.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2620984, upper bound: 1.2581171
time: 4.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.5975971, 2.6070065
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.3908758, 2.4121528
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.2986350, 2.3062716
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.2886899, 2.2910891
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.3673797, 2.3798199
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.5769687, 2.5789092
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 13.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5857

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2581167, upper bound: 1.2621005
time: 4.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2633837, upper bound: 1.2568405
time: 4.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.6357346, 2.5688581
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.3838573, 2.4191694
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.2983356, 2.3065662
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.2838042, 2.2959745
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.3850741, 2.3621221
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.5772538, 2.5786242
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 13.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5857

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2597416, upper bound: 1.2604770
time: 4.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2650081, upper bound: 1.2552164
time: 4.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.6000190, 2.6046152
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.3885417, 2.4144921
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.3029761, 2.3019254
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.2946751, 2.2851052
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.3839388, 2.3632693
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.5653048, 2.5905738
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 13.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5857

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2605231, upper bound: 1.2597049
time: 4.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2657829, upper bound: 1.2544383
time: 5.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.6381259, 2.5664365
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.3815174, 2.4215038
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.3026819, 2.3022251
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.2897875, 2.2899888
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.4016237, 2.3455667
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.5655890, 2.5902886
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 13.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5857

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2621465, upper bound: 1.2580794
time: 4.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2674056, upper bound: 1.2528132
time: 4.82 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 22.69 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 7, lower bound: -1.2528126, upper bound: 1.2674077
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 7, lower bound: -1.2580791, upper bound: 1.2621468
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 7, lower bound: -1.2544381, upper bound: 1.2657849
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 7, lower bound: -1.2597041, upper bound: 1.2605253
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 7, lower bound: -1.2552161, upper bound: 1.2650103
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 7, lower bound: -1.2604750, upper bound: 1.2597422
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 7, lower bound: -1.2568402, upper bound: 1.2633842
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 7, lower bound: -1.2620984, upper bound: 1.2581171
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 7, lower bound: -1.2581167, upper bound: 1.2621005
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 7, lower bound: -1.2633837, upper bound: 1.2568405
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 7, lower bound: -1.2597416, upper bound: 1.2604771
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 7, lower bound: -1.2650081, upper bound: 1.2552167
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 7, lower bound: -1.2605231, upper bound: 1.2597046
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 7, lower bound: -1.2657829, upper bound: 1.2544386
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 7, lower bound: -1.2621465, upper bound: 1.2580794
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 7, lower bound: -1.2674056, upper bound: 1.2528132
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 7, lower bound: -1.2528126, upper bound: 1.2674077
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 7, lower bound: -1.2580791, upper bound: 1.2621468
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 7, lower bound: -1.2544381, upper bound: 1.2657850
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 7, lower bound: -1.2597041, upper bound: 1.2605253
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 7, lower bound: -1.2552161, upper bound: 1.2650103
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 7, lower bound: -1.2604750, upper bound: 1.2597420
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 7, lower bound: -1.2568402, upper bound: 1.2633843
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 7, lower bound: -1.2620984, upper bound: 1.2581171
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 7, lower bound: -1.2581167, upper bound: 1.2621005
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 7, lower bound: -1.2633837, upper bound: 1.2568405
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 7, lower bound: -1.2597416, upper bound: 1.2604770
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 7, lower bound: -1.2650081, upper bound: 1.2552164
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 7, lower bound: -1.2605231, upper bound: 1.2597049
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 7, lower bound: -1.2657829, upper bound: 1.2544383
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 7, lower bound: -1.2621465, upper bound: 1.2580794
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.69
Output dim: 7, lower bound: -1.2674056, upper bound: 1.2528132

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.5690742, 2.6232293
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.4215088, 2.3814886
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.3045540, 2.2896342
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.2926941, 2.2746294
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.3482676, 2.3865056
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.5924706, 2.5532851
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 13.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5751

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2528058, upper bound: 1.2667343
time: 4.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2521332, upper bound: 1.2674008
time: 4.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.5515399, 2.6381259
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.4214749, 2.3815174
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.2891774, 2.3026819
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.2748299, 2.2897880
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.3304482, 2.4016237
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.5779843, 2.5655894
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 13.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5751

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2580722, upper bound: 1.2614742
time: 8.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2573993, upper bound: 1.2621404
time: 8.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.6072531, 2.5851221
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.4144969, 2.3885126
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.3042541, 2.2899282
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.2878113, 2.2795167
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.3659706, 2.3688202
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.5927563, 2.5530005
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 13.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 5751

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2544313, upper bound: 1.2651115
time: 4.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2537583, upper bound: 1.2657763
time: 4.92 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 23.71 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 23.71
Output dim: 7, lower bound: -1.2528058, upper bound: 1.2667343
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 23.71
Output dim: 7, lower bound: -1.2521332, upper bound: 1.2674008
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 23.71
Output dim: 7, lower bound: -1.2580722, upper bound: 1.2614742
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 23.71
Output dim: 7, lower bound: -1.2573993, upper bound: 1.2621404
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 23.71
Output dim: 7, lower bound: -1.2544313, upper bound: 1.2651115
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 23.71
Output dim: 7, lower bound: -1.2537583, upper bound: 1.2657763
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.71
Output dim: 7, lower bound: -1.2597041, upper bound: 1.2605253
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.71
Output dim: 7, lower bound: -1.2552161, upper bound: 1.2650103
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.71
Output dim: 7, lower bound: -1.2604750, upper bound: 1.2597422
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.71
Output dim: 7, lower bound: -1.2568402, upper bound: 1.2633842
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.71
Output dim: 7, lower bound: -1.2620984, upper bound: 1.2581171
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.71
Output dim: 7, lower bound: -1.2581167, upper bound: 1.2621005
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.71
Output dim: 7, lower bound: -1.2633837, upper bound: 1.2568405
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.71
Output dim: 7, lower bound: -1.2597416, upper bound: 1.2604771
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.71
Output dim: 7, lower bound: -1.2650081, upper bound: 1.2552167
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.71
Output dim: 7, lower bound: -1.2605231, upper bound: 1.2597046
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.71
Output dim: 7, lower bound: -1.2657829, upper bound: 1.2544386
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.71
Output dim: 7, lower bound: -1.2621465, upper bound: 1.2580794
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.71
Output dim: 7, lower bound: -1.2674056, upper bound: 1.2528132
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.71
Output dim: 7, lower bound: -1.2528126, upper bound: 1.2674077
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.71
Output dim: 7, lower bound: -1.2580791, upper bound: 1.2621468
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.71
Output dim: 7, lower bound: -1.2544381, upper bound: 1.2657850
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.71
Output dim: 7, lower bound: -1.2597041, upper bound: 1.2605253
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.71
Output dim: 7, lower bound: -1.2552161, upper bound: 1.2650103
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.71
Output dim: 7, lower bound: -1.2604750, upper bound: 1.2597420
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.71
Output dim: 7, lower bound: -1.2568402, upper bound: 1.2633843
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.71
Output dim: 7, lower bound: -1.2620984, upper bound: 1.2581171
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.71
Output dim: 7, lower bound: -1.2581167, upper bound: 1.2621005
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.71
Output dim: 7, lower bound: -1.2633837, upper bound: 1.2568405
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.71
Output dim: 7, lower bound: -1.2597416, upper bound: 1.2604770
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.71
Output dim: 7, lower bound: -1.2650081, upper bound: 1.2552164
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.71
Output dim: 7, lower bound: -1.2605231, upper bound: 1.2597049
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.71
Output dim: 7, lower bound: -1.2657829, upper bound: 1.2544383
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.71
Output dim: 7, lower bound: -1.2621465, upper bound: 1.2580794
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.71
Output dim: 7, lower bound: -1.2674056, upper bound: 1.2528132
Binary search (step 0): status=Status.UNKNOWN, k_low=4, k_high=12, k_mid=8, eps_mid=0.0312500, abs_max=1.783203125
rel_dist={7: [-1.2674345485483558, 1.2674366290257746]}

## Binary search (step 1) starts
Candidate k: 5, corresponding eps: 0.0195312


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6235
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 6235

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0712187, upper bound: 1.0712174
time: 4.73 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0712187, upper bound: 1.0712173
time: 4.46 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 9.35 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 9.35
Output dim: 7, lower bound: -1.0712187, upper bound: 1.0712174
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 9.35
Output dim: 7, lower bound: -1.0712187, upper bound: 1.0712173

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.4209104, 2.4196024
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.2013903, 2.2001212
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.1047640, 2.1057198
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.0592399, 2.0593553
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.4403911, 2.4405856
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.2106237, 2.2097516
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0260139, 3.0339413
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7172592, 1.7176968
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.3258386, 2.3203566
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4058070, 2.4061255

Time for backsubstitution: 13.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0682682, upper bound: 1.0712136
time: 4.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0712117, upper bound: 1.0682676
time: 4.98 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.4196024, 2.4209104
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.2001214, 2.2013903
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.1057196, 2.1047637
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.0593553, 2.0592399
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.4405856, 2.4403911
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.2097516, 2.2106237
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0339408, 3.0260143
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7176969, 1.7172595
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.3203564, 2.3258386
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4061255, 2.4058075

Time for backsubstitution: 13.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0682682, upper bound: 1.0712108
time: 4.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0712117, upper bound: 1.0682672
time: 4.59 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 22.10 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 22.10
Output dim: 7, lower bound: -1.0682682, upper bound: 1.0712136
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 22.10
Output dim: 7, lower bound: -1.0712117, upper bound: 1.0682676
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 22.10
Output dim: 7, lower bound: -1.0682682, upper bound: 1.0712108
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 22.10
Output dim: 7, lower bound: -1.0712117, upper bound: 1.0682672

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.3797121, 2.3991876
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1838536, 2.1647067
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.1016102, 2.0993664
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.0583215, 2.0575085
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.4256406, 2.4108124
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1818790, 2.1955125
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0223479, 3.0265656
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7126701, 1.7154154
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.3230152, 2.3146904
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4028625, 2.4046621

Time for backsubstitution: 13.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0650754, upper bound: 1.0712037
time: 4.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0682603, upper bound: 1.0679976
time: 4.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.4004955, 2.3784041
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1659756, 2.1825843
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0984106, 2.1025655
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.0573936, 2.0584369
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.4106183, 2.4258356
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1963844, 2.1810076
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0186391, 3.0302753
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7149780, 1.7131078
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.3201723, 2.3175330
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4043436, 2.4031811

Time for backsubstitution: 13.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0679984, upper bound: 1.0682593
time: 4.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0712049, upper bound: 1.0650740
time: 4.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.3784041, 2.4004955
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1825843, 2.1659760
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.1025658, 2.0984104
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.0584369, 2.0573933
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.4258351, 2.4106178
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1810074, 2.1963842
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0302749, 3.0186386
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7131078, 1.7149781
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.3175330, 2.3201723
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4031806, 2.4043436

Time for backsubstitution: 13.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0650774, upper bound: 1.0712032
time: 4.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0682603, upper bound: 1.0679974
time: 11.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.3991876, 2.3797121
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1647067, 2.1838536
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0993662, 2.1016097
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.0575089, 2.0583217
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.4108129, 2.4256406
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1955128, 2.1818793
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0265660, 3.0223484
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7154157, 1.7126706
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.3146906, 2.3230152
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4046617, 2.4028630

Time for backsubstitution: 13.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0679984, upper bound: 1.0682593
time: 4.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0712049, upper bound: 1.0650739
time: 4.71 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 22.44 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.44
Output dim: 7, lower bound: -1.0650754, upper bound: 1.0712037
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.44
Output dim: 7, lower bound: -1.0682603, upper bound: 1.0679976
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.44
Output dim: 7, lower bound: -1.0679984, upper bound: 1.0682593
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.44
Output dim: 7, lower bound: -1.0712049, upper bound: 1.0650740
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.44
Output dim: 7, lower bound: -1.0650774, upper bound: 1.0712032
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.44
Output dim: 7, lower bound: -1.0682603, upper bound: 1.0679974
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.44
Output dim: 7, lower bound: -1.0679984, upper bound: 1.0682593
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.44
Output dim: 7, lower bound: -1.0712049, upper bound: 1.0650739

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.3724427, 2.3934317
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1858454, 2.1652360
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0915265, 2.0919995
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.0403204, 2.0432470
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.4157839, 2.4051151
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1380792, 2.1620617
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0388107, 3.0392532
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7129734, 1.7156054
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.3029094, 2.2872946
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4007039, 2.4015985

Time for backsubstitution: 13.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 4627

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0641260, upper bound: 1.0712024
time: 4.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0650746, upper bound: 1.0702456
time: 4.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.3739562, 2.3919182
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1843829, 2.1666982
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0942430, 2.0892832
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.0440598, 2.0395069
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.4199438, 2.4009552
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1484261, 2.1517119
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0350361, 3.0430279
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7128603, 1.7157180
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2956195, 2.2945848
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3997989, 2.4025030

Time for backsubstitution: 13.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 4627

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0673033, upper bound: 1.0679962
time: 4.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0682593, upper bound: 1.0670463
time: 4.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.3932261, 2.3726482
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1679673, 2.1831138
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0883274, 2.0951989
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.0393915, 2.0441754
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.4007607, 2.4201384
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1525841, 2.1475539
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0351009, 3.0429626
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7152808, 1.7132978
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.3000669, 2.2901373
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4021850, 2.4001174

Time for backsubstitution: 13.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 4627

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0670469, upper bound: 1.0682581
time: 3.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0679973, upper bound: 1.0673020
time: 4.20 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.3947396, 2.3711348
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1665053, 2.1845760
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0910435, 2.0924826
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.0431318, 2.0404356
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.4049206, 2.4159784
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1629333, 2.1372070
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0313263, 3.0467372
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7151682, 1.7134106
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2927766, 2.2974277
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4012799, 2.4010224

Time for backsubstitution: 13.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 4627

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0702467, upper bound: 1.0650737
time: 4.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0712035, upper bound: 1.0641248
time: 4.20 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.3711348, 2.3947396
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1845760, 2.1665053
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0924826, 2.0910435
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.0404353, 2.0431318
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.4159784, 2.4049206
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1372070, 2.1629333
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0467377, 3.0313263
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7134106, 1.7151681
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2974277, 2.2927766
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4010220, 2.4012804

Time for backsubstitution: 13.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 4627

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0641260, upper bound: 1.0712023
time: 4.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0650746, upper bound: 1.0702459
time: 4.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.3726482, 2.3932261
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1831136, 2.1679676
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0951986, 2.0883274
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.0441751, 2.0393918
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.4201384, 2.4007607
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1475539, 2.1525841
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0429630, 3.0351009
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7132981, 1.7152808
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2901373, 2.3000669
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4001174, 2.4021850

Time for backsubstitution: 14.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 4627

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0673033, upper bound: 1.0679962
time: 4.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0682593, upper bound: 1.0670463
time: 4.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.3919182, 2.3739562
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1666985, 2.1843829
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0892830, 2.0942430
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.0395069, 2.0440602
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.4009552, 2.4199438
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1517119, 2.1484261
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0430279, 3.0350356
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7157180, 1.7128606
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2945848, 2.2956195
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4025030, 2.3997993

Time for backsubstitution: 13.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 4627

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0670469, upper bound: 1.0682581
time: 4.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0679973, upper bound: 1.0673020
time: 4.22 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.3934317, 2.3724427
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1652360, 2.1858451
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0919995, 2.0915265
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.0432467, 2.0403204
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.4051151, 2.4157834
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1620617, 2.1380792
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0392532, 3.0388103
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7156055, 1.7129732
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2872949, 2.3029096
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4015985, 2.4007044

Time for backsubstitution: 13.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 4627

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0702467, upper bound: 1.0650731
time: 4.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0712035, upper bound: 1.0641248
time: 4.09 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 22.05 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.05
Output dim: 7, lower bound: -1.0641260, upper bound: 1.0712024
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.05
Output dim: 7, lower bound: -1.0650746, upper bound: 1.0702456
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.05
Output dim: 7, lower bound: -1.0673033, upper bound: 1.0679962
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.05
Output dim: 7, lower bound: -1.0682593, upper bound: 1.0670463
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.05
Output dim: 7, lower bound: -1.0670469, upper bound: 1.0682581
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.05
Output dim: 7, lower bound: -1.0679973, upper bound: 1.0673020
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.05
Output dim: 7, lower bound: -1.0702467, upper bound: 1.0650737
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.05
Output dim: 7, lower bound: -1.0712035, upper bound: 1.0641248
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.05
Output dim: 7, lower bound: -1.0641260, upper bound: 1.0712023
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.05
Output dim: 7, lower bound: -1.0650746, upper bound: 1.0702459
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.05
Output dim: 7, lower bound: -1.0673033, upper bound: 1.0679962
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.05
Output dim: 7, lower bound: -1.0682593, upper bound: 1.0670463
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.05
Output dim: 7, lower bound: -1.0670469, upper bound: 1.0682581
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.05
Output dim: 7, lower bound: -1.0679973, upper bound: 1.0673020
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.05
Output dim: 7, lower bound: -1.0702467, upper bound: 1.0650731
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.05
Output dim: 7, lower bound: -1.0712035, upper bound: 1.0641248

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.2578959, 2.3027020
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1919594, 2.1669679
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0923572, 2.0926428
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.0287049, 2.0285792
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3813534, 2.3582177
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1087904, 2.1438260
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0504332, 3.0544753
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7140851, 1.7164657
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.3035150, 2.2880778
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4014115, 2.3991127

Time for backsubstitution: 13.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5857

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0604356, upper bound: 1.0712004
time: 4.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0641234, upper bound: 1.0675166
time: 4.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.2817574, 2.2788849
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1875768, 2.1713579
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0921698, 2.0928264
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.0256522, 2.0316339
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3688860, 2.3707013
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1198545, 2.1327729
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0540371, 3.0508761
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7138333, 1.7167166
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.3036933, 2.2879000
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3982182, 2.4023099

Time for backsubstitution: 13.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5857

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0613842, upper bound: 1.0702435
time: 4.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0650720, upper bound: 1.0665563
time: 4.25 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.2594094, 2.3012071
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1905007, 2.1684301
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0950704, 2.0899265
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.0324461, 2.0248394
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3855152, 2.3540578
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1191373, 2.1334825
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0466585, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7139721, 1.7165785
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2962246, 2.2953684
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4005089, 2.4000173

Time for backsubstitution: 13.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5857

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0636214, upper bound: 1.0679943
time: 4.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0673007, upper bound: 1.0643064
time: 4.22 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.2832522, 2.2773714
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1861148, 2.1728170
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0948863, 2.0901132
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.0293925, 2.0278926
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3730469, 2.3665400
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1301985, 2.1224232
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0502605, 3.0546503
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7137208, 1.7168295
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2964029, 2.2951903
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3973131, 2.4032125

Time for backsubstitution: 13.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 5857

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0645771, upper bound: 1.0670435
time: 4.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0682568, upper bound: 1.0633563
time: 3.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.2786794, 2.2819443
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1740861, 2.1848457
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0891576, 2.0958421
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.0277779, 2.0295076
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3663454, 2.3732409
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1232953, 2.1293268
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0467234, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7163920, 1.7141582
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.3006721, 2.2909207
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4028940, 2.3976316

Time for backsubstitution: 13.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 5857

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0633573, upper bound: 1.0682568
time: 4.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0670443, upper bound: 1.0645758
time: 4.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.3025150, 2.2581015
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1696992, 2.1892312
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0889707, 2.0960264
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.0247242, 2.0325608
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3538628, 2.3857098
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1343546, 2.1182656
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0503254, 3.0545855
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7161412, 1.7144095
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.3008504, 2.2907429
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3996987, 2.4008274

Time for backsubstitution: 13.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5857

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0643077, upper bound: 1.0672998
time: 4.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0679947, upper bound: 1.0636202
time: 4.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.2801929, 2.2804496
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1726270, 2.1863079
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0918708, 2.0931258
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.0315182, 2.0257678
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3705063, 2.3690805
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1336451, 2.1189828
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0429487, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7162795, 1.7142708
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2933817, 2.2982111
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4019918, 2.3985362

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 5857

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0665573, upper bound: 1.0650709
time: 4.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0702442, upper bound: 1.0613833
time: 4.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.3040099, 2.2565880
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1682372, 2.1906900
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0916867, 2.0933132
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.0284636, 2.0288198
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3580227, 2.3815484
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1446981, 2.1079183
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0465488, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7160287, 1.7145224
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2935600, 2.2980330
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3987942, 2.4017301

Time for backsubstitution: 13.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5857

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0675176, upper bound: 1.0641222
time: 4.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0712009, upper bound: 1.0604346
time: 4.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.2565880, 2.3040099
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1906900, 2.1682372
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0933132, 2.0916867
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.0288203, 2.0284638
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3815479, 2.3580232
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1079183, 2.1446981
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0465484
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7145224, 1.7160285
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2980328, 2.2935600
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4017296, 2.3987942

Time for backsubstitution: 13.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5857

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0604356, upper bound: 1.0712005
time: 4.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0641234, upper bound: 1.0675166
time: 4.21 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.2804499, 2.2801929
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1863079, 2.1726272
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0931258, 2.0918705
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.0257676, 2.0315185
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3690805, 2.3705068
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1189828, 2.1336451
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0429492
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7142711, 1.7162794
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2982111, 2.2933822
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3985362, 2.4019918

Time for backsubstitution: 13.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5857

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0613842, upper bound: 1.0702435
time: 4.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0650720, upper bound: 1.0665563
time: 4.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.2581015, 2.3025150
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1892314, 2.1696994
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0960264, 2.0889707
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.0325606, 2.0247240
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3857098, 2.3538632
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1182656, 2.1343546
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0545855, 3.0503254
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7144094, 1.7161412
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2907429, 2.3008504
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4008274, 2.3996992

Time for backsubstitution: 13.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5857

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0636214, upper bound: 1.0679965
time: 4.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0673007, upper bound: 1.0643067
time: 4.27 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.2819443, 2.2786794
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1848454, 2.1740861
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0958419, 2.0891573
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.0295079, 2.0277774
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3732414, 2.3663454
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1293268, 2.1232953
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0467234
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7141581, 1.7163923
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2909207, 2.3006723
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3976312, 2.4028945

Time for backsubstitution: 13.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 5857

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0645771, upper bound: 1.0670433
time: 4.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0682568, upper bound: 1.0633565
time: 4.21 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.2773714, 2.2832522
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1728168, 2.1861148
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0901132, 2.0948863
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.0278924, 2.0293922
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3665400, 2.3730464
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1224232, 2.1301985
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0546503, 3.0502601
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7168298, 1.7137209
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2951899, 2.2964029
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4032125, 2.3973131

Time for backsubstitution: 13.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5857

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0633573, upper bound: 1.0682568
time: 4.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0670443, upper bound: 1.0645758
time: 4.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.3012071, 2.2594094
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1684299, 2.1905005
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0899262, 2.0950704
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.0248396, 2.0324457
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3540573, 2.3855152
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1334825, 2.1191378
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0466585
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7165785, 1.7139722
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2953682, 2.2962248
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4000168, 2.4005094

Time for backsubstitution: 13.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5857

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0643077, upper bound: 1.0672998
time: 4.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0679947, upper bound: 1.0636202
time: 4.17 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.2788849, 2.2817576
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1713576, 2.1875770
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0928264, 2.0921700
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.0316336, 2.0256524
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3707008, 2.3688860
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1327729, 2.1198545
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0508757, 3.0540371
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7167168, 1.7138336
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2879000, 2.3036933
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4023099, 2.3982182

Time for backsubstitution: 13.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5857

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0665573, upper bound: 1.0650709
time: 4.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0702442, upper bound: 1.0613833
time: 4.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.3027020, 2.2578959
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1669679, 2.1919594
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0926428, 2.0923572
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.0285790, 2.0287046
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3582172, 2.3813539
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1438260, 2.1087904
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0544758, 3.0504332
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7164655, 1.7140851
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2880778, 2.3035150
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3991122, 2.4014120

Time for backsubstitution: 13.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5857

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0675176, upper bound: 1.0641222
time: 4.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0712009, upper bound: 1.0604346
time: 4.36 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 22.10 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.10
Output dim: 7, lower bound: -1.0604356, upper bound: 1.0712004
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.10
Output dim: 7, lower bound: -1.0641234, upper bound: 1.0675166
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.10
Output dim: 7, lower bound: -1.0613842, upper bound: 1.0702435
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.10
Output dim: 7, lower bound: -1.0650720, upper bound: 1.0665563
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.10
Output dim: 7, lower bound: -1.0636214, upper bound: 1.0679943
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.10
Output dim: 7, lower bound: -1.0673007, upper bound: 1.0643064
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.10
Output dim: 7, lower bound: -1.0645771, upper bound: 1.0670435
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.10
Output dim: 7, lower bound: -1.0682568, upper bound: 1.0633563
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.10
Output dim: 7, lower bound: -1.0633573, upper bound: 1.0682568
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.10
Output dim: 7, lower bound: -1.0670443, upper bound: 1.0645758
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.10
Output dim: 7, lower bound: -1.0643077, upper bound: 1.0672998
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.10
Output dim: 7, lower bound: -1.0679947, upper bound: 1.0636202
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.10
Output dim: 7, lower bound: -1.0665573, upper bound: 1.0650709
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.10
Output dim: 7, lower bound: -1.0702442, upper bound: 1.0613833
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.10
Output dim: 7, lower bound: -1.0675176, upper bound: 1.0641222
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.10
Output dim: 7, lower bound: -1.0712009, upper bound: 1.0604346
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.10
Output dim: 7, lower bound: -1.0604356, upper bound: 1.0712005
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.10
Output dim: 7, lower bound: -1.0641234, upper bound: 1.0675166
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.10
Output dim: 7, lower bound: -1.0613842, upper bound: 1.0702435
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.10
Output dim: 7, lower bound: -1.0650720, upper bound: 1.0665563
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.10
Output dim: 7, lower bound: -1.0636214, upper bound: 1.0679965
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.10
Output dim: 7, lower bound: -1.0673007, upper bound: 1.0643067
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.10
Output dim: 7, lower bound: -1.0645771, upper bound: 1.0670433
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.10
Output dim: 7, lower bound: -1.0682568, upper bound: 1.0633565
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.10
Output dim: 7, lower bound: -1.0633573, upper bound: 1.0682568
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.10
Output dim: 7, lower bound: -1.0670443, upper bound: 1.0645758
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.10
Output dim: 7, lower bound: -1.0643077, upper bound: 1.0672998
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.10
Output dim: 7, lower bound: -1.0679947, upper bound: 1.0636202
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.10
Output dim: 7, lower bound: -1.0665573, upper bound: 1.0650709
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.10
Output dim: 7, lower bound: -1.0702442, upper bound: 1.0613833
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.10
Output dim: 7, lower bound: -1.0675176, upper bound: 1.0641222
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.10
Output dim: 7, lower bound: -1.0712009, upper bound: 1.0604346

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.2539582, 2.2878051
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1919518, 2.1669393
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0889201, 2.0795951
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.0247107, 2.0134206
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3769913, 2.3570719
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1048093, 2.1287079
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0366688, 3.0508614
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7101388, 1.7154261
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.3002648, 2.2757738
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3927274, 2.3968277

Time for backsubstitution: 14.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5751

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0604239, upper bound: 1.0702080
time: 4.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0594310, upper bound: 1.0711882
time: 4.46 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.2429991, 2.2987635
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1919303, 2.1669602
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0793095, 2.0892045
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.0135460, 2.0245857
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3802090, 2.3538547
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.0936718, 2.1398449
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0468178, 3.0407114
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7130456, 1.7125193
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2912107, 2.2848225
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3991270, 2.3904285

Time for backsubstitution: 14.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5751

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0641117, upper bound: 1.0665191
time: 4.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0631121, upper bound: 1.0675054
time: 4.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.2778196, 2.2639880
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1875691, 2.1713290
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0887327, 2.0797787
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.0216589, 2.0164752
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3645229, 2.3695550
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1158733, 2.1176543
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0402727, 3.0472617
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7098870, 1.7156770
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.3004432, 2.2755959
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3895340, 2.4000254

Time for backsubstitution: 14.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5751

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0613725, upper bound: 1.0692500
time: 4.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0603789, upper bound: 1.0702317
time: 4.32 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 23.25 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 23.25
Output dim: 7, lower bound: -1.0604239, upper bound: 1.0702080
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 23.25
Output dim: 7, lower bound: -1.0594310, upper bound: 1.0711882
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 23.25
Output dim: 7, lower bound: -1.0641117, upper bound: 1.0665191
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 23.25
Output dim: 7, lower bound: -1.0631121, upper bound: 1.0675054
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 23.25
Output dim: 7, lower bound: -1.0613725, upper bound: 1.0692500
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 23.25
Output dim: 7, lower bound: -1.0603789, upper bound: 1.0702317
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.0650720, upper bound: 1.0665563
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.0636214, upper bound: 1.0679943
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.0673007, upper bound: 1.0643064
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.0645771, upper bound: 1.0670435
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.0682568, upper bound: 1.0633563
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.0633573, upper bound: 1.0682568
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.0670443, upper bound: 1.0645758
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.0643077, upper bound: 1.0672998
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.0679947, upper bound: 1.0636202
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.0665573, upper bound: 1.0650709
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.0702442, upper bound: 1.0613833
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.0675176, upper bound: 1.0641222
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.0712009, upper bound: 1.0604346
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.0604356, upper bound: 1.0712005
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.0641234, upper bound: 1.0675166
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.0613842, upper bound: 1.0702435
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.0650720, upper bound: 1.0665563
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.0636214, upper bound: 1.0679965
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.0673007, upper bound: 1.0643067
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.0645771, upper bound: 1.0670433
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.0682568, upper bound: 1.0633565
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.0633573, upper bound: 1.0682568
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.0670443, upper bound: 1.0645758
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.0643077, upper bound: 1.0672998
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.0679947, upper bound: 1.0636202
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.0665573, upper bound: 1.0650709
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.0702442, upper bound: 1.0613833
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.0675176, upper bound: 1.0641222
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 7, lower bound: -1.0712009, upper bound: 1.0604346
Binary search (step 1): status=Status.UNKNOWN, k_low=4, k_high=7, k_mid=5, eps_mid=0.0195312, abs_max=1.7166380882263184
rel_dist={7: [-1.071222731717584, 1.071219330481989]}

## Binary search (step 2) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6235
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6235

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9703744, upper bound: 0.9701923
time: 4.40 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9701920, upper bound: 0.9703727
time: 4.89 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 9.46 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 9.46
Output dim: 7, lower bound: -0.9703744, upper bound: 0.9701923
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 9.46
Output dim: 7, lower bound: -0.9701920, upper bound: 0.9703727

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.3180633, 2.3170171
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1296201, 2.1286047
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0354853, 2.0362499
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9729414, 1.9730334
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3754020, 2.3755579
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1316981, 2.1310005
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9355712, 2.9419136
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6693637, 1.6697135
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2322736, 2.2278883
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3530722, 2.3533268

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9677510, upper bound: 0.9701844
time: 6.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9703683, upper bound: 0.9675362
time: 4.48 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.3170171, 2.3180637
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1286044, 2.1296201
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0362501, 2.0354853
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9730334, 1.9729414
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3755584, 2.3754025
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1310005, 2.1316981
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9419131, 2.9355717
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6697137, 1.6693637
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2278881, 2.2322741
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3533268, 2.3530722

Time for backsubstitution: 14.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9675379, upper bound: 0.9703666
time: 4.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9701859, upper bound: 0.9677529
time: 5.00 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 24.00 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.00
Output dim: 7, lower bound: -0.9677510, upper bound: 0.9701844
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.00
Output dim: 7, lower bound: -0.9703683, upper bound: 0.9675362
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.00
Output dim: 7, lower bound: -0.9675379, upper bound: 0.9703666
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.00
Output dim: 7, lower bound: -0.9701859, upper bound: 0.9677529

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.2768655, 2.2924457
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1085076, 2.0931902
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0316916, 2.0298965
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9718375, 1.9711869
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3576474, 2.3457851
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1029539, 2.1138604
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9311633, 2.9345374
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6647747, 1.6669706
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2288818, 2.2222223
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3501277, 2.3515673

Time for backsubstitution: 14.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9649621, upper bound: 0.9701816
time: 4.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9677450, upper bound: 0.9673802
time: 6.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.2934923, 2.2758188
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.0942054, 2.1074924
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0291319, 2.0324562
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9710946, 1.9719296
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3456292, 2.3578033
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1145582, 2.1022565
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9281964, 2.9375052
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6666210, 1.6651245
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2266078, 2.2244964
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3513126, 2.3503828

Time for backsubstitution: 14.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9675632, upper bound: 0.9675337
time: 5.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9703619, upper bound: 0.9647695
time: 4.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.2758188, 2.2934923
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1074924, 2.0942056
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0324564, 2.0291317
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9719300, 1.9710946
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3578038, 2.3456292
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1022568, 2.1145580
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9375052, 2.9281960
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6651247, 1.6666209
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2244964, 2.2266078
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3503823, 2.3513126

Time for backsubstitution: 14.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 6181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9647710, upper bound: 0.9703638
time: 4.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9675318, upper bound: 0.9675612
time: 4.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.2924457, 2.2768655
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.0931902, 2.1085076
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0298967, 2.0316913
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9711871, 1.9718375
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3457847, 2.3576479
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1138601, 2.1029541
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9345374, 2.9311638
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6669710, 1.6647748
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2222223, 2.2288821
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3515673, 2.3501282

Time for backsubstitution: 14.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9673813, upper bound: 0.9677466
time: 4.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9701797, upper bound: 0.9649610
time: 6.81 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 25.96 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.96
Output dim: 7, lower bound: -0.9649621, upper bound: 0.9701816
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.96
Output dim: 7, lower bound: -0.9677450, upper bound: 0.9673802
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.96
Output dim: 7, lower bound: -0.9675632, upper bound: 0.9675337
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.96
Output dim: 7, lower bound: -0.9703619, upper bound: 0.9647695
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.96
Output dim: 7, lower bound: -0.9647710, upper bound: 0.9703638
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.96
Output dim: 7, lower bound: -0.9675318, upper bound: 0.9675612
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.96
Output dim: 7, lower bound: -0.9673813, upper bound: 0.9677466
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.96
Output dim: 7, lower bound: -0.9701797, upper bound: 0.9649610

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.2695961, 2.2863870
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1102066, 2.0937195
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0216084, 2.0219865
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9538360, 1.9561772
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3477907, 2.3392558
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.0591536, 2.0783396
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9468708, 2.9472251
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6650550, 1.6671606
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2073183, 2.1948266
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3477883, 2.3485036

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 4627

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9642240, upper bound: 0.9701797
time: 4.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9642259, upper bound: 0.9694779
time: 4.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.2708068, 2.2851763
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1090369, 2.0948894
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0237813, 2.0198133
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9568276, 1.9531853
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3511190, 2.3359280
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.0674314, 2.0700598
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9438515, 2.9502449
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6649649, 1.6672508
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2014861, 2.2006586
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3470640, 2.3492279

Time for backsubstitution: 14.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 4627

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9670074, upper bound: 0.9673792
time: 4.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9677443, upper bound: 0.9666946
time: 4.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.2862225, 2.2697606
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.0959044, 2.1080217
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0190487, 2.0245459
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9530931, 1.9569201
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3357725, 2.3512745
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.0707574, 2.0667338
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9439030, 2.9501925
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6669009, 1.6653146
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2050443, 2.1971006
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3489728, 2.3473191

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 4627

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9668744, upper bound: 0.9675300
time: 4.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9675623, upper bound: 0.9668087
time: 4.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.2874336, 2.2685494
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.0947351, 2.1091914
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0212216, 2.0223727
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9560852, 1.9539282
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3390999, 2.3479462
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.0790372, 2.0584559
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9408836, 2.9532123
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6668108, 1.6654048
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.1992121, 2.2029328
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3482490, 2.3480430

Time for backsubstitution: 14.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 4627

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9696551, upper bound: 0.9647724
time: 4.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9703610, upper bound: 0.9640496
time: 4.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.2685494, 2.2874336
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1091914, 2.0947349
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0223727, 2.0212216
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9539280, 1.9560852
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3479462, 2.3391004
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.0584559, 2.0790372
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9532127, 2.9408836
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6654046, 1.6668109
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2029328, 2.1992121
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3480425, 2.3482494

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 4627

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9640502, upper bound: 0.9703591
time: 4.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9647704, upper bound: 0.9696553
time: 5.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.2697606, 2.2862225
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1080217, 2.0959048
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0245457, 2.0190487
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9569201, 1.9530933
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3512745, 2.3357720
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.0667338, 2.0707574
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9501925, 2.9439030
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6653144, 1.6669010
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.1971006, 2.2050443
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3473186, 2.3489733

Time for backsubstitution: 14.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 4627

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9668105, upper bound: 0.9675605
time: 4.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9675310, upper bound: 0.9668724
time: 5.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.2851763, 2.2708068
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.0948892, 2.1090372
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0198135, 2.0237813
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9531851, 1.9568279
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3359280, 2.3511186
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.0700598, 2.0674314
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9502449, 2.9438510
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6672509, 1.6649648
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2006583, 2.2014863
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3492274, 2.3470645

Time for backsubstitution: 14.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 4627

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9666969, upper bound: 0.9677442
time: 4.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9673805, upper bound: 0.9670092
time: 4.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.2863870, 2.2695961
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.0937195, 2.1102068
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0219865, 2.0216081
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9561772, 1.9538360
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3392553, 2.3477907
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.0783396, 2.0591536
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9472256, 2.9468708
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6671607, 1.6650549
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.1948266, 2.2073183
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3485036, 2.3477883

Time for backsubstitution: 14.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 4627

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9694800, upper bound: 0.9649599
time: 4.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9701788, upper bound: 0.9642259
time: 4.46 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 23.64 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.64
Output dim: 7, lower bound: -0.9642240, upper bound: 0.9701797
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.64
Output dim: 7, lower bound: -0.9642259, upper bound: 0.9694779
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.64
Output dim: 7, lower bound: -0.9670074, upper bound: 0.9673792
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.64
Output dim: 7, lower bound: -0.9677443, upper bound: 0.9666946
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.64
Output dim: 7, lower bound: -0.9668744, upper bound: 0.9675300
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.64
Output dim: 7, lower bound: -0.9675623, upper bound: 0.9668087
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.64
Output dim: 7, lower bound: -0.9696551, upper bound: 0.9647724
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.64
Output dim: 7, lower bound: -0.9703610, upper bound: 0.9640496
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.64
Output dim: 7, lower bound: -0.9640502, upper bound: 0.9703591
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.64
Output dim: 7, lower bound: -0.9647704, upper bound: 0.9696553
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.64
Output dim: 7, lower bound: -0.9668105, upper bound: 0.9675605
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.64
Output dim: 7, lower bound: -0.9675310, upper bound: 0.9668724
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.64
Output dim: 7, lower bound: -0.9666969, upper bound: 0.9677442
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.64
Output dim: 7, lower bound: -0.9673805, upper bound: 0.9670092
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.64
Output dim: 7, lower bound: -0.9694800, upper bound: 0.9649599
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.64
Output dim: 7, lower bound: -0.9701788, upper bound: 0.9642259

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.1550493, 2.1908939
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1154447, 2.0954514
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0224013, 2.0226297
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9416096, 1.9415095
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3108673, 2.2923579
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.0298648, 2.0578938
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9584942, 2.9617276
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6661162, 1.6680210
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2079239, 2.1955743
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3478570, 2.3460178

Time for backsubstitution: 14.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5857

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9613484, upper bound: 0.9701789
time: 4.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9642221, upper bound: 0.9673233
time: 7.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.1741385, 2.1718402
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1119390, 2.0989635
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0222516, 2.0227766
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9391682, 1.9439533
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3008928, 2.3023448
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.0387163, 2.0490508
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9613762, 2.9588480
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6659150, 1.6682217
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2080665, 2.1954317
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3453021, 2.3485756

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5857

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9620860, upper bound: 0.9694802
time: 5.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9649596, upper bound: 0.9666241
time: 4.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.1562600, 2.1896982
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1142774, 2.0966210
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0245719, 2.0204568
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9446032, 1.9385176
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3141956, 2.2890301
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.0381427, 2.0496187
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9554739, 2.9647489
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6660261, 1.6681111
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2020917, 2.2014065
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3471351, 2.3467417

Time for backsubstitution: 14.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5857

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9641318, upper bound: 0.9673770
time: 4.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9670055, upper bound: 0.9645254
time: 4.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.1753340, 2.1706295
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1107688, 2.1001306
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0244246, 2.0206060
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9421599, 1.9409604
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3042202, 2.2990160
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.0469913, 2.0407715
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9583549, 2.9618673
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6658254, 1.6683120
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2022343, 2.2012639
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3445783, 2.3492980

Time for backsubstitution: 14.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5857

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9648709, upper bound: 0.9666964
time: 4.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9677424, upper bound: 0.9638358
time: 4.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.1716757, 2.1742878
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1011462, 2.1097536
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0198412, 2.0251894
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9408686, 1.9422522
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.2988605, 2.3043766
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.0414691, 2.0462942
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9555264, 2.9646969
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6679621, 1.6661749
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2056494, 2.1978483
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3490429, 2.3448329

Time for backsubstitution: 14.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5857

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9640146, upper bound: 0.9675283
time: 5.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9668725, upper bound: 0.9646732
time: 5.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.1907444, 2.1552136
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.0976367, 2.1132619
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0196919, 2.0253367
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9384253, 1.9446950
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.2888746, 2.3143520
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.0503163, 2.0374451
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9584074, 2.9618154
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6677613, 1.6663760
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2057920, 2.1977060
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3464870, 2.3473897

Time for backsubstitution: 14.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5857

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9647018, upper bound: 0.9668072
time: 4.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9675604, upper bound: 0.9639508
time: 4.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.1728868, 2.1730919
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.0999789, 2.1109233
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0220118, 2.0230162
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9438612, 1.9392605
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3021889, 2.3010483
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.0497484, 2.0380187
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9525061, 2.9677181
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6678720, 1.6662651
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.1998177, 2.2036808
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3483210, 2.3455567

Time for backsubstitution: 14.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5857

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9667955, upper bound: 0.9647677
time: 4.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9696532, upper bound: 0.9619085
time: 5.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.1919403, 2.1540027
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.0964665, 2.1144292
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0218649, 2.0231662
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9414170, 1.9417021
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.2922020, 2.3110228
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.0585914, 2.0291672
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9553862, 2.9648352
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6676712, 1.6664662
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.1999598, 2.2035382
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3457632, 2.3481121

Time for backsubstitution: 14.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5857

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9675039, upper bound: 0.9640476
time: 7.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9703591, upper bound: 0.9611773
time: 4.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.1540027, 2.1919403
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1144290, 2.0964668
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0231662, 2.0218649
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9417021, 1.9414175
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3110228, 2.2922025
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.0291672, 2.0585909
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9648352, 2.9553862
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6664662, 1.6676712
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2035379, 2.1999598
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3481116, 2.3457632

Time for backsubstitution: 14.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5857

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9611788, upper bound: 0.9703610
time: 4.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9640484, upper bound: 0.9675024
time: 4.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.1730919, 2.1728866
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1109233, 2.0999789
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0230165, 2.0220120
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9392607, 1.9438610
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3010483, 2.3021894
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.0380187, 2.0497484
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9677181, 2.9525065
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6662650, 1.6678720
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2036810, 2.1998174
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3455567, 2.3483214

Time for backsubstitution: 14.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5857

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9619094, upper bound: 0.9696552
time: 4.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9647686, upper bound: 0.9667974
time: 4.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.1552134, 2.1907444
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1132617, 2.0976367
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0253367, 2.0196919
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9446948, 1.9384255
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3143520, 2.2888746
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.0374451, 2.0503163
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9618149, 2.9584074
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6663761, 1.6677613
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.1977062, 2.2057922
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3473897, 2.3464870

Time for backsubstitution: 14.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5857

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9639504, upper bound: 0.9675592
time: 4.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9668087, upper bound: 0.9647000
time: 4.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.1742878, 2.1716759
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1097536, 2.1011460
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0251894, 2.0198414
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9422524, 1.9408681
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3043766, 2.2988601
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.0462937, 2.0414691
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9646969, 2.9555259
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6661749, 1.6679622
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.1978483, 2.2056496
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3448329, 2.3490434

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5857

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9646712, upper bound: 0.9668705
time: 6.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9675292, upper bound: 0.9640126
time: 4.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.1706295, 2.1753342
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1001306, 2.1107690
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0206060, 2.0244246
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9409602, 1.9421601
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.2990160, 2.3042207
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.0407715, 2.0469913
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9618673, 2.9583554
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6683121, 1.6658251
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2012639, 2.2022340
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3492975, 2.3445783

Time for backsubstitution: 14.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5857

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9638339, upper bound: 0.9677410
time: 4.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9666950, upper bound: 0.9648727
time: 4.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.1896982, 2.1562600
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.0966210, 2.1142774
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0204568, 2.0245719
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9385178, 1.9446027
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.2890301, 2.3141961
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.0496187, 2.0381427
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9647484, 2.9554739
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6681113, 1.6660261
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2014065, 2.2020917
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3467417, 2.3471355

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5857

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9645234, upper bound: 0.9670047
time: 6.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9673787, upper bound: 0.9641302
time: 5.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.1718402, 2.1741385
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.0989633, 2.1119387
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0227766, 2.0222516
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9439528, 1.9391682
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3023453, 2.3008928
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.0490508, 2.0387163
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9588480, 2.9613767
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6682220, 1.6659153
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.1954317, 2.2080665
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3485756, 2.3453026

Time for backsubstitution: 14.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5857

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9666251, upper bound: 0.9649577
time: 5.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9694782, upper bound: 0.9620847
time: 4.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.1908937, 2.1550493
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.0954514, 2.1154447
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0226297, 2.0224013
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9415095, 1.9416099
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.2923584, 2.3108673
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.0578938, 2.0298648
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9617271, 2.9584937
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6680207, 1.6661165
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.1955743, 2.2079239
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3460174, 2.3478575

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5857

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9673241, upper bound: 0.9642210
time: 4.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9701770, upper bound: 0.9613482
time: 4.60 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 24.25 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.25
Output dim: 7, lower bound: -0.9613484, upper bound: 0.9701789
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.25
Output dim: 7, lower bound: -0.9642221, upper bound: 0.9673233
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.25
Output dim: 7, lower bound: -0.9620860, upper bound: 0.9694802
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.25
Output dim: 7, lower bound: -0.9649596, upper bound: 0.9666241
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.25
Output dim: 7, lower bound: -0.9641318, upper bound: 0.9673770
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.25
Output dim: 7, lower bound: -0.9670055, upper bound: 0.9645254
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.25
Output dim: 7, lower bound: -0.9648709, upper bound: 0.9666964
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.25
Output dim: 7, lower bound: -0.9677424, upper bound: 0.9638358
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.25
Output dim: 7, lower bound: -0.9640146, upper bound: 0.9675283
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.25
Output dim: 7, lower bound: -0.9668725, upper bound: 0.9646732
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.25
Output dim: 7, lower bound: -0.9647018, upper bound: 0.9668072
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.25
Output dim: 7, lower bound: -0.9675604, upper bound: 0.9639508
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.25
Output dim: 7, lower bound: -0.9667955, upper bound: 0.9647677
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.25
Output dim: 7, lower bound: -0.9696532, upper bound: 0.9619085
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.25
Output dim: 7, lower bound: -0.9675039, upper bound: 0.9640476
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.25
Output dim: 7, lower bound: -0.9703591, upper bound: 0.9611773
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.25
Output dim: 7, lower bound: -0.9611788, upper bound: 0.9703610
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.25
Output dim: 7, lower bound: -0.9640484, upper bound: 0.9675024
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.25
Output dim: 7, lower bound: -0.9619094, upper bound: 0.9696552
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.25
Output dim: 7, lower bound: -0.9647686, upper bound: 0.9667974
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.25
Output dim: 7, lower bound: -0.9639504, upper bound: 0.9675592
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.25
Output dim: 7, lower bound: -0.9668087, upper bound: 0.9647000
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.25
Output dim: 7, lower bound: -0.9646712, upper bound: 0.9668705
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.25
Output dim: 7, lower bound: -0.9675292, upper bound: 0.9640126
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.25
Output dim: 7, lower bound: -0.9638339, upper bound: 0.9677410
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.25
Output dim: 7, lower bound: -0.9666950, upper bound: 0.9648727
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.25
Output dim: 7, lower bound: -0.9645234, upper bound: 0.9670047
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.25
Output dim: 7, lower bound: -0.9673787, upper bound: 0.9641302
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.25
Output dim: 7, lower bound: -0.9666251, upper bound: 0.9649577
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.25
Output dim: 7, lower bound: -0.9694782, upper bound: 0.9620847
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.25
Output dim: 7, lower bound: -0.9673241, upper bound: 0.9642210
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.25
Output dim: 7, lower bound: -0.9701770, upper bound: 0.9613482

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.1489196, 2.1759973
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1154327, 2.0954227
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0170417, 2.0095820
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9353838, 1.9263508
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3065042, 2.2905688
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.0236564, 2.0427752
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9447298, 2.9560833
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6621699, 1.6664000
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2028627, 2.1832700
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3391728, 2.3424530

Time for backsubstitution: 14.67 seconds
Binary search (step 2): status=Status.UNKNOWN, k_low=4, k_high=4, k_mid=4, eps_mid=0.0156250, abs_max=1.6687421798706055
rel_dist={7: [-0.9703767544746134, 0.9703745834961706]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01171875
execution time: 2422.90 seconds
