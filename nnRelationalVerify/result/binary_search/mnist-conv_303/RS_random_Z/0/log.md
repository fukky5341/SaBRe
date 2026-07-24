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
execution time: IAR + LP analysis = 14.91 + 33.05 = 47.96 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -1.4649108, upper bound: 1.4649096


# Binary Search by BASE starts (time budget: 3552.04 seconds, max iter: 100)

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
Binary search time: 147.69 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.01171875


# Relational Split (RS_random_Z) starts
Time budget: 3404.35 seconds

## Binary search (step 0) starts
Candidate k: 8, corresponding eps: 0.0312500


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 6235
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 551

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 943

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2629399, upper bound: 1.2637078
time: 4.52 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2637078, upper bound: 1.2629421
time: 4.15 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 8.68 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 8.68
Output dim: 7, lower bound: -1.2629399, upper bound: 1.2637078
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 8.68
Output dim: 7, lower bound: -1.2637078, upper bound: 1.2629421

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.7371187, 2.7327147
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.4207282, 2.4209340
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.3160853, 2.3119898
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.3226502, 2.3175430
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.4508891, 2.4461827
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.6077023, 2.6048045
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 12.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 6235

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2576201, upper bound: 1.2636955
time: 4.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2629271, upper bound: 1.2583884
time: 4.16 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.7327147, 2.7337170
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.4207649, 2.4207282
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.3119898, 2.3129158
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.3175433, 2.3186898
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.4461827, 2.4472423
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.6048045, 2.6054630
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 12.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 6235
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 874

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4627

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2620830, upper bound: 1.2629360
time: 5.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2637054, upper bound: 1.2613172
time: 4.23 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 21.83 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 21.83
Output dim: 7, lower bound: -1.2576201, upper bound: 1.2636955
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 21.83
Output dim: 7, lower bound: -1.2629271, upper bound: 1.2583884
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 21.83
Output dim: 7, lower bound: -1.2620830, upper bound: 1.2629360
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 21.83
Output dim: 7, lower bound: -1.2637054, upper bound: 1.2613172

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.6959205, 2.7247696
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.4139180, 2.3855195
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.3148508, 2.3056364
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.3222892, 2.3156965
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.4221444, 2.4406457
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.6065836, 2.5991375
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 12.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 6235

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5857

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2567737, upper bound: 1.2636876
time: 9.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2568041, upper bound: 1.2620828
time: 4.24 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.7291741, 2.6915164
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.3853135, 2.4141238
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.3097320, 2.3107555
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.3208039, 2.3171821
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.4453521, 2.4174380
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.6020350, 2.6036859
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 12.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 6235
type: RSZ, layer: 1, pos: 5751

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2629260, upper bound: 1.2581072
time: 4.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2626768, upper bound: 1.2583895
time: 4.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.6181955, 2.6573761
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.4295282, 2.4224687
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.3129315, 2.3135591
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.3077602, 2.3040223
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.4168892, 2.4356511
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.6054091, 2.6063530
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 12.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 6235
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2567629, upper bound: 1.2629268
time: 4.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2620699, upper bound: 1.2576205
time: 4.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.6563740, 2.6191976
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.4225044, 2.4294927
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.3126330, 2.3138578
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.3028746, 2.3089075
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.4345922, 2.4179485
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.6056933, 2.6060681
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 12.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6235
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 5751

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5857

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2620909, upper bound: 1.2605011
time: 4.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2636986, upper bound: 1.2604708
time: 4.41 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 21.15 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.15
Output dim: 7, lower bound: -1.2567737, upper bound: 1.2636876
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.15
Output dim: 7, lower bound: -1.2568041, upper bound: 1.2620828
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.15
Output dim: 7, lower bound: -1.2629260, upper bound: 1.2581072
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.15
Output dim: 7, lower bound: -1.2626768, upper bound: 1.2583895
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.15
Output dim: 7, lower bound: -1.2567629, upper bound: 1.2629268
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.15
Output dim: 7, lower bound: -1.2620699, upper bound: 1.2576205
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.15
Output dim: 7, lower bound: -1.2620909, upper bound: 1.2605011
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.15
Output dim: 7, lower bound: -1.2636986, upper bound: 1.2604708

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.6985998, 2.7098734
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.4139233, 2.3854909
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.3171983, 2.2925885
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.3249967, 2.3005385
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.4248533, 2.4255271
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.6087980, 2.5868435
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 12.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6235
type: RSZ, layer: 1, pos: 5751

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2543593, upper bound: 1.2636844
time: 4.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2567677, upper bound: 1.2612770
time: 4.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.6810246, 2.7247696
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.4138899, 2.3855195
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.3018031, 2.3056364
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.3071315, 2.3156965
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.4070258, 2.4406457
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.5942893, 2.5991375
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 12.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 6235
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 874

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2568033, upper bound: 1.2618032
time: 4.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2565226, upper bound: 1.2620816
time: 4.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.7310615, 2.6947043
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.3858829, 2.4150777
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.3093042, 2.3104694
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.3209646, 2.3172796
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.4436946, 2.4163232
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.6010566, 2.6022277
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 12.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 6235
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 551

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 874

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2434490, upper bound: 1.2574823
time: 5.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2622978, upper bound: 1.2386335
time: 4.26 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.7323580, 2.6934035
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.3862662, 2.4146924
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.3094463, 2.3103278
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.3209016, 2.3173432
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.4442401, 2.4157801
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.6005769, 2.6027088
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 12.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6235
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 874

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6235

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2626763, upper bound: 1.2583862
time: 4.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2626763, upper bound: 1.2583862
time: 4.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.5769687, 2.6493609
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.4227133, 2.3870566
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.3116980, 2.3072052
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.3073974, 2.3021755
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.3881416, 2.4301033
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.6042910, 2.6006873
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 12.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6235
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 5857

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6235

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2567599, upper bound: 1.2629212
time: 4.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2567599, upper bound: 1.2629231
time: 4.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.6102219, 2.6161489
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.3941164, 2.4156609
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.3065782, 2.3123240
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.3059144, 2.3036611
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.4113498, 2.4069037
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.5997429, 2.6052358
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 12.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 6235

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5857

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2604554, upper bound: 1.2568045
time: 4.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2620632, upper bound: 1.2567722
time: 5.20 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.6590109, 2.6043010
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.4225097, 2.4294639
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.3149619, 2.3008103
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.3055811, 2.2937489
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.4372940, 2.4028304
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.6078763, 2.5937636
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 12.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 6235
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 874

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2426158, upper bound: 1.2598718
time: 7.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2614643, upper bound: 1.2410242
time: 4.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.6414776, 2.6191976
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.4224758, 2.4294927
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.2995849, 2.3138578
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.2877169, 2.3089075
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.4194736, 2.4179485
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.5933986, 2.6060681
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 12.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6235

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 874

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2442220, upper bound: 1.2598442
time: 4.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2630708, upper bound: 1.2409959
time: 4.26 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 21.31 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.31
Output dim: 7, lower bound: -1.2543593, upper bound: 1.2636844
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.31
Output dim: 7, lower bound: -1.2567677, upper bound: 1.2612770
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.31
Output dim: 7, lower bound: -1.2568033, upper bound: 1.2618032
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.31
Output dim: 7, lower bound: -1.2565226, upper bound: 1.2620816
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.31
Output dim: 7, lower bound: -1.2434490, upper bound: 1.2574823
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.31
Output dim: 7, lower bound: -1.2622978, upper bound: 1.2386335
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.31
Output dim: 7, lower bound: -1.2626763, upper bound: 1.2583862
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.31
Output dim: 7, lower bound: -1.2626763, upper bound: 1.2583862
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.31
Output dim: 7, lower bound: -1.2567599, upper bound: 1.2629212
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.31
Output dim: 7, lower bound: -1.2567599, upper bound: 1.2629231
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.31
Output dim: 7, lower bound: -1.2604554, upper bound: 1.2568045
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.31
Output dim: 7, lower bound: -1.2620632, upper bound: 1.2567722
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.31
Output dim: 7, lower bound: -1.2426158, upper bound: 1.2598718
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.31
Output dim: 7, lower bound: -1.2614643, upper bound: 1.2410242
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.31
Output dim: 7, lower bound: -1.2442220, upper bound: 1.2598442
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.31
Output dim: 7, lower bound: -1.2630708, upper bound: 1.2409959

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.6913300, 2.7050257
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.4167933, 2.3860211
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.3071165, 2.2868528
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.3069947, 2.2885201
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.3810520, 2.3982854
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.5930667, 2.5594478
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 12.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 6235
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 551

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2543590, upper bound: 1.2633676
time: 5.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2540366, upper bound: 1.2636841
time: 4.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.6937518, 2.7026041
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.4144540, 2.3883607
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.3114624, 2.2825065
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.3129785, 2.2825363
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.3976073, 2.3817258
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.5814023, 2.5711119
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 12.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 6235
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 874

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5751

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2567609, upper bound: 1.2605978
time: 4.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2560948, upper bound: 1.2612683
time: 4.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.6829109, 2.7279584
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.4144588, 2.3864720
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.3013754, 2.3053505
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.3072925, 2.3157940
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.4053683, 2.4395361
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.5933108, 2.5976794
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 12.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 6235

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 874

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2373279, upper bound: 1.2611764
time: 4.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2561766, upper bound: 1.2423301
time: 4.19 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.6842074, 2.7266567
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.4148436, 2.3860884
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.3015175, 2.3052087
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.3072286, 2.3158574
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.4059091, 2.4389882
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.5928297, 2.5981591
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 12.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 6235
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 5751

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 874

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2370475, upper bound: 1.2614525
time: 5.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2558959, upper bound: 1.2426057
time: 4.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.7303448, 2.6910195
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.4062133, 2.4297090
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.3085999, 2.3113863
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.3262699, 2.3275242
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.4438124, 2.4162323
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.5948033, 2.5928571
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 12.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6235
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 4627

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6235

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2434450, upper bound: 1.2574822
time: 4.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2434450, upper bound: 1.2574799
time: 6.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.7273765, 2.6939878
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.4005141, 2.4354086
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.3102212, 2.3097646
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.3312094, 2.3225846
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.4436035, 2.4164412
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.5916862, 2.5959740
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 12.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 6235

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 551

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2622975, upper bound: 1.2385779
time: 4.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2620161, upper bound: 1.2385786
time: 4.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.7280679, 2.6870441
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.3821926, 2.4085956
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.3091345, 2.3115454
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.3203471, 2.3169732
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.4443893, 2.4145427
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.6016760, 2.5950303
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 13.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4627

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2610513, upper bound: 1.2583844
time: 4.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2626738, upper bound: 1.2567610
time: 4.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.7259989, 2.6891370
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.3801694, 2.4106250
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.3106627, 2.3100159
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.3205311, 2.3167887
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.4430027, 2.4159379
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.5928984, 2.6038113
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 13.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 874

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5751

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2626695, upper bound: 1.2577134
time: 4.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2620033, upper bound: 1.2583794
time: 4.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.5727024, 2.6430027
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.4186492, 2.3809595
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.3113823, 2.3084180
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.3068426, 2.3018055
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.3882999, 2.4288671
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.6053596, 2.5929837
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 12.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 5751

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2543473, upper bound: 1.2629153
time: 5.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2567530, upper bound: 1.2605092
time: 4.25 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.5706100, 2.6450956
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.4166183, 2.3829904
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.3129110, 2.3068886
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.3070271, 2.3016210
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.3869047, 2.4302623
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.5965886, 2.6017549
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 12.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 5857

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2543473, upper bound: 1.2629171
time: 4.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2567530, upper bound: 1.2605110
time: 4.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.6128588, 2.6012521
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.3941216, 2.4156322
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.3089070, 2.2992761
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.3086205, 2.2885022
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.4140511, 2.3917856
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.6019254, 2.5929308
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 12.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 6235
type: RSZ, layer: 1, pos: 551

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 874

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2409805, upper bound: 1.2561777
time: 4.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2598288, upper bound: 1.2373271
time: 4.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.5953250, 2.6161489
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.3940878, 2.4156609
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.2935305, 2.3123240
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.2907562, 2.3036611
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.3962317, 2.4069037
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.5874476, 2.6052358
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 12.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 6235
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2596480, upper bound: 1.2567657
time: 6.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2620564, upper bound: 1.2543581
time: 4.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.6582894, 2.6006105
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.4428406, 2.4440954
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.3142562, 2.3017263
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.3108845, 2.3039913
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.4374108, 2.4027386
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.6016226, 2.5843940
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 12.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6235
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 6208

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6235

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2426130, upper bound: 1.2598673
time: 5.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2426130, upper bound: 1.2598679
time: 5.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.6553206, 2.6035788
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.4371409, 2.4497952
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.3158774, 2.3001046
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.3158236, 2.2990522
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.4372020, 2.4029474
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.5985060, 2.5875108
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 13.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6235

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 551

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2614640, upper bound: 1.2407089
time: 4.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2611497, upper bound: 1.2410259
time: 4.19 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.6407561, 2.6155071
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.4428067, 2.4441235
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.2988791, 2.3147733
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.2930202, 2.3191500
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.4195914, 2.4178576
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.5871449, 2.5966978
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 12.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 6235
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5751

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2442152, upper bound: 1.2591713
time: 5.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2435486, upper bound: 1.2598372
time: 4.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.6377873, 2.6184754
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.4371071, 2.4498234
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.3005013, 2.3131516
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.2979593, 2.3142109
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.4193826, 2.4180660
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.5840282, 2.5998144
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 13.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 6235
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 6208

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2606968, upper bound: 1.2409865
time: 4.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2630648, upper bound: 1.2386335
time: 4.51 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 21.88 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.88
Output dim: 7, lower bound: -1.2543590, upper bound: 1.2633676
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.88
Output dim: 7, lower bound: -1.2540366, upper bound: 1.2636841
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.88
Output dim: 7, lower bound: -1.2567609, upper bound: 1.2605978
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.88
Output dim: 7, lower bound: -1.2560948, upper bound: 1.2612683
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.88
Output dim: 7, lower bound: -1.2373279, upper bound: 1.2611764
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.88
Output dim: 7, lower bound: -1.2561766, upper bound: 1.2423301
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.88
Output dim: 7, lower bound: -1.2370475, upper bound: 1.2614525
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.88
Output dim: 7, lower bound: -1.2558959, upper bound: 1.2426057
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.88
Output dim: 7, lower bound: -1.2434450, upper bound: 1.2574822
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.88
Output dim: 7, lower bound: -1.2434450, upper bound: 1.2574799
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.88
Output dim: 7, lower bound: -1.2622975, upper bound: 1.2385779
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.88
Output dim: 7, lower bound: -1.2620161, upper bound: 1.2385786
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.88
Output dim: 7, lower bound: -1.2610513, upper bound: 1.2583844
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.88
Output dim: 7, lower bound: -1.2626738, upper bound: 1.2567610
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.88
Output dim: 7, lower bound: -1.2626695, upper bound: 1.2577134
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.88
Output dim: 7, lower bound: -1.2620033, upper bound: 1.2583794
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.88
Output dim: 7, lower bound: -1.2543473, upper bound: 1.2629153
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.88
Output dim: 7, lower bound: -1.2567530, upper bound: 1.2605092
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.88
Output dim: 7, lower bound: -1.2543473, upper bound: 1.2629171
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.88
Output dim: 7, lower bound: -1.2567530, upper bound: 1.2605110
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.88
Output dim: 7, lower bound: -1.2409805, upper bound: 1.2561777
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.88
Output dim: 7, lower bound: -1.2598288, upper bound: 1.2373271
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.88
Output dim: 7, lower bound: -1.2596480, upper bound: 1.2567657
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.88
Output dim: 7, lower bound: -1.2620564, upper bound: 1.2543581
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.88
Output dim: 7, lower bound: -1.2426130, upper bound: 1.2598673
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.88
Output dim: 7, lower bound: -1.2426130, upper bound: 1.2598679
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.88
Output dim: 7, lower bound: -1.2614640, upper bound: 1.2407089
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.88
Output dim: 7, lower bound: -1.2611497, upper bound: 1.2410259
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.88
Output dim: 7, lower bound: -1.2442152, upper bound: 1.2591713
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.88
Output dim: 7, lower bound: -1.2435486, upper bound: 1.2598372
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.88
Output dim: 7, lower bound: -1.2606968, upper bound: 1.2409865
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.88
Output dim: 7, lower bound: -1.2630648, upper bound: 1.2386335

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.6636696, 2.6635609
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.4183197, 2.3857512
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.2953005, 2.2796435
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.2363777, 2.2414548
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.3634124, 2.3890314
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.5507431, 2.4959769
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 12.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6235
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5751

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6235

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2543560, upper bound: 1.2633644
time: 4.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2543560, upper bound: 1.2633643
time: 4.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.6498652, 2.6773686
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.4165230, 2.3875489
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.2999072, 2.2750368
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.2599144, 2.2179034
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.3717966, 2.3806458
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.5295954, 2.5171323
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 12.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 6235
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 4627

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 874

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2346203, upper bound: 1.2630554
time: 4.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2534521, upper bound: 1.2442068
time: 4.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.6960731, 2.7019584
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.4147906, 2.3882658
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.3125596, 2.2822018
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.3118474, 2.2865784
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.3962712, 2.3865008
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.5816159, 2.5710535
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 12.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6235
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6235

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2567579, upper bound: 1.2605946
time: 4.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2567579, upper bound: 1.2605946
time: 4.24 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.6931062, 2.7026041
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.4143586, 2.3883607
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.3111577, 2.2825065
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.3129785, 2.2814054
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.3976073, 2.3803897
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.5813441, 2.5711119
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185

Time for backsubstitution: 12.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 6235
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 551

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2560945, upper bound: 1.2609474
time: 4.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2557797, upper bound: 1.2612676
time: 4.35 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 21.02 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 21.02
Output dim: 7, lower bound: -1.2543560, upper bound: 1.2633644
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 21.02
Output dim: 7, lower bound: -1.2543560, upper bound: 1.2633643
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 21.02
Output dim: 7, lower bound: -1.2346203, upper bound: 1.2630554
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 21.02
Output dim: 7, lower bound: -1.2534521, upper bound: 1.2442068
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 21.02
Output dim: 7, lower bound: -1.2567579, upper bound: 1.2605946
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 21.02
Output dim: 7, lower bound: -1.2567579, upper bound: 1.2605946
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 21.02
Output dim: 7, lower bound: -1.2560945, upper bound: 1.2609474
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 21.02
Output dim: 7, lower bound: -1.2557797, upper bound: 1.2612676
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.02
Output dim: 7, lower bound: -1.2373279, upper bound: 1.2611764
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.02
Output dim: 7, lower bound: -1.2561766, upper bound: 1.2423301
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.02
Output dim: 7, lower bound: -1.2370475, upper bound: 1.2614525
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.02
Output dim: 7, lower bound: -1.2558959, upper bound: 1.2426057
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.02
Output dim: 7, lower bound: -1.2434450, upper bound: 1.2574822
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.02
Output dim: 7, lower bound: -1.2434450, upper bound: 1.2574799
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.02
Output dim: 7, lower bound: -1.2622975, upper bound: 1.2385779
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.02
Output dim: 7, lower bound: -1.2620161, upper bound: 1.2385786
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.02
Output dim: 7, lower bound: -1.2610513, upper bound: 1.2583844
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.02
Output dim: 7, lower bound: -1.2626738, upper bound: 1.2567610
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.02
Output dim: 7, lower bound: -1.2626695, upper bound: 1.2577134
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.02
Output dim: 7, lower bound: -1.2620033, upper bound: 1.2583794
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.02
Output dim: 7, lower bound: -1.2543473, upper bound: 1.2629153
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.02
Output dim: 7, lower bound: -1.2567530, upper bound: 1.2605092
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.02
Output dim: 7, lower bound: -1.2543473, upper bound: 1.2629171
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.02
Output dim: 7, lower bound: -1.2567530, upper bound: 1.2605110
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.02
Output dim: 7, lower bound: -1.2409805, upper bound: 1.2561777
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.02
Output dim: 7, lower bound: -1.2598288, upper bound: 1.2373271
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.02
Output dim: 7, lower bound: -1.2596480, upper bound: 1.2567657
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.02
Output dim: 7, lower bound: -1.2620564, upper bound: 1.2543581
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.02
Output dim: 7, lower bound: -1.2426130, upper bound: 1.2598673
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.02
Output dim: 7, lower bound: -1.2426130, upper bound: 1.2598679
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.02
Output dim: 7, lower bound: -1.2614640, upper bound: 1.2407089
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.02
Output dim: 7, lower bound: -1.2611497, upper bound: 1.2410259
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.02
Output dim: 7, lower bound: -1.2442152, upper bound: 1.2591713
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.02
Output dim: 7, lower bound: -1.2435486, upper bound: 1.2598372
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.02
Output dim: 7, lower bound: -1.2606968, upper bound: 1.2409865
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.02
Output dim: 7, lower bound: -1.2630648, upper bound: 1.2386335
Binary search (step 0): status=Status.UNKNOWN, k_low=4, k_high=12, k_mid=8, eps_mid=0.0312500, abs_max=1.783203125
rel_dist={7: [-1.2674345485483558, 1.2674366290257746]}

## Binary search (step 1) starts
Candidate k: 5, corresponding eps: 0.0195312


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6235
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 6208

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6235

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0712187, upper bound: 1.0712174
time: 4.71 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0712187, upper bound: 1.0712173
time: 4.44 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 9.17 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 9.17
Output dim: 7, lower bound: -1.0712187, upper bound: 1.0712174
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 9.17
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

Time for backsubstitution: 13.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 5751

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 874

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0585796, upper bound: 1.0704336
time: 4.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0704315, upper bound: 1.0585783
time: 4.61 seconds

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

Time for backsubstitution: 13.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 5857

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 551

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0712182, upper bound: 1.0706414
time: 4.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0706422, upper bound: 1.0712170
time: 4.59 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 22.04 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 22.04
Output dim: 7, lower bound: -1.0585796, upper bound: 1.0704336
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 22.04
Output dim: 7, lower bound: -1.0704315, upper bound: 1.0585783
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 22.04
Output dim: 7, lower bound: -1.0712182, upper bound: 1.0706414
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 22.04
Output dim: 7, lower bound: -1.0706422, upper bound: 1.0712170

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.4190793, 2.4159160
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.2195830, 2.2147515
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.1040583, 2.1060276
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.0645437, 2.0677466
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.4406185, 2.4377613
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.2106619, 2.2096598
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0239668, 3.0322475
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7058411, 1.7086529
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.3184152, 2.3109856
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3941011, 2.3913412

Time for backsubstitution: 12.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 4627

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0554206, upper bound: 1.0704269
time: 4.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0585709, upper bound: 1.0673029
time: 4.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.4172244, 2.4177713
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.2160206, 2.2183137
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.1050720, 2.1050143
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.0676308, 2.0646594
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.4375668, 2.4408135
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.2105322, 2.2097905
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0243206, 3.0318942
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7082157, 1.7062787
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.3164673, 2.3129334
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3910227, 2.3944197

Time for backsubstitution: 13.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 6181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0704303, upper bound: 1.0582238
time: 4.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0700943, upper bound: 1.0585766
time: 4.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.3867645, 2.3794429
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.2009745, 2.2011206
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0939074, 2.0958307
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9887276, 2.0033317
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.4389291, 2.4383001
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1921129, 2.1982238
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0327225, 3.0251760
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7181935, 1.7173536
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2701311, 2.2623911
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3937235, 2.3960567

Time for backsubstitution: 12.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 6208

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5857

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0675321, upper bound: 1.0706386
time: 4.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0712156, upper bound: 1.0670295
time: 4.44 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.3781347, 2.3880761
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1998515, 2.2022445
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0967865, 2.0929515
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.0034471, 1.9886121
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.4384942, 2.4387341
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1973543, 2.1929848
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0331030, 3.0247951
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7177906, 1.7177564
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2569089, 2.2756143
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3963752, 2.3934054

Time for backsubstitution: 12.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 5751

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0676902, upper bound: 1.0712120
time: 4.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0706352, upper bound: 1.0682665
time: 4.48 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 21.53 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.53
Output dim: 7, lower bound: -1.0554206, upper bound: 1.0704269
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.53
Output dim: 7, lower bound: -1.0585709, upper bound: 1.0673029
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.53
Output dim: 7, lower bound: -1.0704303, upper bound: 1.0582238
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.53
Output dim: 7, lower bound: -1.0700943, upper bound: 1.0585766
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.53
Output dim: 7, lower bound: -1.0675321, upper bound: 1.0706386
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.53
Output dim: 7, lower bound: -1.0712156, upper bound: 1.0670295
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.53
Output dim: 7, lower bound: -1.0676902, upper bound: 1.0712120
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.53
Output dim: 7, lower bound: -1.0706352, upper bound: 1.0682665

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.4118099, 2.4101605
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.2215781, 2.2152839
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0939775, 2.0986643
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.0465460, 2.0534897
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.4307575, 2.4320602
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1668620, 2.1762066
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0404282, 3.0449338
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7061443, 1.7088432
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2983184, 2.2835984
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3919425, 2.3882775

Time for backsubstitution: 12.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 5857

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5751

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0554088, upper bound: 1.0694322
time: 5.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0544074, upper bound: 1.0704125
time: 9.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.4133234, 2.4086471
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.2201152, 2.2167459
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0966940, 2.0959473
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.0502858, 2.0497484
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.4349184, 2.4279003
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1772089, 2.1658592
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0366535, 3.0487084
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7060318, 1.7089560
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2910280, 2.2908883
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3910375, 2.3891821

Time for backsubstitution: 12.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 551

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5857

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0548886, upper bound: 1.0673005
time: 6.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0585684, upper bound: 1.0636089
time: 6.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.4191127, 2.4204731
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.2165871, 2.2191203
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.1046491, 2.1046801
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.0677686, 2.0647569
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.4349465, 2.4374914
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.2088742, 2.2084718
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0237589, 3.0314527
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7079654, 1.7059648
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.3153400, 2.3115001
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3913975, 2.3946824

Time for backsubstitution: 12.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 5751

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5857

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0667263, upper bound: 1.0582208
time: 4.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0704278, upper bound: 1.0546029
time: 4.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.4199090, 2.4196601
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.2168221, 2.2188802
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.1047378, 2.1045914
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.0677290, 2.0647964
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.4342446, 2.4381795
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.2092061, 2.2081327
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0238781, 3.0313325
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7079015, 1.7060264
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.3150339, 2.3118000
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3912854, 2.3947921

Time for backsubstitution: 12.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 551

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4627

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0691325, upper bound: 1.0585763
time: 4.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0700923, upper bound: 1.0576250
time: 4.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.3828201, 2.3645456
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.2009673, 2.2010922
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0904703, 2.0827842
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9847364, 1.9881759
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.4345660, 2.4371552
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1881313, 2.1831048
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0189590, 3.0215607
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7142467, 1.7163136
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2668676, 2.2500863
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3850403, 2.3937697

Time for backsubstitution: 12.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 943

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0645855, upper bound: 1.0706346
time: 4.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0675253, upper bound: 1.0676876
time: 4.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.3718681, 2.3755047
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.2009463, 2.2011135
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0808611, 2.0923953
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9735713, 1.9993380
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.4377828, 2.4339380
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1769934, 2.1942432
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0291080, 3.0114121
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7171535, 1.7134069
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2578263, 2.2591405
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3914394, 2.3873730

Time for backsubstitution: 13.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5751

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0712144, upper bound: 1.0670182
time: 4.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0706292, upper bound: 1.0670174
time: 4.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.3369379, 2.3676620
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1823149, 2.1668305
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0936322, 2.0865977
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.0025291, 1.9867656
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.4237447, 2.4089608
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1686091, 2.1787448
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0294371, 3.0174203
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7132015, 1.7154747
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2540855, 2.2699482
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3934307, 2.3919420

Time for backsubstitution: 13.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 874

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0644910, upper bound: 1.0712031
time: 4.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0676819, upper bound: 1.0679963
time: 4.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.3577213, 2.3468790
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1644373, 2.1847081
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0904331, 2.0897970
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.0016007, 1.9876941
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.4087214, 2.4239841
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1831141, 2.1642399
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0257282, 3.0211296
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7155089, 1.7131672
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2512426, 2.2727909
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3949118, 2.3904610

Time for backsubstitution: 13.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 874

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0706217, upper bound: 1.0676772
time: 4.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0706218, upper bound: 1.0682665
time: 4.95 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 22.82 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.82
Output dim: 7, lower bound: -1.0554088, upper bound: 1.0694322
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.82
Output dim: 7, lower bound: -1.0544074, upper bound: 1.0704125
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.82
Output dim: 7, lower bound: -1.0548886, upper bound: 1.0673005
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.82
Output dim: 7, lower bound: -1.0585684, upper bound: 1.0636089
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.82
Output dim: 7, lower bound: -1.0667263, upper bound: 1.0582208
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.82
Output dim: 7, lower bound: -1.0704278, upper bound: 1.0546029
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.82
Output dim: 7, lower bound: -1.0691325, upper bound: 1.0585763
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.82
Output dim: 7, lower bound: -1.0700923, upper bound: 1.0576250
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.82
Output dim: 7, lower bound: -1.0645855, upper bound: 1.0706346
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.82
Output dim: 7, lower bound: -1.0675253, upper bound: 1.0676876
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.82
Output dim: 7, lower bound: -1.0712144, upper bound: 1.0670182
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.82
Output dim: 7, lower bound: -1.0706292, upper bound: 1.0670174
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.82
Output dim: 7, lower bound: -1.0644910, upper bound: 1.0712031
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.82
Output dim: 7, lower bound: -1.0676819, upper bound: 1.0679963
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.82
Output dim: 7, lower bound: -1.0706217, upper bound: 1.0676772
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.82
Output dim: 7, lower bound: -1.0706218, upper bound: 1.0682665

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.4130187, 2.4095149
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.2217531, 2.2151892
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0945506, 2.0983610
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.0454149, 2.0555921
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.4315653, 2.4316254
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1655254, 2.1786895
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0394688, 3.0467157
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7074509, 1.7081368
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2984300, 2.2835398
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3908410, 2.3903198

Time for backsubstitution: 13.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4627

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0544558, upper bound: 1.0694315
time: 6.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0554077, upper bound: 1.0684691
time: 4.21 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.4111643, 2.4101605
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.2214832, 2.2152839
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0936742, 2.0986643
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.0465460, 2.0523589
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.4303236, 2.4320602
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1668620, 2.1748700
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0404282, 3.0439744
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7054377, 1.7088432
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2982602, 2.2835984
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3919425, 2.3871760

Time for backsubstitution: 13.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 4627

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0514191, upper bound: 1.0704065
time: 7.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0544003, upper bound: 1.0675961
time: 4.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.4093862, 2.3937502
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.2201080, 2.2167177
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0932574, 2.0829005
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.0462923, 2.0345898
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.4305553, 2.4267540
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1732278, 2.1507411
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0228901, 3.0450945
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7020845, 1.7079161
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2877774, 2.2785835
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3823538, 2.3868980

Time for backsubstitution: 12.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0519061, upper bound: 1.0672932
time: 7.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0548815, upper bound: 1.0643995
time: 4.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.3984270, 2.4047084
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.2200871, 2.2167387
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0836473, 2.0925100
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.0351276, 2.0457549
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.4337730, 2.4235368
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1620903, 2.1618781
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0330391, 3.0349445
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7049913, 1.7050092
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2787232, 2.2876320
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3887534, 2.3804984

Time for backsubstitution: 13.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 4627

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 551

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0585678, upper bound: 1.0631056
time: 7.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0579908, upper bound: 1.0636084
time: 6.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.4151750, 2.4055762
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.2165804, 2.2190919
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.1012120, 2.0916328
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.0637755, 2.0495992
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.4305825, 2.4363451
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.2048917, 2.1933522
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0099945, 3.0278387
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7040181, 1.7049248
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.3120890, 2.2991951
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3827143, 2.3923979

Time for backsubstitution: 13.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 4627

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0639162, upper bound: 1.0582150
time: 4.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0667196, upper bound: 1.0552475
time: 4.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.4042163, 2.4165325
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.2165589, 2.2191131
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0916018, 2.1012425
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.0526109, 2.0607641
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.4338002, 2.4331279
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1937547, 2.2044897
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0201435, 3.0176888
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7069249, 1.7020180
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.3030353, 2.3082438
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3891125, 2.3859987

Time for backsubstitution: 13.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 551

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0672999, upper bound: 1.0545943
time: 4.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0704211, upper bound: 1.0514546
time: 4.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.3053794, 2.3289921
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.2229533, 2.2206213
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.1055675, 2.1052341
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.0561121, 2.0501268
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3998270, 2.3912807
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1799092, 2.1898999
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0355034, 3.0465579
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7090130, 1.7068868
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.3156400, 2.3125842
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3919959, 2.3923063

Time for backsubstitution: 13.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 5857

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0659868, upper bound: 1.0585667
time: 4.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0691258, upper bound: 1.0554183
time: 4.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.3292384, 2.3051302
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.2185631, 2.2250113
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.1053805, 2.1054211
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.0530593, 2.0531800
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3873453, 2.4037609
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1909738, 2.1788359
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0391045, 3.0429578
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7087622, 1.7071381
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.3158183, 2.3124061
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3887997, 2.3955026

Time for backsubstitution: 13.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 551

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0698623, upper bound: 1.0570366
time: 4.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0698596, upper bound: 1.0576247
time: 4.17 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.3416233, 2.3441319
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1834307, 2.1656778
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0873165, 2.0764313
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9838185, 1.9863294
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.4198155, 2.4073820
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1593862, 2.1688647
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0152931, 3.0141859
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7096577, 1.7140322
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2640438, 2.2444196
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3820949, 2.3923056

Time for backsubstitution: 13.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 4627

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0645844, upper bound: 1.0706180
time: 4.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0640704, upper bound: 1.0706213
time: 4.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.3624067, 2.3233490
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1655531, 2.1835556
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0841174, 2.0796306
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9828901, 1.9872580
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.4047923, 2.4224048
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1738911, 2.1543598
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0115833, 3.0178957
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7119651, 1.7117248
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2612009, 2.2472625
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3835759, 2.3908248

Time for backsubstitution: 13.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 5751

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 943

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0672913, upper bound: 1.0653859
time: 4.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0672913, upper bound: 1.0638730
time: 4.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.3737569, 2.3781881
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1998024, 2.2002046
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0808582, 2.0924811
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9737089, 1.9994359
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.4351492, 2.4306154
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1777883, 2.1953752
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0290990, 3.0115228
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7173643, 1.7135540
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2575874, 2.2586036
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3915424, 2.3873672

Time for backsubstitution: 13.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 5751

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0680010, upper bound: 1.0670132
time: 4.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0712076, upper bound: 1.0638026
time: 4.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.3729796, 2.3773942
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1995673, 2.1999695
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0809469, 2.0923924
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9736693, 1.9993949
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.4344606, 2.4313178
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1781278, 2.1950381
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0292192, 3.0114031
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7173004, 1.7136172
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2572894, 2.2589095
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3914337, 2.3872590

Time for backsubstitution: 13.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 943

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0676757, upper bound: 1.0670131
time: 4.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0706227, upper bound: 1.0640673
time: 4.14 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.3296690, 2.3619070
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1843066, 2.1673598
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0835495, 2.0792317
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9845278, 1.9725041
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.4138870, 2.4032636
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1248083, 2.1452940
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0458994, 3.0301075
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7135046, 1.7156649
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2339807, 2.2425530
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3912711, 2.3888779

Time for backsubstitution: 13.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 4627

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 874

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0518424, upper bound: 1.0704173
time: 5.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0638173, upper bound: 1.0585617
time: 4.47 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.3311825, 2.3603935
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1828442, 2.1688221
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0862665, 2.0765154
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9882677, 1.9687643
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.4180470, 2.3991032
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1351557, 2.1349442
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0421247, 3.0338821
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7133920, 1.7157775
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2266903, 2.2498431
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3903661, 2.3897824

Time for backsubstitution: 13.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 5857

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4627

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0667261, upper bound: 1.0679956
time: 4.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0676809, upper bound: 1.0670449
time: 4.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.3596077, 2.3479712
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1632938, 2.1833255
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0904293, 2.0898817
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.0016577, 1.9877923
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.4060912, 2.4206662
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1839128, 2.1653705
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0257177, 3.0212388
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7157183, 1.7133149
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2510056, 2.2722540
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3947945, 2.3904548

Time for backsubstitution: 13.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 4627

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 943

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0676799, upper bound: 1.0653849
time: 4.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0683301, upper bound: 1.0647384
time: 4.17 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.3604212, 2.3487651
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1635332, 2.1835644
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0905180, 2.0897932
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.0016987, 1.9878320
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.4054036, 2.4213696
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1842599, 2.1650386
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0258379, 3.0211196
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7156572, 1.7133794
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2507057, 2.2725608
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3949051, 2.3905668

Time for backsubstitution: 13.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 5857

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4627

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0696624, upper bound: 1.0682642
time: 4.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0706198, upper bound: 1.0673086
time: 4.28 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 22.02 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.02
Output dim: 7, lower bound: -1.0544558, upper bound: 1.0694315
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.02
Output dim: 7, lower bound: -1.0554077, upper bound: 1.0684691
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.02
Output dim: 7, lower bound: -1.0514191, upper bound: 1.0704065
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.02
Output dim: 7, lower bound: -1.0544003, upper bound: 1.0675961
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.02
Output dim: 7, lower bound: -1.0519061, upper bound: 1.0672932
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.02
Output dim: 7, lower bound: -1.0548815, upper bound: 1.0643995
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.02
Output dim: 7, lower bound: -1.0585678, upper bound: 1.0631056
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.02
Output dim: 7, lower bound: -1.0579908, upper bound: 1.0636084
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.02
Output dim: 7, lower bound: -1.0639162, upper bound: 1.0582150
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.02
Output dim: 7, lower bound: -1.0667196, upper bound: 1.0552475
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.02
Output dim: 7, lower bound: -1.0672999, upper bound: 1.0545943
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.02
Output dim: 7, lower bound: -1.0704211, upper bound: 1.0514546
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.02
Output dim: 7, lower bound: -1.0659868, upper bound: 1.0585667
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.02
Output dim: 7, lower bound: -1.0691258, upper bound: 1.0554183
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.02
Output dim: 7, lower bound: -1.0698623, upper bound: 1.0570366
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.02
Output dim: 7, lower bound: -1.0698596, upper bound: 1.0576247
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.02
Output dim: 7, lower bound: -1.0645844, upper bound: 1.0706180
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.02
Output dim: 7, lower bound: -1.0640704, upper bound: 1.0706213
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.02
Output dim: 7, lower bound: -1.0672913, upper bound: 1.0653859
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.02
Output dim: 7, lower bound: -1.0672913, upper bound: 1.0638730
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.02
Output dim: 7, lower bound: -1.0680010, upper bound: 1.0670132
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.02
Output dim: 7, lower bound: -1.0712076, upper bound: 1.0638026
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.02
Output dim: 7, lower bound: -1.0676757, upper bound: 1.0670131
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.02
Output dim: 7, lower bound: -1.0706227, upper bound: 1.0640673
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.02
Output dim: 7, lower bound: -1.0518424, upper bound: 1.0704173
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.02
Output dim: 7, lower bound: -1.0638173, upper bound: 1.0585617
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.02
Output dim: 7, lower bound: -1.0667261, upper bound: 1.0679956
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.02
Output dim: 7, lower bound: -1.0676809, upper bound: 1.0670449
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.02
Output dim: 7, lower bound: -1.0676799, upper bound: 1.0653849
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.02
Output dim: 7, lower bound: -1.0683301, upper bound: 1.0647384
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.02
Output dim: 7, lower bound: -1.0696624, upper bound: 1.0682642
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.02
Output dim: 7, lower bound: -1.0706198, upper bound: 1.0673086

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.2984939, 2.3188326
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.2278719, 2.2169213
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0953808, 2.0990043
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.0337994, 2.0409231
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3971562, 2.3847346
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1362400, 2.1604652
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0510921, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7085629, 1.7089968
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2990351, 2.2843227
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3915491, 2.3878341

Time for backsubstitution: 13.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 943

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0514738, upper bound: 1.0671329
time: 4.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0521490, upper bound: 1.0664316
time: 6.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.3223557, 2.2949901
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.2234850, 2.2213111
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0951939, 2.0991879
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.0307462, 2.0439773
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3846745, 2.3972178
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1473045, 2.1494040
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0546942, 3.0562620
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7083111, 1.7092481
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2992134, 2.2841446
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3883557, 2.3910303

Time for backsubstitution: 13.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 943

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 551

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0554072, upper bound: 1.0679109
time: 4.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0548227, upper bound: 1.0684687
time: 4.27 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.3699660, 2.3897457
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.2039471, 2.1798697
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0905204, 2.0923109
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.0456281, 2.0505123
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.4155741, 2.4022875
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1381173, 2.1606328
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0367632, 3.0365996
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7008486, 1.7065619
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2954364, 2.2779322
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3889980, 2.3857121

Time for backsubstitution: 13.38 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=4, k_high=7, k_mid=5, eps_mid=0.0195312, abs_max=1.7166380882263184
rel_dist={7: [-1.071222731717584, 1.071219330481989]}

## Binary search (step 2) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4627
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 6235
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 5857

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4627

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9696684, upper bound: 0.9703751
time: 4.40 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9703739, upper bound: 0.9696669
time: 4.90 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 9.31 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 9.31
Output dim: 7, lower bound: -0.9696684, upper bound: 0.9703751
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 9.31
Output dim: 7, lower bound: -0.9703739, upper bound: 0.9696669

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.2088566, 2.2279458
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1399508, 2.1364386
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0365939, 2.0364447
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9612706, 1.9588280
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3387599, 2.3287735
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1029444, 2.1117957
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9852285, 2.9881082
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6698034, 1.6696028
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2361956, 2.2363379
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3536029, 2.3510461

Time for backsubstitution: 13.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 6235
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 5751

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 874

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9585131, upper bound: 0.9700074
time: 9.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9692993, upper bound: 0.9592030
time: 4.61 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.2279458, 2.2088566
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1364384, 2.1399505
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0364447, 2.0365939
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9588282, 1.9612706
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3287740, 2.3387594
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1117954, 2.1029441
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9881077, 2.9852285
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6696031, 1.6698037
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2363377, 2.2361956
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3510461, 2.3536029

Time for backsubstitution: 13.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 6235
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9677508, upper bound: 0.9696608
time: 4.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9703678, upper bound: 0.9670161
time: 4.61 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 22.13 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 22.13
Output dim: 7, lower bound: -0.9585131, upper bound: 0.9700074
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 22.13
Output dim: 7, lower bound: -0.9692993, upper bound: 0.9592030
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 22.13
Output dim: 7, lower bound: -0.9677508, upper bound: 0.9696608
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 22.13
Output dim: 7, lower bound: -0.9703678, upper bound: 0.9670161

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.2066503, 2.2242556
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1574311, 2.1510694
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0358877, 2.0365491
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9665744, 1.9666009
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3383818, 2.3259535
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1029577, 2.1117048
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9832530, 2.9864149
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6583853, 1.6600835
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2283831, 2.2269673
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3412824, 2.3362627

Time for backsubstitution: 12.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6235
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 551

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9585102, upper bound: 0.9695570
time: 7.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9580600, upper bound: 0.9700071
time: 4.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.2051663, 2.2257400
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1545811, 2.1539192
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0366983, 2.0357385
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9690435, 1.9641314
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3359394, 2.3283954
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1028533, 2.1118093
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9835353, 2.9861321
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6602845, 1.6581842
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2268248, 2.2285256
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3388195, 2.3387256

Time for backsubstitution: 12.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6235
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 6208

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6235

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9692988, upper bound: 0.9589964
time: 4.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9691206, upper bound: 0.9592028
time: 6.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.1867185, 2.1842561
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1153283, 2.1045384
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0326505, 2.0302401
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9577239, 1.9594240
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3110156, 2.3089833
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.0830479, 2.0858006
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9837022, 2.9778528
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6650140, 1.6670609
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2329464, 2.2305298
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3481011, 2.3518443

Time for backsubstitution: 12.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 6235
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 551

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5857

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9648774, upper bound: 0.9696591
time: 4.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9677489, upper bound: 0.9668013
time: 4.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.2033248, 2.1676292
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1010265, 2.1188371
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0300908, 2.0328002
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9569814, 1.9601657
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.2989974, 2.3209901
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.0946479, 2.0741968
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9807334, 2.9808207
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6668603, 1.6652150
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2306719, 2.2328041
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3492861, 2.3506579

Time for backsubstitution: 13.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 6235
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 874

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9675626, upper bound: 0.9670078
time: 7.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9703614, upper bound: 0.9642265
time: 4.52 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 25.28 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.28
Output dim: 7, lower bound: -0.9585102, upper bound: 0.9695570
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.28
Output dim: 7, lower bound: -0.9580600, upper bound: 0.9700071
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.28
Output dim: 7, lower bound: -0.9692988, upper bound: 0.9589964
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.28
Output dim: 7, lower bound: -0.9691206, upper bound: 0.9592028
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.28
Output dim: 7, lower bound: -0.9648774, upper bound: 0.9696591
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.28
Output dim: 7, lower bound: -0.9677489, upper bound: 0.9668013
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.28
Output dim: 7, lower bound: -0.9675626, upper bound: 0.9670078
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.28
Output dim: 7, lower bound: -0.9703614, upper bound: 0.9642265

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.1720891, 2.1827903
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1580596, 2.1507993
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0240731, 2.0270381
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.8959460, 1.9077491
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3366394, 2.3238635
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.0853205, 2.0982604
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9820280, 2.9854941
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6588025, 1.6601784
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.1755018, 2.1635079
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3288789, 2.3259802

Time for backsubstitution: 12.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 6235
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9585094, upper bound: 0.9695330
time: 6.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9580388, upper bound: 0.9695367
time: 5.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.1651850, 2.1896939
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1571612, 2.1516976
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0263767, 2.0247347
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9077220, 1.8959732
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3362913, 2.3242111
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.0895133, 2.0940671
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9823322, 2.9851899
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6584802, 1.6605006
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.1649237, 2.1740856
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3309994, 2.3238592

Time for backsubstitution: 12.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 6235
type: RSZ, layer: 1, pos: 6208

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9552594, upper bound: 0.9699990
time: 6.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9580519, upper bound: 0.9672166
time: 6.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.1998534, 2.2193809
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1495032, 2.1478257
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0363832, 2.0361876
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9684880, 1.9636683
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3356676, 2.3282795
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1023145, 2.1105723
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9455051, 2.9544435
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6609066, 1.6591560
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2235084, 2.2208233
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3383608, 2.3385215

Time for backsubstitution: 12.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 551

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9692964, upper bound: 0.9585428
time: 5.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9688464, upper bound: 0.9589945
time: 4.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.1988072, 2.2204270
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1484876, 2.1488411
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0371480, 2.0354230
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9685805, 1.9635761
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3358231, 2.3281236
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.1016169, 2.1112700
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9518461, 2.9481020
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6612566, 1.6588062
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2191229, 2.2252090
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3386154, 2.3382668

Time for backsubstitution: 12.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 6181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5857

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9662644, upper bound: 0.9592014
time: 4.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9691188, upper bound: 0.9563457
time: 6.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.1805887, 2.1693592
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1153164, 2.1045096
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0272903, 2.0171919
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9514976, 1.9442654
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3066535, 2.3071947
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.0768399, 2.0706825
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9699378, 2.9722075
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6610668, 1.6654392
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2278805, 2.2182248
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3394179, 2.3482804

Time for backsubstitution: 12.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6235
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 943

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9648768, upper bound: 0.9693978
time: 4.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9645909, upper bound: 0.9696578
time: 4.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.1718216, 2.1781261
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1152997, 2.1045265
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0196018, 2.0248804
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9425654, 1.9531975
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3092265, 2.3046203
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.0679297, 2.0795922
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9780564, 2.9640884
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6633923, 1.6631138
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2206411, 2.2254636
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3445377, 2.3431606

Time for backsubstitution: 12.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 6235
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 551

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5751

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9677371, upper bound: 0.9657404
time: 4.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9666965, upper bound: 0.9667924
time: 5.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.1960564, 2.1615717
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1027150, 2.1193559
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0200076, 2.0248878
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9389796, 1.9451568
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.2891474, 2.3144693
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.0508547, 2.0386815
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9964390, 2.9935050
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6671395, 1.6654044
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2091098, 2.2054098
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3469453, 2.3475933

Time for backsubstitution: 12.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 6235
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 551

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9675600, upper bound: 0.9665621
time: 7.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9671099, upper bound: 0.9670068
time: 4.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.1972518, 2.1603608
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1015449, 2.1205230
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0221810, 2.0227172
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9419713, 1.9421639
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.2924747, 2.3111401
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.0591297, 2.0304036
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9934177, 2.9965248
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6670494, 1.6654948
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2032781, 2.2112420
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3462214, 2.3483152

Time for backsubstitution: 12.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 6235

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5857

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9675044, upper bound: 0.9642213
time: 5.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9703596, upper bound: 0.9613473
time: 4.98 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 22.50 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.50
Output dim: 7, lower bound: -0.9585094, upper bound: 0.9695330
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.50
Output dim: 7, lower bound: -0.9580388, upper bound: 0.9695367
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.50
Output dim: 7, lower bound: -0.9552594, upper bound: 0.9699990
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.50
Output dim: 7, lower bound: -0.9580519, upper bound: 0.9672166
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.50
Output dim: 7, lower bound: -0.9692964, upper bound: 0.9585428
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.50
Output dim: 7, lower bound: -0.9688464, upper bound: 0.9589945
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.50
Output dim: 7, lower bound: -0.9662644, upper bound: 0.9592014
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.50
Output dim: 7, lower bound: -0.9691188, upper bound: 0.9563457
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.50
Output dim: 7, lower bound: -0.9648768, upper bound: 0.9693978
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.50
Output dim: 7, lower bound: -0.9645909, upper bound: 0.9696578
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.50
Output dim: 7, lower bound: -0.9677371, upper bound: 0.9657404
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.50
Output dim: 7, lower bound: -0.9666965, upper bound: 0.9667924
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.50
Output dim: 7, lower bound: -0.9675600, upper bound: 0.9665621
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.50
Output dim: 7, lower bound: -0.9671099, upper bound: 0.9670068
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.50
Output dim: 7, lower bound: -0.9675044, upper bound: 0.9642213
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.50
Output dim: 7, lower bound: -0.9703596, upper bound: 0.9613473

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.1739745, 2.1853247
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1569161, 2.1498477
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0240655, 2.0271010
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.8960752, 1.9078462
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3338757, 2.3205395
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.0861144, 2.0993295
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9820185, 2.9855804
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6590033, 1.6603277
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.1751719, 2.1629395
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3289638, 2.3259749

Time for backsubstitution: 12.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6235
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 6208

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6235

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9585090, upper bound: 0.9693531
time: 5.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9583103, upper bound: 0.9695324
time: 4.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.1733527, 2.1846757
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1567287, 2.1496558
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0241365, 2.0270302
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.8960433, 1.9078134
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3333149, 2.3211012
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.0863857, 2.0990543
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9821138, 2.9854846
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6589518, 1.6603782
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.1749334, 2.1631794
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3288741, 2.3258886

Time for backsubstitution: 12.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6235
type: RSZ, layer: 1, pos: 6181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5857

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9551823, upper bound: 0.9695348
time: 6.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9580370, upper bound: 0.9666779
time: 9.07 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.1579142, 2.1836190
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1588516, 2.1522207
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0162969, 2.0168290
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.8897233, 1.8809680
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3264365, 2.3176837
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.0457196, 2.0585489
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9980345, 2.9978728
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6587596, 1.6606902
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.1433687, 2.1466980
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3286562, 2.3207941

Time for backsubstitution: 12.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6235
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 6208

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9552392, upper bound: 0.9695282
time: 6.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9552391, upper bound: 0.9699979
time: 9.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.1591253, 2.1824231
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1576843, 2.1533906
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0184674, 2.0146551
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.8927164, 1.8779749
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3297658, 2.3143559
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.0539975, 2.0502739
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9950151, 3.0008945
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6586699, 1.6607803
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.1375365, 2.1525304
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3279343, 2.3215179

Time for backsubstitution: 12.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6235
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 6208

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5857

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9551952, upper bound: 0.9672178
time: 5.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.9580500, upper bound: 0.9643589
time: 5.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.1652937, 2.1779149
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1501331, 2.1475568
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0245709, 2.0266786
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.8978605, 1.9048167
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3339233, 2.3261876
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.0846748, 2.0971260
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9442859, 2.9535289
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6613228, 1.6592500
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.1706395, 2.1573758
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3259578, 2.3282399

Time for backsubstitution: 12.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5751

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9692841, upper bound: 0.9574543
time: 5.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9682264, upper bound: 0.9585305
time: 5.46 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.1583872, 2.1848185
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1492343, 2.1484554
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0268741, 2.0243752
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9096365, 1.8930409
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3335762, 2.3265347
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.0888662, 2.0929332
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9445901, 2.9532242
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6610010, 1.6595721
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.1600609, 2.1679535
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3280783, 2.3261185

Time for backsubstitution: 12.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6208

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5751

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9688342, upper bound: 0.9579064
time: 4.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9677765, upper bound: 0.9589824
time: 4.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.1926770, 2.2055304
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1484766, 2.1488130
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0317874, 2.0223749
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9623539, 1.9484177
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3314590, 2.3263340
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.0954084, 2.0961518
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9380827, 2.9424567
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6573102, 1.6571851
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2140565, 2.2129045
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3299308, 2.3347020

Time for backsubstitution: 12.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 943

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.9634939, upper bound: 0.9591927
time: 7.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9662580, upper bound: 0.9563868
time: 4.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.1839104, 2.2142978
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1484594, 2.1488299
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0240998, 2.0300632
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9534218, 1.9573498
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3340330, 2.3237600
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.0864987, 2.1050615
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9462023, 2.9343381
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6596353, 1.6548597
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2068181, 2.2201476
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3350506, 2.3295822

Time for backsubstitution: 12.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 6181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9691179, upper bound: 0.9560874
time: 5.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9688526, upper bound: 0.9563483
time: 4.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.1824727, 2.1718943
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1158843, 2.1052697
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0268631, 2.0168362
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9516263, 1.9443626
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3038940, 2.3038735
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.0751786, 2.0692949
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9693747, 2.9717402
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6608057, 1.6651272
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2266622, 2.2167661
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3397679, 2.3485413

Time for backsubstitution: 12.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 6235

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9610635, upper bound: 0.9693912
time: 4.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9648708, upper bound: 0.9666131
time: 4.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.1831198, 2.1712437
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1160769, 2.1050775
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0269346, 2.0167651
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9515948, 1.9443941
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3033323, 2.3044333
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.0754499, 2.0690212
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9694700, 2.9716454
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6607552, 1.6651777
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2264214, 2.2170060
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3396783, 2.3486295

Time for backsubstitution: 12.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 6235
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 5751

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 874

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9534669, upper bound: 0.9692897
time: 6.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.9642368, upper bound: 0.9585024
time: 6.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.1726594, 2.1774800
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1154208, 2.1044319
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0200000, 2.0245774
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9414334, 1.9546521
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3097849, 2.3041849
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.0665936, 2.0813112
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9770961, 2.9653215
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6642966, 1.6624076
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2207193, 2.2254057
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3434353, 2.3445730

Time for backsubstitution: 12.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6235
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 6181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 551

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9677367, upper bound: 0.9652846
time: 5.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9672876, upper bound: 0.9657387
time: 5.07 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.1711760, 2.1781261
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1152053, 2.1045265
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0192990, 2.0248804
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9425654, 1.9520655
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3087912, 2.3046203
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.0679297, 2.0782557
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9780564, 2.9631286
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6626863, 1.6631138
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.2205830, 2.2254636
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3445377, 2.3420582

Time for backsubstitution: 12.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 6235
type: RSZ, layer: 1, pos: 6181
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 551

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9666961, upper bound: 0.9663392
time: 4.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9662455, upper bound: 0.9667919
time: 5.17 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.1614957, 2.1201077
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1033430, 2.1190853
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0081930, 2.0153761
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.8683522, 1.8863052
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.2874041, 2.3123789
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.0332174, 2.0252364
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9952135, 2.9925847
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6675572, 1.6654997
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.1562285, 2.1419504
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3345418, 2.3373108

Time for backsubstitution: 12.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 6235
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 5857

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 874

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9563814, upper bound: 0.9662075
time: 6.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9672120, upper bound: 0.9554310
time: 4.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.1545920, 2.1270115
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1024446, 2.1199839
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0104961, 2.0130730
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.8801281, 1.8745295
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.2870569, 2.3127260
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.0374098, 2.0210435
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9955187, 2.9922800
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6672349, 1.6658220
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.1456509, 2.1525280
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3366628, 2.3351903

Time for backsubstitution: 12.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 6235
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 874

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9559314, upper bound: 0.9666538
time: 5.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9667615, upper bound: 0.9558772
time: 4.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.1911225, 2.1454647
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1015339, 2.1204944
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0168214, 2.0096695
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9357445, 1.9270053
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.2881126, 2.3093510
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.0529213, 2.0152855
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9796534, 2.9908795
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6631026, 1.6638734
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.1982117, 2.1989369
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3375368, 2.3447504

Time for backsubstitution: 12.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 6235
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 551

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9675019, upper bound: 0.9637780
time: 4.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9670513, upper bound: 0.9642210
time: 5.14 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.1823559, 2.1542315
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1015167, 2.1205113
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0091333, 2.0173578
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.9268124, 1.9359374
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.2906857, 2.3067770
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.0440116, 2.0241952
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9877720, 2.9827609
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6654282, 1.6615480
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.1909728, 2.2061758
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3426566, 2.3396311

Time for backsubstitution: 12.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 6235
type: RSZ, layer: 1, pos: 943

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9703589, upper bound: 0.9610604
time: 4.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9701015, upper bound: 0.9613498
time: 5.30 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 22.59 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 7, lower bound: -0.9585090, upper bound: 0.9693531
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 7, lower bound: -0.9583103, upper bound: 0.9695324
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 7, lower bound: -0.9551823, upper bound: 0.9695348
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 7, lower bound: -0.9580370, upper bound: 0.9666779
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 7, lower bound: -0.9552392, upper bound: 0.9695282
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 7, lower bound: -0.9552391, upper bound: 0.9699979
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 7, lower bound: -0.9551952, upper bound: 0.9672178
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 22.59
Output dim: 7, lower bound: -0.9580500, upper bound: 0.9643589
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 7, lower bound: -0.9692841, upper bound: 0.9574543
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 7, lower bound: -0.9682264, upper bound: 0.9585305
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 7, lower bound: -0.9688342, upper bound: 0.9579064
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 7, lower bound: -0.9677765, upper bound: 0.9589824
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 22.59
Output dim: 7, lower bound: -0.9634939, upper bound: 0.9591927
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 7, lower bound: -0.9662580, upper bound: 0.9563868
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 7, lower bound: -0.9691179, upper bound: 0.9560874
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 7, lower bound: -0.9688526, upper bound: 0.9563483
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 7, lower bound: -0.9610635, upper bound: 0.9693912
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 7, lower bound: -0.9648708, upper bound: 0.9666131
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 7, lower bound: -0.9534669, upper bound: 0.9692897
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 22.59
Output dim: 7, lower bound: -0.9642368, upper bound: 0.9585024
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 7, lower bound: -0.9677367, upper bound: 0.9652846
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 7, lower bound: -0.9672876, upper bound: 0.9657387
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 7, lower bound: -0.9666961, upper bound: 0.9663392
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 7, lower bound: -0.9662455, upper bound: 0.9667919
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 7, lower bound: -0.9563814, upper bound: 0.9662075
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 7, lower bound: -0.9672120, upper bound: 0.9554310
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 7, lower bound: -0.9559314, upper bound: 0.9666538
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 7, lower bound: -0.9667615, upper bound: 0.9558772
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 7, lower bound: -0.9675019, upper bound: 0.9637780
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 7, lower bound: -0.9670513, upper bound: 0.9642210
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 7, lower bound: -0.9703589, upper bound: 0.9610604
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.59
Output dim: 7, lower bound: -0.9701015, upper bound: 0.9613498

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.1686621, 2.1789632
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1518393, 2.1437542
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0237560, 2.0275564
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.8955209, 1.9073842
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3336029, 2.3204217
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.0855727, 2.0980906
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9439940, 2.9538984
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6596227, 1.6612968
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.1719055, 2.1552806
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3285065, 2.3257728

Time for backsubstitution: 12.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5857
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 5751
type: RSZ, layer: 1, pos: 6181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5857

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9556523, upper bound: 0.9693481
time: 5.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9585072, upper bound: 0.9664971
time: 5.20 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.2881069, -6.2814474, -9.2881069, -6.2814474, -2.1676130, 2.1799960
1: -6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.1508226, 2.1447659
2: -8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.0245204, 2.0267916
3: -10.1440105, -7.5077662, -10.1440105, -7.5077662, -1.8956130, 1.9072920
4: -5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.3337469, 2.3202662
5: -5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.0848751, 2.0987806
6: -13.7059364, -10.6496744, -13.7059364, -10.6496744, -2.9503350, 2.9475565
7: 3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.6599712, 1.6609471
8: -4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.1675129, 2.1596661
9: -2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.3287587, 2.3255177

Time for backsubstitution: 12.36 seconds
Binary search (step 2): status=Status.UNKNOWN, k_low=4, k_high=4, k_mid=4, eps_mid=0.0156250, abs_max=1.6687421798706055
rel_dist={7: [-0.9703767544746134, 0.9703745834961706]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01171875
execution time: 2409.23 seconds
