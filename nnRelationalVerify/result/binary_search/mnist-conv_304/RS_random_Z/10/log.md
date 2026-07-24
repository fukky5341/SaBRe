## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 1.0070000638
Search space: {k/256 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-14.2722282, -11.0764151, -14.2722282, -11.0764151, -3.1958132, 3.1958132)
1: (-10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.7144113, 2.7144113)
2: (-10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.8229380, 2.8229380)
3: (-12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.4258022, 2.4258022)
4: (5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.5450792, 2.5450792)
5: (-8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.6159039, 2.6159039)
6: (-12.7108383, -9.7072067, -12.7108383, -9.7072067, -3.0036316, 3.0036316)
7: (-6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8832743, 2.8832743)
8: (-3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.7740321, 2.7740321)
9: (-5.4689426, -3.2161665, -5.4689426, -3.2161665, -2.2485318, 2.2485321)

## BASE Result
execution time: IAR + LP analysis = 14.26 + 33.04 = 47.29 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3552.71 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.4249300956726074
rel_dist={4: [-1.3292876093097172, 1.3292852400204573]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=2.249711513519287
rel_dist={4: [-1.0090186137045807, 1.0090191641306294]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=2.132899761199951
rel_dist={4: [-0.7676964433562325, 0.7676953753322513]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=2.191305637359619
rel_dist={4: [-0.8893789777736325, 0.8893816357061688]}

## Binary Search Result
Binary search time: 205.51 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Relational Split (RS_random_Z) starts
Time budget: 3347.20 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 4569

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5805

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4212995, upper bound: 1.4104138
time: 6.13 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4104139, upper bound: 1.4212996
time: 4.55 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 10.70 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 10.70
Output dim: 4, lower bound: -1.4212995, upper bound: 1.4104138
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 10.70
Output dim: 4, lower bound: -1.4104139, upper bound: 1.4212996

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -3.0064774, 3.0089166
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.3421755, 2.3557494
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.6784372, 2.6697659
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.2270861, 2.2313764
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.4806128, 2.4794927
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.2948613, 2.2944412
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.6540403, 2.6508379
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8832743, 2.8832743
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.4718008, 2.4820089
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.9376583, 1.9410462

Time for backsubstitution: 14.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 5735

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4040374, upper bound: 1.4100690
time: 5.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4209811, upper bound: 1.4040356
time: 5.79 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -3.0089169, 3.0064774
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.3557491, 2.3421755
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.6697664, 2.6784365
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.2313766, 2.2270861
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.4794931, 2.4806123
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.2944412, 2.2948613
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.6508379, 2.6540403
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8832743, 2.8832743
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.4820089, 2.4718008
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.9410467, 1.9376583

Time for backsubstitution: 14.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 821

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5735

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4103979, upper bound: 1.3970954
time: 4.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3862674, upper bound: 1.4212834
time: 4.56 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 24.00 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.00
Output dim: 4, lower bound: -1.4040374, upper bound: 1.4100690
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.00
Output dim: 4, lower bound: -1.4209811, upper bound: 1.4040356
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.00
Output dim: 4, lower bound: -1.4103979, upper bound: 1.3970954
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.00
Output dim: 4, lower bound: -1.3862674, upper bound: 1.4212834

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -3.0064678, 3.0089290
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.3421338, 2.3557200
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.6785250, 2.6698248
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.2271285, 2.2314041
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.4806290, 2.4795175
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.2949076, 2.2945027
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.6539974, 2.6508079
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8832743, 2.8832743
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.4718313, 2.4820280
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.9376664, 1.9410524

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 821

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5735

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4149184, upper bound: 1.3861516
time: 5.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3907967, upper bound: 1.4100527
time: 4.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -3.0064898, 3.0089071
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.3421462, 2.3557076
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.6784954, 2.6698546
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.2271137, 2.2314191
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.4806366, 2.4795089
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.2949228, 2.2944870
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.6540098, 2.6507950
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8832743, 2.8832743
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.4718208, 2.4820390
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.9376645, 1.9410543

Time for backsubstitution: 14.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 4560

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 891

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4209772, upper bound: 1.4029061
time: 7.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4089386, upper bound: 1.4029146
time: 4.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -3.0288835, 3.0347085
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.3668077, 2.3573670
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.6716690, 2.6810498
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.2435503, 2.2359490
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.4621401, 2.4561267
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.2984939, 2.3004293
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.6461806, 2.6474667
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8832743, 2.8832743
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.4809961, 2.4703712
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.9112434, 1.9165471

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 4560

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4083371, upper bound: 1.3964532
time: 6.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4097242, upper bound: 1.3949625
time: 8.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -3.0371480, 3.0264440
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.3709409, 2.3532341
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.6723795, 2.6803391
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.2402396, 2.2392600
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.4550066, 2.4632597
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.3000093, 2.2989144
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.6442637, 2.6493833
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8832743, 2.8832743
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.4805784, 2.4707885
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.9199352, 1.9078555

Time for backsubstitution: 14.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 494

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3844057, upper bound: 1.4212839
time: 5.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3862651, upper bound: 1.4195274
time: 4.56 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 24.45 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.45
Output dim: 4, lower bound: -1.4149184, upper bound: 1.3861516
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.45
Output dim: 4, lower bound: -1.3907967, upper bound: 1.4100527
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.45
Output dim: 4, lower bound: -1.4209772, upper bound: 1.4029061
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.45
Output dim: 4, lower bound: -1.4089386, upper bound: 1.4029146
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.45
Output dim: 4, lower bound: -1.4083371, upper bound: 1.3964532
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.45
Output dim: 4, lower bound: -1.4097242, upper bound: 1.3949625
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.45
Output dim: 4, lower bound: -1.3844057, upper bound: 1.4212839
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.45
Output dim: 4, lower bound: -1.3862651, upper bound: 1.4195274

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -3.0264325, 3.0371590
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.3531928, 2.3709118
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.6804256, 2.6724362
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.2393041, 2.2402687
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.4632759, 2.4550319
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.2989597, 2.3000698
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.6493406, 2.6442344
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8832743, 2.8832743
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.4708195, 2.4805989
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.9078622, 1.9199400

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 821

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4560

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4148165, upper bound: 1.3861527
time: 4.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4149183, upper bound: 1.3860476
time: 4.48 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -3.0346971, 3.0288949
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.3573260, 2.3667789
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.6811371, 2.6717255
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.2359934, 2.2435794
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.4561434, 2.4621649
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.3004746, 2.2985549
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.6474237, 2.6461511
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8832743, 2.8832743
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.4704018, 2.4810166
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.9165540, 1.9112484

Time for backsubstitution: 14.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3890015, upper bound: 1.3841216
time: 4.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3780976, upper bound: 1.4100424
time: 5.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -3.0076094, 3.0106087
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.3295875, 2.3475993
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.6782827, 2.6672292
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.2263684, 2.2317221
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.4797029, 2.4781914
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.2953801, 2.2947881
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.6505265, 2.6458843
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8832743, 2.8832743
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.4619823, 2.4750724
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.9401274, 1.9447947

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 494

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4189227, upper bound: 1.4022361
time: 4.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4008603, upper bound: 1.4008495
time: 5.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -3.0081882, 3.0100267
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.3340282, 2.3431492
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.6758699, 2.6696410
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.2274175, 2.2306733
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.4793196, 2.4785748
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.2952237, 2.2949390
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.6490998, 2.6473100
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8832743, 2.8832743
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.4648547, 2.4721999
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.9414053, 1.9435172

Time for backsubstitution: 15.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 821

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 523

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3942217, upper bound: 1.4028941
time: 8.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4089204, upper bound: 1.3882151
time: 6.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -3.0379391, 3.0412939
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.3670511, 2.3565378
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.6716590, 2.6810529
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.2453551, 2.2384362
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.4625330, 2.4564128
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.2975349, 2.2990680
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.6474333, 2.6483846
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8832743, 2.8832743
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.4834032, 2.4721174
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.9126916, 1.9176035

Time for backsubstitution: 14.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4019635, upper bound: 1.3963453
time: 4.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4079958, upper bound: 1.3901545
time: 8.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -3.0354691, 3.0437644
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.3659787, 2.3576102
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.6716723, 2.6810396
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.2460375, 2.2377539
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.4624262, 2.4565196
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.2971325, 2.2994699
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.6470995, 2.6487188
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8832743, 2.8832743
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.4827423, 2.4727783
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.9123001, 1.9179950

Time for backsubstitution: 14.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 821

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4033468, upper bound: 1.3948878
time: 4.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4093263, upper bound: 1.3886650
time: 5.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -3.0232596, 3.0191059
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.3772609, 2.3626740
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.6813831, 2.6870024
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.2406731, 2.2398276
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.4494977, 2.4593568
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.2866831, 2.2801051
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.6143508, 2.6281931
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8832743, 2.8832743
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.4724383, 2.4650168
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.9079943, 1.9023368

Time for backsubstitution: 14.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 891

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4569

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3843832, upper bound: 1.4121019
time: 4.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3753446, upper bound: 1.4212585
time: 4.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -3.0298095, 3.0125561
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.3803809, 2.3595538
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.6790428, 2.6893423
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.2408071, 2.2396934
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.4511037, 2.4577498
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.2811999, 2.2855883
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.6230741, 2.6194704
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8832743, 2.8832743
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.4748073, 2.4626484
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.9144168, 1.8959141

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 891

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4560

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3861337, upper bound: 1.4195273
time: 7.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3862644, upper bound: 1.4193763
time: 4.11 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 26.16 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.16
Output dim: 4, lower bound: -1.4148165, upper bound: 1.3861527
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.16
Output dim: 4, lower bound: -1.4149183, upper bound: 1.3860476
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.16
Output dim: 4, lower bound: -1.3890015, upper bound: 1.3841216
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.16
Output dim: 4, lower bound: -1.3780976, upper bound: 1.4100424
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.16
Output dim: 4, lower bound: -1.4189227, upper bound: 1.4022361
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.16
Output dim: 4, lower bound: -1.4008603, upper bound: 1.4008495
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.16
Output dim: 4, lower bound: -1.3942217, upper bound: 1.4028941
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.16
Output dim: 4, lower bound: -1.4089204, upper bound: 1.3882151
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.16
Output dim: 4, lower bound: -1.4019635, upper bound: 1.3963453
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.16
Output dim: 4, lower bound: -1.4079958, upper bound: 1.3901545
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.16
Output dim: 4, lower bound: -1.4033468, upper bound: 1.3948878
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.16
Output dim: 4, lower bound: -1.4093263, upper bound: 1.3886650
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.16
Output dim: 4, lower bound: -1.3843832, upper bound: 1.4121019
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.16
Output dim: 4, lower bound: -1.3753446, upper bound: 1.4212585
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.16
Output dim: 4, lower bound: -1.3861337, upper bound: 1.4195273
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.16
Output dim: 4, lower bound: -1.3862644, upper bound: 1.4193763

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -3.0114121, 3.0314012
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.3502491, 2.3632247
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.6781487, 2.6664863
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.2384291, 2.2379370
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.4616451, 2.4508190
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.2960057, 2.2923250
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.6483974, 2.6439018
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8832743, 2.8832743
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.4659958, 2.4679842
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.8974018, 1.9159269

Time for backsubstitution: 14.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 891

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4148061, upper bound: 1.3840755
time: 5.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3889250, upper bound: 1.3840826
time: 4.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -3.0206771, 3.0221381
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.3455055, 2.3679664
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.6744761, 2.6701593
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.2369723, 2.2393937
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.4590635, 2.4533992
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.2912149, 2.2971153
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.6490078, 2.6432917
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8832743, 2.8832743
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.4582043, 2.4757752
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.9038491, 1.9094796

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 523

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4002202, upper bound: 1.3860279
time: 4.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4148970, upper bound: 1.3714189
time: 4.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -3.0360537, 3.0309513
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.3593426, 2.3698275
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.6826010, 2.6739435
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.2376466, 2.2446718
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.4517326, 2.4559445
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.3009777, 2.2993169
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.6466980, 2.6451259
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8832743, 2.8832743
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.4702692, 2.4808283
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.9093456, 1.9061430

Time for backsubstitution: 14.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 891

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4560

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3888992, upper bound: 1.3841216
time: 4.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3890013, upper bound: 1.3840760
time: 4.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -3.0367537, 3.0302501
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.3603826, 2.3687954
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.6833544, 2.6731894
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.2370858, 2.2452364
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.4499226, 2.4577565
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.3012366, 2.2990580
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.6463985, 2.6454251
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8832743, 2.8832743
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.4702148, 2.4808836
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.9114485, 1.9040399

Time for backsubstitution: 14.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4560

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3888878, upper bound: 1.4100404
time: 4.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3889937, upper bound: 1.4099562
time: 4.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -3.0166655, 3.0171940
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.3298306, 2.3467698
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.6782727, 2.6672330
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.2281728, 2.2342091
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.4800949, 2.4784765
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.2944212, 2.2934270
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.6517806, 2.6468036
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8832743, 2.8832743
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.4643874, 2.4768181
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.9415755, 1.9458513

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 523

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4569

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4188993, upper bound: 1.3930048
time: 5.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4096916, upper bound: 1.4022107
time: 8.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -3.0141945, 3.0196648
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.3287578, 2.3478425
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.6782861, 2.6672196
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.2288551, 2.2335267
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.4799881, 2.4785833
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.2940187, 2.2938292
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.6514459, 2.6471379
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8832743, 2.8832743
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.4637275, 2.4774785
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.9411840, 1.9462426

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4560

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5735

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4202416, upper bound: 1.3767159
time: 4.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3963412, upper bound: 1.4008328
time: 4.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -3.0201387, 2.9618349
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.3339047, 2.3431602
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.6806998, 2.6501462
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.2260695, 2.2310052
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.4684763, 2.4812555
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.2940660, 2.2952232
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.6662989, 2.5777605
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8832743, 2.8832743
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.4668627, 2.4640379
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.9362998, 1.9447703

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 4560

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5735

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3942055, upper bound: 1.3788378
time: 5.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3704082, upper bound: 1.4028799
time: 5.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.9599962, 3.0100267
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.3340282, 2.3430262
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.6563754, 2.6696410
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.2274175, 2.2293258
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.4793196, 2.4677310
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.2952237, 2.2937813
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.5795498, 2.6473100
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8832743, 2.8832743
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.4566917, 2.4721999
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.9414053, 1.9384117

Time for backsubstitution: 14.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 494

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4072017, upper bound: 1.3882135
time: 4.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4089183, upper bound: 1.3864615
time: 4.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -3.0379276, 3.0413046
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.3670096, 2.3565085
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.6717463, 2.6811109
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.2453990, 2.2384653
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.4625502, 2.4564381
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.2975798, 2.2991285
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.6473904, 2.6483543
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8832743, 2.8832743
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.4834328, 2.4721370
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.9126997, 1.9176102

Time for backsubstitution: 14.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4569

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4019511, upper bound: 1.3942474
time: 6.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3760862, upper bound: 1.3942553
time: 4.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -3.0379505, 3.0412827
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.3670220, 2.3564963
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.6717157, 2.6811404
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.2453837, 2.2384803
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.4625587, 2.4564300
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.2975950, 2.2991130
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.6474028, 2.6483414
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8832743, 2.8832743
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.4834223, 2.4721479
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.9126978, 1.9176121

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 891

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4560

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4079042, upper bound: 1.3901542
time: 6.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4079956, upper bound: 1.3900269
time: 4.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -3.0354576, 3.0437753
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.3659372, 2.3575811
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.6717596, 2.6810975
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.2460814, 2.2377830
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.4624434, 2.4565454
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.2971778, 2.2995305
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.6470556, 2.6486886
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8832743, 2.8832743
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.4827728, 2.4727974
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.9123082, 1.9180017

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 523

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 891

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4022274, upper bound: 1.3829355
time: 6.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4022179, upper bound: 1.3948826
time: 6.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -3.0354795, 3.0437534
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.3659492, 2.3575690
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.6717291, 2.6811271
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.2460661, 2.2377980
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.4624510, 2.4565368
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.2971931, 2.2995150
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.6470680, 2.6486757
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8832743, 2.8832743
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.4827614, 2.4728084
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.9123063, 1.9180036

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 523

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4093141, upper bound: 1.3869442
time: 4.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3833769, upper bound: 1.3869512
time: 6.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -3.0233006, 3.0218439
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.3779032, 2.3631186
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.6424875, 2.6615894
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.2264028, 2.2297020
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.4325590, 2.4354787
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.2842140, 2.2795479
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.6129975, 2.6262312
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8832743, 2.8832743
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.4396181, 2.4417715
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.8794355, 1.8826959

Time for backsubstitution: 14.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 821

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4560

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3842351, upper bound: 1.4121042
time: 5.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3843824, upper bound: 1.4120051
time: 4.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -3.0259976, 3.0191467
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.3777053, 2.3633163
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.6559696, 2.6481075
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.2305474, 2.2255578
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.4256191, 2.4424186
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.2861261, 2.2776361
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.6123891, 2.6268396
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8832743, 2.8832743
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.4491940, 2.4321961
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.8883533, 1.8737786

Time for backsubstitution: 14.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 523

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3735694, upper bound: 1.3953675
time: 9.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3735618, upper bound: 1.4212465
time: 8.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -3.0147882, 3.0067990
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.3774352, 2.3518660
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.6767664, 2.6833928
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.2399335, 2.2373626
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.4494720, 2.4535370
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.2782464, 2.2778449
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.6221309, 2.6191368
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8832743, 2.8832743
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.4699836, 2.4500337
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.9039564, 1.8919015

Time for backsubstitution: 14.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 821

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3836904, upper bound: 1.4188768
time: 4.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3854968, upper bound: 1.4188765
time: 5.94 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 24.98 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.98
Output dim: 4, lower bound: -1.4148061, upper bound: 1.3840755
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.98
Output dim: 4, lower bound: -1.3889250, upper bound: 1.3840826
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.98
Output dim: 4, lower bound: -1.4002202, upper bound: 1.3860279
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.98
Output dim: 4, lower bound: -1.4148970, upper bound: 1.3714189
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.98
Output dim: 4, lower bound: -1.3888992, upper bound: 1.3841216
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.98
Output dim: 4, lower bound: -1.3890013, upper bound: 1.3840760
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.98
Output dim: 4, lower bound: -1.3888878, upper bound: 1.4100404
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.98
Output dim: 4, lower bound: -1.3889937, upper bound: 1.4099562
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.98
Output dim: 4, lower bound: -1.4188993, upper bound: 1.3930048
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.98
Output dim: 4, lower bound: -1.4096916, upper bound: 1.4022107
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.98
Output dim: 4, lower bound: -1.4202416, upper bound: 1.3767159
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.98
Output dim: 4, lower bound: -1.3963412, upper bound: 1.4008328
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.98
Output dim: 4, lower bound: -1.3942055, upper bound: 1.3788378
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.98
Output dim: 4, lower bound: -1.3704082, upper bound: 1.4028799
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.98
Output dim: 4, lower bound: -1.4072017, upper bound: 1.3882135
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.98
Output dim: 4, lower bound: -1.4089183, upper bound: 1.3864615
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.98
Output dim: 4, lower bound: -1.4019511, upper bound: 1.3942474
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.98
Output dim: 4, lower bound: -1.3760862, upper bound: 1.3942553
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.98
Output dim: 4, lower bound: -1.4079042, upper bound: 1.3901542
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.98
Output dim: 4, lower bound: -1.4079956, upper bound: 1.3900269
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.98
Output dim: 4, lower bound: -1.4022274, upper bound: 1.3829355
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.98
Output dim: 4, lower bound: -1.4022179, upper bound: 1.3948826
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.98
Output dim: 4, lower bound: -1.4093141, upper bound: 1.3869442
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.98
Output dim: 4, lower bound: -1.3833769, upper bound: 1.3869512
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.98
Output dim: 4, lower bound: -1.3842351, upper bound: 1.4121042
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.98
Output dim: 4, lower bound: -1.3843824, upper bound: 1.4120051
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.98
Output dim: 4, lower bound: -1.3735694, upper bound: 1.3953675
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.98
Output dim: 4, lower bound: -1.3735618, upper bound: 1.4212465
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.98
Output dim: 4, lower bound: -1.3836904, upper bound: 1.4188768
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.98
Output dim: 4, lower bound: -1.3854968, upper bound: 1.4188765
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.98
Output dim: 4, lower bound: -1.3862644, upper bound: 1.4193763
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=2.483335494995117
rel_dist={4: [-1.421316393236534, 1.4213159903507133]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 891

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1255035, upper bound: 1.1216147
time: 6.96 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1216144, upper bound: 1.1255041
time: 6.07 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 13.05 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 13.05
Output dim: 4, lower bound: -1.1255035, upper bound: 1.1216147
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 13.05
Output dim: 4, lower bound: -1.1216144, upper bound: 1.1255041

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.6035805, 2.6039109
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.1019106, 2.1044481
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.4141841, 2.4128053
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.0157714, 2.0163708
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.3070197, 2.3068004
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.0443039, 2.0442178
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.3217568, 2.3209419
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7824702, 2.7802300
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2875643, 2.2892056
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.7390914, 1.7398214

Time for backsubstitution: 14.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 5805

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 494

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1245311, upper bound: 1.1206474
time: 8.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1255022, upper bound: 1.1206479
time: 6.70 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.6039104, 2.6035800
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.1044483, 2.1019106
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.4128060, 2.4141836
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.0163708, 2.0157712
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.3068004, 2.3070192
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.0442181, 2.0443039
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.3209419, 2.3217568
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7802300, 2.7824702
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2892056, 2.2875643
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.7398214, 1.7390912

Time for backsubstitution: 14.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 821

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 494

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1206476, upper bound: 1.1255021
time: 6.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1216131, upper bound: 1.1245316
time: 8.63 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 30.16 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 30.16
Output dim: 4, lower bound: -1.1245311, upper bound: 1.1206474
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 30.16
Output dim: 4, lower bound: -1.1255022, upper bound: 1.1206479
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 30.16
Output dim: 4, lower bound: -1.1206476, upper bound: 1.1255021
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 30.16
Output dim: 4, lower bound: -1.1216131, upper bound: 1.1245316

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.5896912, 2.5937645
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.1082306, 2.1125512
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.4221849, 2.4194694
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.0162053, 2.0168815
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.3015089, 2.3022079
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.0286298, 2.0254104
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.2918448, 2.2960148
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7782612, 2.7767205
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2794237, 2.2824187
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.7271514, 1.7315516

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 5735

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1220753, upper bound: 1.1207163
time: 4.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1245244, upper bound: 1.1119974
time: 5.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1149545, upper bound: 1.1216062
time: 8.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.5934334, 2.5900218
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.1100140, 2.1107683
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.4208479, 2.4208064
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.0162816, 2.0168047
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.3024273, 2.3012896
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.0254965, 2.0285439
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.2968292, 2.2910304
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7789602, 2.7760210
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2807779, 2.2810655
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.7308216, 1.7278814

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 5805

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4569

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1254895, upper bound: 1.1154438
time: 4.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1202924, upper bound: 1.1206343
time: 6.23 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.5900211, 2.5934336
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.1107683, 2.1100140
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.4208059, 2.4208474
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.0168047, 2.0162818
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.3012896, 2.3024273
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.0285439, 2.0254965
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.2910304, 2.2968295
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7760210, 2.7789607
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2810659, 2.2807775
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.7278819, 1.7308214

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 4560

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1203134, upper bound: 1.1251156
time: 6.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1203129, upper bound: 1.1242335
time: 6.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.5937653, 2.5896909
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.1125512, 2.1082308
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.4194689, 2.4221845
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.0168815, 2.0162051
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.3022079, 2.3015089
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.0254102, 2.0286300
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.2960143, 2.2918451
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7767210, 2.7782607
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2824183, 2.2794242
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.7315516, 1.7271514

Time for backsubstitution: 14.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1216064, upper bound: 1.1149568
time: 6.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1119979, upper bound: 1.1245269
time: 4.52 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 25.61 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.61
Output dim: 4, lower bound: -1.1245244, upper bound: 1.1119974
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.61
Output dim: 4, lower bound: -1.1149545, upper bound: 1.1216062
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.61
Output dim: 4, lower bound: -1.1254895, upper bound: 1.1154438
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.61
Output dim: 4, lower bound: -1.1202924, upper bound: 1.1206343
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.61
Output dim: 4, lower bound: -1.1203134, upper bound: 1.1251156
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.61
Output dim: 4, lower bound: -1.1203129, upper bound: 1.1242335
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.61
Output dim: 4, lower bound: -1.1216064, upper bound: 1.1149568
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.61
Output dim: 4, lower bound: -1.1119979, upper bound: 1.1245269

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.5889421, 2.5934157
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.1102476, 2.1151578
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.4236488, 2.4213636
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.0176201, 2.0179758
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2963247, 2.2959895
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.0291333, 2.0260615
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.2909904, 2.2949891
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7787256, 2.7770801
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2792664, 2.2822294
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.7199421, 1.7255445

Time for backsubstitution: 14.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 5805

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 523

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1141050, upper bound: 1.1119916
time: 6.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1245183, upper bound: 1.1016037
time: 17.19 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.5893426, 2.5930152
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.1108375, 2.1145682
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.4240789, 2.4209328
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.0172997, 2.0182960
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2952900, 2.2970238
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.0292811, 2.0259137
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.2908192, 2.2951603
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7786207, 2.7771854
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2792349, 2.2822609
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.7211442, 1.7243423

Time for backsubstitution: 18.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 4569

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5735

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1099879, upper bound: 1.1070354
time: 7.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1099627, upper bound: 1.1215968
time: 8.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.5934744, 2.5916042
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.1105711, 2.1112127
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.3819518, 2.3896143
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.0020118, 2.0049028
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2825141, 2.2774115
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.0230265, 2.0271664
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.2952156, 2.2890687
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7577085, 2.7505236
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2479577, 2.2537169
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.7022634, 1.7044189

Time for backsubstitution: 18.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 5735

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1230427, upper bound: 1.1145984
time: 5.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1246137, upper bound: 1.1129879
time: 5.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.5950155, 2.5900631
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.1104581, 2.1113257
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.3896565, 2.3819103
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.0043802, 2.0025346
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2785487, 2.2813768
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.0241189, 2.0260739
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.2948675, 2.2894163
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7534637, 2.7547693
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2534289, 2.2482452
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.7073588, 1.6993234

Time for backsubstitution: 18.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5805

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1202859, upper bound: 1.1180110
time: 7.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1137795, upper bound: 1.1180186
time: 6.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.5895205, 2.5915208
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.1075964, 2.1062288
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.4207611, 2.4208097
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.0186110, 2.0184779
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.3016376, 2.3027139
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.0274134, 2.0241351
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.2921395, 2.2977481
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7715769, 2.7752728
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2831860, 2.2825203
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.7278109, 1.7305272

Time for backsubstitution: 18.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 5805

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 523

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1098983, upper bound: 1.1251089
time: 6.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1203071, upper bound: 1.1146900
time: 5.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.5881090, 2.5900993
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.1069837, 2.1055927
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.4207535, 2.4208021
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.0190010, 2.0180881
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.3015766, 2.3026528
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.0271821, 2.0243647
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.2919488, 2.2975523
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7723312, 2.7745171
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2828083, 2.2821465
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.7275872, 1.7303019

Time for backsubstitution: 19.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4569

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1203008, upper bound: 1.1190233
time: 7.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1151027, upper bound: 1.1242197
time: 7.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.5930152, 2.5893421
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.1145682, 2.1108375
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.4209328, 2.4240789
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.0182962, 2.0172994
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2970238, 2.2952905
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.0259137, 2.0292811
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.2951598, 2.2908196
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7771854, 2.7786202
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2822609, 2.2792349
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.7243423, 1.7211440

Time for backsubstitution: 18.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 523

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4569

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1215936, upper bound: 1.1097622
time: 5.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1163965, upper bound: 1.1149417
time: 5.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.5934157, 2.5889416
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.1151581, 2.1102476
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.4213638, 2.4236481
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.0179758, 2.0176198
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2959890, 2.2963247
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.0260615, 2.0291333
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.2949891, 2.2909906
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7770796, 2.7787261
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2822294, 2.2792664
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.7255445, 1.7199421

Time for backsubstitution: 18.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 5805

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 523

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1016032, upper bound: 1.1245182
time: 8.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1119915, upper bound: 1.1141051
time: 6.75 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 33.98 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 33.98
Output dim: 4, lower bound: -1.1141050, upper bound: 1.1119916
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 33.98
Output dim: 4, lower bound: -1.1245183, upper bound: 1.1016037
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 33.98
Output dim: 4, lower bound: -1.1099879, upper bound: 1.1070354
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 33.98
Output dim: 4, lower bound: -1.1099627, upper bound: 1.1215968
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 33.98
Output dim: 4, lower bound: -1.1230427, upper bound: 1.1145984
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 33.98
Output dim: 4, lower bound: -1.1246137, upper bound: 1.1129879
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 33.98
Output dim: 4, lower bound: -1.1202859, upper bound: 1.1180110
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 33.98
Output dim: 4, lower bound: -1.1137795, upper bound: 1.1180186
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 33.98
Output dim: 4, lower bound: -1.1098983, upper bound: 1.1251089
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 33.98
Output dim: 4, lower bound: -1.1203071, upper bound: 1.1146900
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 33.98
Output dim: 4, lower bound: -1.1203008, upper bound: 1.1190233
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 33.98
Output dim: 4, lower bound: -1.1151027, upper bound: 1.1242197
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 33.98
Output dim: 4, lower bound: -1.1215936, upper bound: 1.1097622
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 33.98
Output dim: 4, lower bound: -1.1163965, upper bound: 1.1149417
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 33.98
Output dim: 4, lower bound: -1.1016032, upper bound: 1.1245182
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 33.98
Output dim: 4, lower bound: -1.1119915, upper bound: 1.1141051

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.5751171, 2.5452242
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.1101246, 2.1151114
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.4180532, 2.4018681
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.0162730, 2.0175886
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2854815, 2.2928739
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.0279756, 2.0257280
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.2710109, 2.2254391
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7778301, 2.7768221
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2769160, 2.2740679
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.7148371, 1.7240725

Time for backsubstitution: 19.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 4569

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4560

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1139541, upper bound: 1.1119912
time: 6.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1141049, upper bound: 1.1118540
time: 6.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.5407505, 2.5795913
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.1102014, 2.1150348
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.4041524, 2.4157681
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.0172324, 2.0166287
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2932091, 2.2851458
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.0287995, 2.0249040
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.2214403, 2.2750101
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7784672, 2.7761846
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2711043, 2.2798796
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.7184701, 1.7204390

Time for backsubstitution: 18.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1220614, upper bound: 1.1007719
time: 4.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1236811, upper bound: 1.0991896
time: 4.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.6114130, 2.6198082
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.1218963, 2.1279883
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.4259796, 2.4232392
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.0280547, 2.0271595
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2748823, 2.2725396
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.0333333, 2.0308313
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.2853403, 2.2885857
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7927861, 2.7888570
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2780437, 2.2808309
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6913400, 1.6995053

Time for backsubstitution: 19.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1075320, upper bound: 1.1060952
time: 7.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1091437, upper bound: 1.1045807
time: 6.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.6161356, 2.6150856
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.1242623, 2.1256266
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.4263859, 2.4228332
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.0261626, 2.0290537
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2708063, 2.2766166
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.0341992, 2.0299656
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.2842450, 2.2896810
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7902913, 2.7913518
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2778053, 2.2810693
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6963067, 1.6945386

Time for backsubstitution: 18.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 4560

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 523

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0995433, upper bound: 1.1215894
time: 6.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1099566, upper bound: 1.1111778
time: 8.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.5934634, 2.5916059
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.1105301, 2.1111784
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.3820405, 2.3896861
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.0020552, 2.0049379
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2825346, 2.2774372
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.0230722, 2.0272207
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.2951727, 2.2890334
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7577348, 2.7505450
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2479887, 2.2537417
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.7022696, 1.7044244

Time for backsubstitution: 18.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 5735

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4560

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1228976, upper bound: 1.1145957
time: 7.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1230426, upper bound: 1.1144493
time: 8.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.5934768, 2.5915933
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.1105368, 2.1111715
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.3820233, 2.3897030
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.0020466, 2.0049465
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2825403, 2.2774320
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.0230808, 2.0272121
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.2951803, 2.2890260
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7577310, 2.7505498
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2479811, 2.2537479
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.7022686, 1.7044253

Time for backsubstitution: 14.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 5735

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1246070, upper bound: 1.1034564
time: 4.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1044008, upper bound: 1.1129813
time: 5.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.6015387, 2.5979810
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.0666242, 2.0752535
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.3705430, 2.3578429
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -1.9925232, 1.9931293
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2753453, 2.2775335
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.0254998, 2.0272114
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.2856827, 2.2784004
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7215567, 2.7159996
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2184119, 2.2190619
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.7164245, 1.7103243

Time for backsubstitution: 14.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 5735

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1178391, upper bound: 1.1171671
time: 5.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1193734, upper bound: 1.1155559
time: 5.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.6029320, 2.5965853
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.0743809, 2.0674915
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.3655887, 2.3627975
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -1.9949751, 1.9906778
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2747045, 2.2781730
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.0252566, 2.0274515
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.2838516, 2.2802305
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7146940, 2.7228632
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2242446, 2.2132287
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.7183604, 1.7083886

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 523

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1137728, upper bound: 1.1084297
time: 4.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1041763, upper bound: 1.1180089
time: 6.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.5756960, 2.5433295
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.1074729, 2.1061819
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.4151649, 2.4013143
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.0172634, 2.0180902
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2907944, 2.2995987
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.0262556, 2.0238013
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.2721605, 2.2281981
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7706823, 2.7750149
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2808361, 2.2743587
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.7227063, 1.7290559

Time for backsubstitution: 14.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 821

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4569

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1098860, upper bound: 1.1198736
time: 4.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1046886, upper bound: 1.1250958
time: 6.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.5413294, 2.5776966
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.1075492, 2.1061053
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.4012661, 2.4152145
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.0182233, 2.0171309
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2985220, 2.2918706
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.0270796, 2.0229774
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.2225895, 2.2777689
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7713194, 2.7743778
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2750244, 2.2801704
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.7263393, 1.7254224

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 4569

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5735

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1202968, upper bound: 1.1017947
time: 6.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1074027, upper bound: 1.1146760
time: 5.38 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 26.40 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.40
Output dim: 4, lower bound: -1.1139541, upper bound: 1.1119912
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.40
Output dim: 4, lower bound: -1.1141049, upper bound: 1.1118540
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.40
Output dim: 4, lower bound: -1.1220614, upper bound: 1.1007719
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.40
Output dim: 4, lower bound: -1.1236811, upper bound: 1.0991896
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.40
Output dim: 4, lower bound: -1.1075320, upper bound: 1.1060952
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.40
Output dim: 4, lower bound: -1.1091437, upper bound: 1.1045807
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.40
Output dim: 4, lower bound: -1.0995433, upper bound: 1.1215894
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.40
Output dim: 4, lower bound: -1.1099566, upper bound: 1.1111778
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.40
Output dim: 4, lower bound: -1.1228976, upper bound: 1.1145957
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.40
Output dim: 4, lower bound: -1.1230426, upper bound: 1.1144493
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.40
Output dim: 4, lower bound: -1.1246070, upper bound: 1.1034564
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.40
Output dim: 4, lower bound: -1.1044008, upper bound: 1.1129813
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.40
Output dim: 4, lower bound: -1.1178391, upper bound: 1.1171671
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.40
Output dim: 4, lower bound: -1.1193734, upper bound: 1.1155559
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.40
Output dim: 4, lower bound: -1.1137728, upper bound: 1.1084297
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.40
Output dim: 4, lower bound: -1.1041763, upper bound: 1.1180089
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.40
Output dim: 4, lower bound: -1.1098860, upper bound: 1.1198736
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.40
Output dim: 4, lower bound: -1.1046886, upper bound: 1.1250958
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.40
Output dim: 4, lower bound: -1.1202968, upper bound: 1.1017947
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.40
Output dim: 4, lower bound: -1.1074027, upper bound: 1.1146760
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.40
Output dim: 4, lower bound: -1.1203008, upper bound: 1.1190233
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.40
Output dim: 4, lower bound: -1.1151027, upper bound: 1.1242197
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.40
Output dim: 4, lower bound: -1.1215936, upper bound: 1.1097622
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.40
Output dim: 4, lower bound: -1.1163965, upper bound: 1.1149417
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.40
Output dim: 4, lower bound: -1.1016032, upper bound: 1.1245182
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.40
Output dim: 4, lower bound: -1.1119915, upper bound: 1.1141051
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=2.308117389678955
rel_dist={4: [-1.1255975120485147, 1.1255947565116422]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 891

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 523

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0013866, upper bound: 1.0090146
time: 6.62 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0090146, upper bound: 1.0013862
time: 10.04 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 16.67 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 16.67
Output dim: 4, lower bound: -1.0013866, upper bound: 1.0090146
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 16.67
Output dim: 4, lower bound: -1.0090146, upper bound: 1.0013862

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.4475460, 2.4217706
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.0265427, 2.0266004
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.3192158, 2.3087907
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -1.9438753, 1.9445949
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2388663, 2.2446632
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -1.9596305, 1.9602485
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.1814804, 2.1443024
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7237387, 2.7242169
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2237968, 2.2194381
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6675339, 1.6702588

Time for backsubstitution: 14.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 891

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5735

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0013789, upper bound: 0.9993679
time: 6.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9917397, upper bound: 1.0090064
time: 13.40 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.4217701, 2.4475455
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.0266004, 2.0265429
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.3087912, 2.3192158
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -1.9445949, 1.9438753
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2446628, 2.2388668
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -1.9602485, 1.9596305
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.1443024, 2.1814804
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7242174, 2.7237387
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2194386, 2.2237968
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6702590, 1.6675339

Time for backsubstitution: 14.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 5805

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0069332, upper bound: 1.0005539
time: 5.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0081802, upper bound: 0.9993136
time: 6.33 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 26.39 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 26.39
Output dim: 4, lower bound: -1.0013789, upper bound: 0.9993679
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 26.39
Output dim: 4, lower bound: -0.9917397, upper bound: 1.0090064
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 26.39
Output dim: 4, lower bound: -1.0069332, upper bound: 1.0005539
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 26.39
Output dim: 4, lower bound: -1.0081802, upper bound: 0.9993136

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.4710550, 2.4417377
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.0393734, 2.0376596
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.3214231, 2.3106933
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -1.9527388, 1.9548771
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2143812, 2.2232351
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -1.9643312, 1.9643002
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.1749067, 2.1385498
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7354088, 2.7377591
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2223663, 2.2181869
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6414557, 1.6404557

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9904622, upper bound: 0.9981067
time: 4.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9904592, upper bound: 1.0090020
time: 6.21 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.4217691, 2.4475355
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.0265646, 2.0265021
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.3088675, 2.3193049
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -1.9446316, 1.9439182
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2446899, 2.2388902
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -1.9603004, 1.9596758
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.1442652, 2.1814375
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7242403, 2.7237654
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2194643, 2.2238274
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6702647, 1.6675406

Time for backsubstitution: 14.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 4569

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0081753, upper bound: 0.9922696
time: 5.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0011360, upper bound: 0.9993117
time: 4.75 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 25.36 seconds
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 25.36
Output dim: 4, lower bound: -0.9904622, upper bound: 0.9981067
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.36
Output dim: 4, lower bound: -0.9904592, upper bound: 1.0090020
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.36
Output dim: 4, lower bound: -1.0081753, upper bound: 0.9922696
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 25.36
Output dim: 4, lower bound: -1.0011360, upper bound: 0.9993117

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.4727087, 2.4430909
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.0418358, 2.0396760
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.3232088, 2.3121562
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -1.9538321, 1.9562130
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2081642, 2.2177935
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -1.9649467, 1.9648042
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.1738815, 2.1376531
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7357683, 2.7381978
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2221775, 2.2180219
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6351480, 1.6332464

Time for backsubstitution: 14.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 5805

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4569

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9904497, upper bound: 1.0051236
time: 4.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9865868, upper bound: 1.0089917
time: 4.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.4210196, 2.4470859
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.0285816, 2.0289614
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.3103313, 2.3210917
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -1.9459648, 1.9450114
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2392454, 2.2326694
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -1.9608035, 1.9602900
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.1433682, 2.1804128
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7246780, 2.7241235
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2192984, 2.2236385
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6630554, 1.6612327

Time for backsubstitution: 14.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5805

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0081706, upper bound: 0.9880509
time: 5.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0039484, upper bound: 0.9922657
time: 5.03 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 25.53 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 25.53
Output dim: 4, lower bound: -0.9904497, upper bound: 1.0051236
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.53
Output dim: 4, lower bound: -0.9865868, upper bound: 1.0089917
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.53
Output dim: 4, lower bound: -1.0081706, upper bound: 0.9880509
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 25.53
Output dim: 4, lower bound: -1.0039484, upper bound: 0.9922657

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.4739065, 2.4431334
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.0422812, 2.0402064
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.2900910, 2.2732601
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -1.9413381, 1.9419425
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.1842847, 2.1968884
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -1.9632955, 1.9623337
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.1719189, 2.1359513
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7102718, 2.7158842
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.1934614, 2.1852012
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6104116, 1.6046884

Time for backsubstitution: 14.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 4560

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9844990, upper bound: 1.0081576
time: 7.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9857294, upper bound: 1.0069111
time: 6.45 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.4275417, 2.4546533
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -1.9820383, 1.9882352
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.2843194, 2.2913647
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -1.9312620, 1.9321470
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2358823, 2.2288265
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -1.9621201, 1.9614265
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.1337247, 2.1693966
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.6886988, 2.6829967
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.1842833, 2.1929984
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6721215, 1.6717510

Time for backsubstitution: 14.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 5735

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0073781, upper bound: 0.9878151
time: 6.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0079337, upper bound: 0.9872576
time: 5.20 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 26.01 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.01
Output dim: 4, lower bound: -0.9844990, upper bound: 1.0081576
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 26.01
Output dim: 4, lower bound: -0.9857294, upper bound: 1.0069111
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.01
Output dim: 4, lower bound: -1.0073781, upper bound: 0.9878151
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.01
Output dim: 4, lower bound: -1.0079337, upper bound: 0.9872576

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.4738960, 2.4431314
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.0422397, 2.0401700
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.2901802, 2.2733364
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -1.9413819, 1.9419799
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.1843061, 2.1969142
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -1.9633403, 1.9623854
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.1718769, 2.1359146
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7102971, 2.7159066
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.1934924, 2.1852274
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6104178, 1.6046939

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 494

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9835163, upper bound: 1.0081556
time: 4.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9844978, upper bound: 1.0071590
time: 6.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.4351873, 2.4612398
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -1.9816682, 1.9874053
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.2843103, 2.2913611
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -1.9330678, 1.9342456
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2362132, 2.2291117
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -1.9609318, 1.9600658
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.1347866, 2.1703155
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.6842537, 2.6791177
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.1863117, 2.1947436
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6733465, 1.6728079

Time for backsubstitution: 15.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 494

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4569

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0073685, upper bound: 0.9839400
time: 4.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0035057, upper bound: 0.9878025
time: 4.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.4341288, 2.4622984
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -1.9812081, 1.9878652
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.2843161, 2.2913551
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -1.9333606, 1.9339533
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2361674, 2.2291579
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -1.9607596, 1.9602385
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.1346431, 2.1704588
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.6848202, 2.6785522
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.1860285, 2.1950264
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6731787, 1.6729755

Time for backsubstitution: 14.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 4560

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 891

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0077616, upper bound: 0.9866531
time: 4.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0031194, upper bound: 0.9866589
time: 6.46 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 25.95 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 25.95
Output dim: 4, lower bound: -0.9835163, upper bound: 1.0081556
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 25.95
Output dim: 4, lower bound: -0.9844978, upper bound: 1.0071590
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 25.95
Output dim: 4, lower bound: -1.0073685, upper bound: 0.9839400
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 25.95
Output dim: 4, lower bound: -1.0035057, upper bound: 0.9878025
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 25.95
Output dim: 4, lower bound: -1.0077616, upper bound: 0.9866531
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 25.95
Output dim: 4, lower bound: -1.0031194, upper bound: 0.9866589

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.4600077, 2.4320507
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.0485592, 2.0478268
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.2978468, 2.2800007
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -1.9418159, 1.9424710
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.1787963, 2.1920929
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -1.9468837, 1.9435785
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.1419644, 2.1097407
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7060885, 2.7122226
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.1853523, 2.1781025
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.5984783, 1.5955071

Time for backsubstitution: 14.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 4560

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5805

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9835116, upper bound: 1.0039291
time: 4.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9792785, upper bound: 1.0081513
time: 4.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.4628153, 2.4292436
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.0498967, 2.0464895
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.2968440, 2.2810035
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -1.9418736, 1.9424136
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.1794848, 2.1914039
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -1.9445338, 1.9459286
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.1457028, 2.1060023
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7066131, 2.7116981
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.1863670, 2.1770873
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6012311, 1.5927544

Time for backsubstitution: 14.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5805

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9844931, upper bound: 1.0029406
time: 5.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9802528, upper bound: 1.0071577
time: 4.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.4352312, 2.4624410
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -1.9821968, 1.9878495
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.2454147, 2.2582433
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -1.9187956, 1.9217496
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2153082, 2.2052326
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -1.9584632, 1.9584167
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.1330843, 2.1683524
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.6619444, 2.6536231
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.1534920, 2.1660275
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6447892, 1.6480725

Time for backsubstitution: 14.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 891

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 494

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0069507, upper bound: 0.9839389
time: 4.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0069809, upper bound: 0.9829669
time: 4.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.4352474, 2.4636669
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -1.9686494, 1.9772136
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.2827244, 2.2887297
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -1.9326143, 1.9336567
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2350149, 2.2278404
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -1.9611268, 1.9605384
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.1303449, 2.1655488
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.6753993, 2.6674523
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.1761889, 2.1864181
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6756415, 1.6759861

Time for backsubstitution: 14.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 4560

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 494

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0067870, upper bound: 0.9862660
time: 4.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0077603, upper bound: 0.9862351
time: 4.71 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 24.13 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 24.13
Output dim: 4, lower bound: -0.9835116, upper bound: 1.0039291
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 24.13
Output dim: 4, lower bound: -0.9792785, upper bound: 1.0081513
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 24.13
Output dim: 4, lower bound: -0.9844931, upper bound: 1.0029406
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 24.13
Output dim: 4, lower bound: -0.9802528, upper bound: 1.0071577
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 24.13
Output dim: 4, lower bound: -1.0069507, upper bound: 0.9839389
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 24.13
Output dim: 4, lower bound: -1.0069809, upper bound: 0.9829669
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 24.13
Output dim: 4, lower bound: -1.0067870, upper bound: 0.9862660
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 24.13
Output dim: 4, lower bound: -1.0077603, upper bound: 0.9862351

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.4675751, 2.4385729
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.0078330, 2.0012832
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.2681203, 2.2539899
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -1.9289508, 1.9277675
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.1749525, 2.1887283
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -1.9480205, 1.9448953
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.1309485, 2.1000972
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.6649618, 2.6762443
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.1547108, 2.1430864
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6089964, 1.6045728

Time for backsubstitution: 14.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4560

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9791550, upper bound: 1.0081511
time: 4.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9792783, upper bound: 1.0080241
time: 4.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.4703827, 2.4357657
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.0091705, 1.9999459
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.2671175, 2.2549927
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -1.9290080, 1.9277101
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.1756411, 2.1880398
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -1.9456701, 1.9472456
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.1346869, 2.0963590
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.6654873, 2.6757193
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.1557255, 2.1420717
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6117487, 1.6018202

Time for backsubstitution: 14.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 4560

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 891

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9796967, upper bound: 1.0023353
time: 4.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9796934, upper bound: 1.0069792
time: 4.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.4156680, 2.4412804
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -1.9733510, 1.9805782
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.2893543, 2.2963619
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -1.9331064, 1.9340911
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2301927, 2.2223301
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -1.9423194, 1.9440818
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.1041713, 2.1356363
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.6717196, 2.6632457
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.1690607, 2.1782751
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6651044, 1.6626959

Time for backsubstitution: 14.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 4560

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5735

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0077526, upper bound: 0.9823533
time: 5.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9968422, upper bound: 0.9823721
time: 4.37 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 24.41 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 24.41
Output dim: 4, lower bound: -0.9791550, upper bound: 1.0081511
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 24.41
Output dim: 4, lower bound: -0.9792783, upper bound: 1.0080241
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 24.41
Output dim: 4, lower bound: -0.9796967, upper bound: 1.0023353
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 24.41
Output dim: 4, lower bound: -0.9796934, upper bound: 1.0069792
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 24.41
Output dim: 4, lower bound: -1.0077526, upper bound: 0.9823533
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 24.41
Output dim: 4, lower bound: -0.9968422, upper bound: 0.9823721

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.4525547, 2.4275224
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.0021775, 1.9935956
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.2637439, 2.2480395
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -1.9272447, 1.9254372
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.1718450, 2.1845150
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -1.9423285, 1.9371505
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.1300063, 2.0994158
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.6617804, 2.6718922
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.1454353, 2.1304717
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.5985355, 1.5968754

Time for backsubstitution: 14.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 891

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9789325, upper bound: 1.0079146
time: 4.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9789314, upper bound: 1.0069726
time: 8.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.4565239, 2.4235520
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.0001457, 1.9956286
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.2621698, 2.2496138
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -1.9266205, 1.9260616
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.1707392, 2.1856227
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -1.9402752, 1.9392035
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.1302676, 2.0991547
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.6606102, 2.6730633
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.1420956, 2.1338105
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6012988, 1.5941124

Time for backsubstitution: 14.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 891

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9790557, upper bound: 1.0077883
time: 4.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9790546, upper bound: 1.0068469
time: 6.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.4377384, 2.4668932
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -1.9844103, 1.9934120
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.2912545, 2.2985673
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -1.9433908, 1.9429550
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2087650, 2.1978450
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -1.9463720, 1.9487839
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.0984182, 2.1290622
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.6852598, 2.6749153
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.1678114, 2.1768460
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6353006, 1.6366177

Time for backsubstitution: 14.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 4569

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4560

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0076254, upper bound: 0.9823523
time: 4.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0077525, upper bound: 0.9822246
time: 4.83 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 24.36 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 24.36
Output dim: 4, lower bound: -0.9789325, upper bound: 1.0079146
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 24.36
Output dim: 4, lower bound: -0.9789314, upper bound: 1.0069726
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 24.36
Output dim: 4, lower bound: -0.9790557, upper bound: 1.0077883
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 24.36
Output dim: 4, lower bound: -0.9790546, upper bound: 1.0068469
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 24.36
Output dim: 4, lower bound: -1.0076254, upper bound: 0.9823523
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 24.36
Output dim: 4, lower bound: -1.0077525, upper bound: 0.9822246

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.4517040, 2.4256134
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -1.9988520, 1.9898105
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.2636991, 2.2480001
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -1.9290490, 1.9275339
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.1721759, 2.1848006
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -1.9411411, 1.9357901
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.1310673, 2.1003339
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.6573405, 2.6680188
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.1474614, 2.1322145
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.5984101, 1.5965822

Time for backsubstitution: 14.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 891

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 891

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9783583, upper bound: 1.0030994
time: 4.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9783526, upper bound: 1.0077455
time: 4.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.4556742, 2.4216430
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -1.9968202, 1.9918435
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.2621250, 2.2495742
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -1.9284248, 1.9281583
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.1710706, 2.1859083
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -1.9390883, 1.9378431
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.1313291, 2.1000729
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.6561694, 2.6691899
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.1441216, 2.1355534
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6011734, 1.5938191

Time for backsubstitution: 15.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 891

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 891

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9784871, upper bound: 1.0029726
time: 4.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9784814, upper bound: 1.0076153
time: 4.20 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.4227161, 2.4558408
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -1.9787550, 1.9857240
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.2868786, 2.2926173
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -1.9416847, 1.9406245
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2056599, 2.1936321
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -1.9406791, 1.9410381
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.0974760, 2.1283810
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.6820812, 2.6705647
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.1585340, 2.1642303
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6248403, 1.6289201

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4569

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4569

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0076159, upper bound: 0.9784807
time: 6.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0037521, upper bound: 0.9823440
time: 4.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.4266863, 2.4518709
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -1.9767222, 1.9877555
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.2853050, 2.2941914
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -1.9410601, 1.9412489
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2045527, 2.1947379
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -1.9386263, 1.9430912
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.0977373, 2.1281197
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.6809101, 2.6717358
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.1551962, 2.1675696
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6276035, 1.6261568

Time for backsubstitution: 15.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4569

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4569

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0077429, upper bound: 0.9783551
time: 4.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0038793, upper bound: 0.9822151
time: 4.58 seconds

## Summary of splitting (split count: 9)
- Time for RS candidates: 24.39 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 24.39
Output dim: 4, lower bound: -0.9783583, upper bound: 1.0030994
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 24.39
Output dim: 4, lower bound: -0.9783526, upper bound: 1.0077455
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 24.39
Output dim: 4, lower bound: -0.9784871, upper bound: 1.0029726
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 24.39
Output dim: 4, lower bound: -0.9784814, upper bound: 1.0076153
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 10, time: 24.39
Output dim: 4, lower bound: -1.0076159, upper bound: 0.9784807
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 24.39
Output dim: 4, lower bound: -1.0037521, upper bound: 0.9823440
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 10, time: 24.39
Output dim: 4, lower bound: -1.0077429, upper bound: 0.9783551
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 24.39
Output dim: 4, lower bound: -1.0038793, upper bound: 0.9822151

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.4530730, 2.4267325
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -1.9882007, 1.9772522
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.2610741, 2.2464092
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -1.9287534, 1.9267888
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.1708589, 2.1836476
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -1.9414434, 1.9361589
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.1261573, 2.0960355
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.6462412, 2.6585994
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.1388526, 2.1223750
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6014204, 1.5990450

Time for backsubstitution: 15.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 1825
type: RSZ, layer: 3, pos: 610
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 2574
type: RSZ, layer: 3, pos: 1857
type: RSZ, layer: 3, pos: 2227
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 3116
type: RSZ, layer: 3, pos: 1990
type: RSZ, layer: 3, pos: 1985
type: RSZ, layer: 3, pos: 323
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 2081
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2532
type: RSZ, layer: 3, pos: 1976
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 905
type: RSZ, layer: 3, pos: 1124
type: RSZ, layer: 3, pos: 2907
type: RSZ, layer: 3, pos: 2559
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 1377
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 1726
type: RSZ, layer: 3, pos: 2577
type: RSZ, layer: 3, pos: 897
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 962
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 599
type: RSZ, layer: 3, pos: 2811
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 704
type: RSZ, layer: 3, pos: 551
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1678

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1516

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9781875, upper bound: 1.0003011
time: 6.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9709100, upper bound: 1.0075803
time: 4.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.4570422, 2.4227619
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -1.9861689, 1.9792850
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.2594995, 2.2479835
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -1.9281292, 1.9274132
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.1697531, 2.1847553
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -1.9393902, 1.9382119
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.1264186, 2.0957744
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.6450701, 2.6597710
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.1355138, 2.1257143
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6041837, 1.5962820

Time for backsubstitution: 15.00 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 1990
type: RSZ, layer: 3, pos: 2907
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 1726
type: RSZ, layer: 3, pos: 1857
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1976
type: RSZ, layer: 3, pos: 1377
type: RSZ, layer: 3, pos: 2577
type: RSZ, layer: 3, pos: 599
type: RSZ, layer: 3, pos: 2559
type: RSZ, layer: 3, pos: 3116
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1678
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 962
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1124
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1825
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 905
type: RSZ, layer: 3, pos: 2227
type: RSZ, layer: 3, pos: 2811
type: RSZ, layer: 3, pos: 551
type: RSZ, layer: 3, pos: 323
type: RSZ, layer: 3, pos: 2532
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 1985
type: RSZ, layer: 3, pos: 610
type: RSZ, layer: 3, pos: 897
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2574
type: RSZ, layer: 3, pos: 2081
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 704
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 2488

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 662

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9441240, upper bound: 0.9930176
time: 4.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9641919, upper bound: 0.9735654
time: 5.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.4227614, 2.4570427
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -1.9792852, 1.9861691
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.2479835, 2.2594995
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -1.9274130, 1.9281292
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.1847553, 2.1697531
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -1.9382119, 1.9393902
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.0957742, 2.1264186
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.6597719, 2.6450706
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.1257138, 2.1355138
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.5962820, 1.6041837

Time for backsubstitution: 14.85 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2081
type: RSZ, layer: 3, pos: 2577
type: RSZ, layer: 3, pos: 2227
type: RSZ, layer: 3, pos: 704
type: RSZ, layer: 3, pos: 2559
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 323
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 897
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 962
type: RSZ, layer: 3, pos: 599
type: RSZ, layer: 3, pos: 1726
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 551
type: RSZ, layer: 3, pos: 1976
type: RSZ, layer: 3, pos: 1857
type: RSZ, layer: 3, pos: 1985
type: RSZ, layer: 3, pos: 2907
type: RSZ, layer: 3, pos: 905
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 1377
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 3116
type: RSZ, layer: 3, pos: 1825
type: RSZ, layer: 3, pos: 1990
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 610
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2811
type: RSZ, layer: 3, pos: 1678
type: RSZ, layer: 3, pos: 2574
type: RSZ, layer: 3, pos: 1124
type: RSZ, layer: 3, pos: 2532
type: RSZ, layer: 3, pos: 431

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2488

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9871015, upper bound: 0.9580847
time: 5.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9871338, upper bound: 0.9580848
time: 4.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.4267325, 2.4530730
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -1.9772520, 1.9882009
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.2464094, 2.2610738
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -1.9267888, 1.9287534
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.1836476, 2.1708589
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -1.9361591, 1.9414432
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.0960355, 2.1261573
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.6585989, 2.6462412
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.1223750, 2.1388531
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.5990453, 1.6014204

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1124
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2341
type: RSZ, layer: 3, pos: 2081
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1726
type: RSZ, layer: 3, pos: 1678
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 962
type: RSZ, layer: 3, pos: 1516
type: RSZ, layer: 3, pos: 2574
type: RSZ, layer: 3, pos: 897
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1990
type: RSZ, layer: 3, pos: 323
type: RSZ, layer: 3, pos: 2811
type: RSZ, layer: 3, pos: 610
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 3116
type: RSZ, layer: 3, pos: 1857
type: RSZ, layer: 3, pos: 1976
type: RSZ, layer: 3, pos: 186
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 431
type: RSZ, layer: 3, pos: 1985
type: RSZ, layer: 3, pos: 2532
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2383
type: RSZ, layer: 3, pos: 2577
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 599
type: RSZ, layer: 3, pos: 704
type: RSZ, layer: 3, pos: 2227
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 551
type: RSZ, layer: 3, pos: 2907
type: RSZ, layer: 3, pos: 2559
type: RSZ, layer: 3, pos: 2867
type: RSZ, layer: 3, pos: 1486
type: RSZ, layer: 3, pos: 1377
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 1825
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 905

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1124

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0053579, upper bound: 0.9780320
time: 4.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0074209, upper bound: 0.9759595
time: 4.78 seconds

## Summary of splitting (split count: 10)
- Time for RS candidates: 24.02 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 11, time: 24.02
Output dim: 4, lower bound: -0.9781875, upper bound: 1.0003011
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 11, time: 24.02
Output dim: 4, lower bound: -0.9709100, upper bound: 1.0075803
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 11, time: 24.02
Output dim: 4, lower bound: -0.9441240, upper bound: 0.9930176
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 11, time: 24.02
Output dim: 4, lower bound: -0.9641919, upper bound: 0.9735654
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 11, time: 24.02
Output dim: 4, lower bound: -0.9871015, upper bound: 0.9580847
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 11, time: 24.02
Output dim: 4, lower bound: -0.9871338, upper bound: 0.9580848
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 11, time: 24.02
Output dim: 4, lower bound: -1.0053579, upper bound: 0.9780320
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 11, time: 24.02
Output dim: 4, lower bound: -1.0074209, upper bound: 0.9759595
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=2.249711513519287
rel_dist={4: [-1.0090186137045807, 1.0090191641306294]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2414.69 seconds
