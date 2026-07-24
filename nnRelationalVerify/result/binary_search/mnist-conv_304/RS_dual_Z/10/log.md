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
execution time: IAR + LP analysis = 14.87 + 33.06 = 47.94 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3552.06 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.4249300956726074
rel_dist={4: [-1.3292853219987384, 1.3292848273935602]}

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
Binary search time: 210.37 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Relational Split (RS_dual_Z) starts
Time budget: 3341.69 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 5735

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4213006, upper bound: 1.3971147
time: 5.48 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3971124, upper bound: 1.4213007
time: 5.49 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 11.22 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 11.22
Output dim: 4, lower bound: -1.4213006, upper bound: 1.3971147
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 11.22
Output dim: 4, lower bound: -1.3971124, upper bound: 1.4213007

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -3.0199232, 3.0281873
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.3997779, 2.4039111
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.7013941, 2.7021053
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.2539639, 2.2506533
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.4659834, 2.4588499
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.2973571, 2.2988720
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.6571956, 2.6552792
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8832743, 2.8832743
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.5058036, 2.5053868
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.8987885, 1.9074802

Time for backsubstitution: 14.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 4560

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3969836, upper bound: 1.3971116
time: 5.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4213004, upper bound: 1.3969816
time: 4.63 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -3.0281878, 3.0199227
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.4039111, 2.3997779
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.7021055, 2.7013946
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.2506533, 2.2539639
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.4588499, 2.4659829
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.2988720, 2.2973571
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.6552796, 2.6571958
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8832743, 2.8832743
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.5053868, 2.5058041
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.9074802, 1.8987887

Time for backsubstitution: 14.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 4560

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3969813, upper bound: 1.4213003
time: 4.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3971117, upper bound: 1.4211990
time: 4.74 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 23.96 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 23.96
Output dim: 4, lower bound: -1.3969836, upper bound: 1.3971116
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 23.96
Output dim: 4, lower bound: -1.4213004, upper bound: 1.3969816
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 23.96
Output dim: 4, lower bound: -1.3969813, upper bound: 1.4213003
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 23.96
Output dim: 4, lower bound: -1.3971117, upper bound: 1.4211990

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -3.0049009, 3.0224283
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.3968334, 2.3962231
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.6991177, 2.6961558
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.2530894, 2.2483220
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.4643526, 2.4546380
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.2944026, 2.2911267
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.6562543, 2.6549468
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8832743, 2.8832743
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.5009809, 2.4927726
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.8883281, 1.9034672

Time for backsubstitution: 14.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 494

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4193926, upper bound: 1.3971096
time: 4.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4211970, upper bound: 1.3952546
time: 8.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -3.0141649, 3.0131652
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.3920898, 2.4009647
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.6954451, 2.6998289
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.2516327, 2.2497787
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.4617710, 2.4572177
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.2896118, 2.2959173
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.6568627, 2.6543369
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8832743, 2.8832743
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.4931893, 2.5005631
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.8947754, 1.8970199

Time for backsubstitution: 14.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 494

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4195448, upper bound: 1.3969795
time: 4.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4212983, upper bound: 1.3951087
time: 5.17 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -3.0131655, 3.0141649
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.4009647, 2.3920898
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.6998291, 2.6954451
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.2497787, 2.2516327
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.4572182, 2.4617710
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.2959170, 2.2896118
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.6543374, 2.6568632
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8832743, 2.8832743
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.5005631, 2.4931893
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.8970199, 1.8947754

Time for backsubstitution: 14.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 494

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3951084, upper bound: 1.4212982
time: 6.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3969790, upper bound: 1.4195449
time: 5.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -3.0224285, 3.0049009
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.3962231, 2.3968332
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.6961555, 2.6991179
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.2483220, 2.2530897
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.4546375, 2.4643531
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.2911267, 2.2944024
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.6549468, 2.6562536
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8832743, 2.8832743
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.4927726, 2.5009809
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.9034672, 1.8883281

Time for backsubstitution: 15.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 494

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3952543, upper bound: 1.4211970
time: 6.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3971094, upper bound: 1.4193922
time: 4.84 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 26.21 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.21
Output dim: 4, lower bound: -1.4193926, upper bound: 1.3971096
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.21
Output dim: 4, lower bound: -1.4211970, upper bound: 1.3952546
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.21
Output dim: 4, lower bound: -1.4195448, upper bound: 1.3969795
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.21
Output dim: 4, lower bound: -1.4212983, upper bound: 1.3951087
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.21
Output dim: 4, lower bound: -1.3951084, upper bound: 1.4212982
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.21
Output dim: 4, lower bound: -1.3969790, upper bound: 1.4195449
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.21
Output dim: 4, lower bound: -1.3952543, upper bound: 1.4211970
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.21
Output dim: 4, lower bound: -1.3971094, upper bound: 1.4193922

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.9910131, 3.0150895
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.4031539, 2.4056637
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.7081223, 2.7028196
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.2535229, 2.2488892
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.4588432, 2.4507351
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.2810774, 2.2723184
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.6263409, 2.6337578
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8832743, 2.8832743
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.4928398, 2.4870000
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.8763876, 1.8979492

Time for backsubstitution: 15.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4193820, upper bound: 1.3953630
time: 5.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3935280, upper bound: 1.3953687
time: 4.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.9975629, 3.0085397
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.4062743, 2.4025435
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.7057819, 2.7051592
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.2536569, 2.2487550
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.4604502, 2.4491282
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.2755942, 2.2778018
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.6350651, 2.6250350
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8832743, 2.8832743
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.4952087, 2.4846320
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.8828101, 1.8915267

Time for backsubstitution: 14.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4211865, upper bound: 1.3936146
time: 5.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3952924, upper bound: 1.3936214
time: 6.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -3.0002770, 3.0058265
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.3984103, 2.4104056
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.7044487, 2.7064927
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.2520657, 2.2503459
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.4562607, 2.4533153
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.2762866, 2.2771089
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.6269512, 2.6331477
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8832743, 2.8832743
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.4850492, 2.4947910
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.8828349, 1.8915019

Time for backsubstitution: 14.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4195322, upper bound: 1.3952554
time: 4.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3936598, upper bound: 1.3952669
time: 4.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -3.0068269, 2.9992766
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.4015307, 2.4072855
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.7021084, 2.7088325
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.2522001, 2.2502117
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.4578686, 2.4517083
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.2708035, 2.2825923
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.6356735, 2.6244252
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8832743, 2.8832743
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.4874172, 2.4924231
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.8892574, 1.8850791

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4212858, upper bound: 1.3934910
time: 5.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3954065, upper bound: 1.3935024
time: 7.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.9992776, 3.0068264
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.4072852, 2.4015307
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.7088327, 2.7021089
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.2502117, 2.2522001
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.4517078, 2.4578681
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.2825923, 2.2708035
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.6244249, 2.6356740
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8832743, 2.8832743
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.4924231, 2.4874172
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.8850789, 1.8892574

Time for backsubstitution: 15.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3935024, upper bound: 1.3954062
time: 6.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3934908, upper bound: 1.4212864
time: 4.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -3.0058274, 3.0002766
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.4104056, 2.3984106
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.7064934, 2.7044487
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.2503462, 2.2520657
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.4533157, 2.4562612
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.2771087, 2.2762868
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.6331482, 2.6269515
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8832743, 2.8832743
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.4947910, 2.4850492
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.8915019, 1.8828347

Time for backsubstitution: 14.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3952666, upper bound: 1.3936603
time: 4.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3952551, upper bound: 1.4195325
time: 4.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -3.0085397, 2.9975622
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.4025435, 2.4062741
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.7051601, 2.7057817
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.2487545, 2.2536569
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.4491282, 2.4604506
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.2778020, 2.2755940
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.6250343, 2.6350646
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8832743, 2.8832743
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.4846325, 2.4952083
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.8915267, 1.8828101

Time for backsubstitution: 14.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3936218, upper bound: 1.3952947
time: 5.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3936141, upper bound: 1.4211888
time: 5.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -3.0150895, 2.9910123
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.4056640, 2.4031539
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.7028198, 2.7081215
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.2488890, 2.2535226
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.4507351, 2.4588437
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.2723184, 2.2810774
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.6337576, 2.6263418
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8832743, 2.8832743
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.4869995, 2.4928403
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.8979492, 1.8763874

Time for backsubstitution: 14.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3953684, upper bound: 1.3935278
time: 4.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3953607, upper bound: 1.4193816
time: 8.91 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 28.59 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.59
Output dim: 4, lower bound: -1.4193820, upper bound: 1.3953630
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.59
Output dim: 4, lower bound: -1.3935280, upper bound: 1.3953687
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.59
Output dim: 4, lower bound: -1.4211865, upper bound: 1.3936146
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.59
Output dim: 4, lower bound: -1.3952924, upper bound: 1.3936214
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.59
Output dim: 4, lower bound: -1.4195322, upper bound: 1.3952554
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.59
Output dim: 4, lower bound: -1.3936598, upper bound: 1.3952669
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.59
Output dim: 4, lower bound: -1.4212858, upper bound: 1.3934910
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.59
Output dim: 4, lower bound: -1.3954065, upper bound: 1.3935024
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.59
Output dim: 4, lower bound: -1.3935024, upper bound: 1.3954062
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.59
Output dim: 4, lower bound: -1.3934908, upper bound: 1.4212864
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.59
Output dim: 4, lower bound: -1.3952666, upper bound: 1.3936603
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.59
Output dim: 4, lower bound: -1.3952551, upper bound: 1.4195325
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.59
Output dim: 4, lower bound: -1.3936218, upper bound: 1.3952947
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.59
Output dim: 4, lower bound: -1.3936141, upper bound: 1.4211888
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.59
Output dim: 4, lower bound: -1.3953684, upper bound: 1.3935278
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.59
Output dim: 4, lower bound: -1.3953607, upper bound: 1.4193816

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.9923668, 3.0171454
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.4051700, 2.4087200
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.7095847, 2.7050371
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.2551818, 2.2499838
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.4544382, 2.4445157
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.2815809, 2.2730811
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.6256146, 2.6327314
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8832743, 2.8832743
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.4927063, 2.4868116
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.8691783, 1.8928435

Time for backsubstitution: 14.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 5805

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4193656, upper bound: 1.3844708
time: 4.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4085062, upper bound: 1.3953443
time: 5.28 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.9930687, 3.0164442
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.4061999, 2.4076798
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.7103391, 2.7042832
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.2546172, 2.2505443
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.4526243, 2.4463258
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.2818398, 2.2728221
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.6253152, 2.6330302
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8832743, 2.8832743
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.4926510, 2.4868665
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.8712816, 1.8907402

Time for backsubstitution: 14.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 5805

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3935078, upper bound: 1.3844805
time: 6.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3826596, upper bound: 1.3953518
time: 5.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.9989166, 3.0105958
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.4082899, 2.4055998
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.7072453, 2.7073770
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.2553158, 2.2498493
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.4560442, 2.4429088
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.2760978, 2.2785645
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.6343369, 2.6240087
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8832743, 2.8832743
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.4950752, 2.4844437
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.8756013, 1.8864207

Time for backsubstitution: 14.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 5805

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4211696, upper bound: 1.3827359
time: 5.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4102835, upper bound: 1.3935968
time: 4.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.9996185, 3.0098946
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.4093204, 2.4045596
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.7079997, 2.7066231
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.2547517, 2.2504101
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.4542303, 2.4447188
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.2763567, 2.2783055
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.6340375, 2.6243074
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8832743, 2.8832743
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.4950199, 2.4844980
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.8777041, 1.8843174

Time for backsubstitution: 14.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 5805

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3952755, upper bound: 1.3827441
time: 4.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3844024, upper bound: 1.3936069
time: 5.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -3.0016308, 3.0078824
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.4004264, 2.4134607
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.7059121, 2.7087102
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.2537246, 2.2514405
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.4518538, 2.4470959
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.2767906, 2.2778716
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.6262231, 2.6321216
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8832743, 2.8832743
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.4849157, 2.4946022
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.8756256, 1.8863962

Time for backsubstitution: 14.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 5805

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4195153, upper bound: 1.3843674
time: 4.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4086482, upper bound: 1.3952404
time: 5.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -3.0023327, 3.0071812
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.4014587, 2.4124217
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.7066665, 2.7079563
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.2531605, 2.2520013
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.4500418, 2.4489088
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.2770495, 2.2776124
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.6259255, 2.6324205
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8832743, 2.8832743
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.4848604, 2.4946575
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.8777289, 1.8842926

Time for backsubstitution: 14.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 5805

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3936427, upper bound: 1.3843772
time: 4.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3827816, upper bound: 1.3952520
time: 5.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -3.0081806, 3.0013328
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.4035468, 2.4103403
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.7035728, 2.7110500
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.2538590, 2.2513063
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.4534607, 2.4454889
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.2713070, 2.2833550
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.6349463, 2.6233988
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8832743, 2.8832743
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.4872837, 2.4922342
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.8820481, 1.8799734

Time for backsubstitution: 14.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 5805

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4212689, upper bound: 1.3826228
time: 4.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3826618, upper bound: 1.3934728
time: 5.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -3.0088825, 3.0006316
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.4045787, 2.4093013
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.7043262, 2.7102962
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.2532945, 2.2518673
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.4516487, 2.4473014
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.2715659, 2.2830958
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.6346478, 2.6236978
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8832743, 2.8832743
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.4872284, 2.4922895
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.8841519, 1.8778701

Time for backsubstitution: 14.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 5805

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3953896, upper bound: 1.3826343
time: 4.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3845162, upper bound: 1.3934824
time: 4.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -3.0006313, 3.0088823
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.4093013, 2.4045787
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.7102962, 2.7043264
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.2518673, 2.2532945
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.4473009, 2.4516487
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.2830958, 2.2715662
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.6236978, 2.6346476
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8832743, 2.8832743
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.4922895, 2.4872284
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.8778701, 1.8841517

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 5805

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3934822, upper bound: 1.3845165
time: 4.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3826340, upper bound: 1.3953893
time: 6.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -3.0013332, 3.0081811
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.4103403, 2.4035468
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.7110496, 2.7035723
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.2513065, 2.2538590
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.4454889, 2.4534607
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.2833552, 2.2713070
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.6233983, 2.6349463
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8832743, 2.8832743
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.4922342, 2.4872837
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.8799734, 1.8820484

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 5805

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3934706, upper bound: 1.4103838
time: 4.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3826225, upper bound: 1.4212712
time: 5.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -3.0071812, 3.0023324
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.4124217, 2.4014587
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.7079558, 2.7066662
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.2520013, 2.2531602
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.4489088, 2.4500418
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.2776122, 2.2770495
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.6324201, 2.6259251
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8832743, 2.8832743
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.4946575, 2.4848604
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.8842931, 1.8777289

Time for backsubstitution: 14.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 5805

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3952497, upper bound: 1.3827816
time: 6.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3843766, upper bound: 1.3936426
time: 6.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -3.0078821, 3.0016313
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.4134607, 2.4004266
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.7087102, 2.7059121
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.2514405, 2.2537246
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.4470959, 2.4518538
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.2778716, 2.2767906
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.6321206, 2.6262238
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8832743, 2.8832743
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.4946022, 2.4849157
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.8863959, 1.8756256

Time for backsubstitution: 14.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 5805

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3952382, upper bound: 1.4086482
time: 4.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3843652, upper bound: 1.4195150
time: 5.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -3.0098944, 2.9996181
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.4045596, 2.4093204
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.7066226, 2.7079992
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.2504101, 2.2547517
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.4447184, 2.4542308
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.2783055, 2.2763567
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.6243072, 2.6340382
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8832743, 2.8832743
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.4844980, 2.4950199
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.8843174, 1.8777041

Time for backsubstitution: 14.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 5805

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3936047, upper bound: 1.3844046
time: 6.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3827436, upper bound: 1.3952749
time: 6.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -3.0105953, 2.9989171
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.4056001, 2.4082899
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.7073770, 2.7072453
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.2498493, 2.2553160
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.4429083, 2.4560447
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.2785645, 2.2760978
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.6240087, 2.6343372
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8832743, 2.8832743
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.4844437, 2.4950747
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.8864207, 1.8756011

Time for backsubstitution: 14.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 5805

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3935970, upper bound: 1.4102857
time: 4.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3827359, upper bound: 1.4211691
time: 6.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -3.0164442, 2.9930682
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.4076800, 2.4062002
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.7042832, 2.7103391
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.2505445, 2.2546172
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.4463263, 2.4526238
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.2728219, 2.2818398
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.6330304, 2.6253154
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8832743, 2.8832743
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.4868660, 2.4926515
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.8907399, 1.8712816

Time for backsubstitution: 14.82 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=2.483335494995117
rel_dist={4: [-1.421316393236534, 1.4213159903507133]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 5735

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1255847, upper bound: 1.1126055
time: 5.42 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1126027, upper bound: 1.1255847
time: 5.38 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 11.05 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 11.05
Output dim: 4, lower bound: -1.1255847, upper bound: 1.1126055
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 11.05
Output dim: 4, lower bound: -1.1126027, upper bound: 1.1255847

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.6224270, 2.6271496
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.1282387, 2.1306005
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.4229898, 2.4233961
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.0301204, 2.0282285
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2877083, 2.2836318
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.0479693, 2.0488353
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.3203740, 2.3192787
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8078508, 2.8053565
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2962127, 2.2959743
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.7068243, 1.7117908

Time for backsubstitution: 14.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 4560

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1254421, upper bound: 1.1126024
time: 6.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1255845, upper bound: 1.1124689
time: 8.81 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.6271496, 2.6224270
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.1306005, 2.1282387
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.4233961, 2.4229901
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.0282283, 2.0301204
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2836313, 2.2877078
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.0488353, 2.0479696
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.3192782, 2.3203740
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8053560, 2.8078513
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2959743, 2.2962127
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.7117910, 1.7068243

Time for backsubstitution: 14.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 4560

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1124688, upper bound: 1.1255870
time: 5.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1126023, upper bound: 1.1254446
time: 5.97 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 26.65 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 26.65
Output dim: 4, lower bound: -1.1254421, upper bound: 1.1126024
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 26.65
Output dim: 4, lower bound: -1.1255845, upper bound: 1.1124689
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 26.65
Output dim: 4, lower bound: -1.1124688, upper bound: 1.1255870
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 26.65
Output dim: 4, lower bound: -1.1126023, upper bound: 1.1254446

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.6074057, 2.6174207
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.1232615, 2.1229124
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.4191399, 2.4174466
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.0286217, 2.0258973
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2849712, 2.2794199
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.0429616, 2.0410900
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.3194318, 2.3186851
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8050623, 2.8010058
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2880502, 2.2833595
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6963634, 1.7050147

Time for backsubstitution: 14.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 494

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1244547, upper bound: 1.1126037
time: 5.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1254410, upper bound: 1.1116770
time: 7.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.6126986, 2.6121278
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.1205506, 2.1256220
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.4170408, 2.4195457
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.0277891, 2.0267296
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2834959, 2.2808943
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.0402246, 2.0438275
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.3197803, 2.3183365
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8035011, 2.8025670
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2835975, 2.2878118
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.7000480, 1.7013302

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 494

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1115282, upper bound: 1.1124704
time: 6.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1255833, upper bound: 1.1115254
time: 6.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.6121273, 2.6126990
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.1256218, 2.1205506
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.4195461, 2.4170406
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.0267296, 2.0277891
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2808943, 2.2834959
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.0438275, 2.0402243
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.3183365, 2.3197799
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8025675, 2.8035007
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2878118, 2.2835979
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.7013302, 1.7000477

Time for backsubstitution: 14.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 494

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1115282, upper bound: 1.1255829
time: 6.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1115282, upper bound: 1.1246086
time: 5.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.6174212, 2.6074052
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.1229124, 2.1232612
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.4174461, 2.4191394
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.0258975, 2.0286217
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2794199, 2.2849717
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.0410900, 2.0429618
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.3186851, 2.3194318
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8010063, 2.8050628
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2833591, 2.2880502
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.7050147, 1.6963637

Time for backsubstitution: 14.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 494

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1116770, upper bound: 1.1254402
time: 5.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1126011, upper bound: 1.1244544
time: 7.16 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 27.65 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.65
Output dim: 4, lower bound: -1.1244547, upper bound: 1.1126037
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.65
Output dim: 4, lower bound: -1.1254410, upper bound: 1.1116770
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.65
Output dim: 4, lower bound: -1.1115282, upper bound: 1.1124704
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.65
Output dim: 4, lower bound: -1.1255833, upper bound: 1.1115254
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.65
Output dim: 4, lower bound: -1.1115282, upper bound: 1.1255829
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.65
Output dim: 4, lower bound: -1.1115282, upper bound: 1.1246086
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.65
Output dim: 4, lower bound: -1.1116770, upper bound: 1.1254402
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.65
Output dim: 4, lower bound: -1.1126011, upper bound: 1.1244544

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.5935159, 2.6072750
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.1295819, 2.1310163
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.4271402, 2.4241104
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.0290546, 2.0264070
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2794619, 2.2748284
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.0272870, 2.0222816
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.2895198, 2.2937574
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8008528, 2.7974973
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2799091, 2.2765727
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6844230, 1.6967440

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1244491, upper bound: 1.1109920
time: 5.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1099080, upper bound: 1.1109960
time: 6.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.5972600, 2.6035323
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.1313648, 2.1292331
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.4258032, 2.4254475
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.0291314, 2.0263302
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2803802, 2.2739100
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.0241532, 2.0254149
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.2945042, 2.2887731
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8015528, 2.7967973
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2812634, 2.2752194
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6880932, 1.6930737

Time for backsubstitution: 14.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1254354, upper bound: 1.1100404
time: 7.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1108745, upper bound: 1.1100437
time: 5.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.5988107, 2.6019816
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.1268716, 2.1337256
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.4250412, 2.4262094
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.0282221, 2.0272393
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2779865, 2.2763028
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.0245495, 2.0250192
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.2898679, 2.2934089
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7992926, 2.7990580
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2754574, 2.2810245
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6881070, 1.6930597

Time for backsubstitution: 14.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1245993, upper bound: 1.1108567
time: 8.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1100659, upper bound: 1.1108605
time: 8.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.6025529, 2.5982389
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.1286545, 2.1319427
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.4237041, 2.4275465
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.0282989, 2.0271626
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2789040, 2.2753844
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.0214162, 2.0281525
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.2948523, 2.2884247
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7999926, 2.7983584
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2768106, 2.2796712
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6917772, 1.6893897

Time for backsubstitution: 14.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1255765, upper bound: 1.1098877
time: 10.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1110147, upper bound: 1.1098943
time: 8.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.5982385, 2.6025531
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.1319427, 2.1286545
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.4275475, 2.4237044
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.0271626, 2.0282989
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2753849, 2.2789044
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.0281525, 2.0214159
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.2884245, 2.2948527
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7983580, 2.7999921
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2796707, 2.2768111
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6893897, 1.6917772

Time for backsubstitution: 14.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1098937, upper bound: 1.1110145
time: 4.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1098905, upper bound: 1.1255767
time: 7.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.6019816, 2.5988104
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.1337256, 2.1268713
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.4262094, 2.4250414
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.0272393, 2.0282221
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2763023, 2.2779860
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.0250192, 2.0245495
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.2934089, 2.2898681
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7990580, 2.7992921
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2810249, 2.2754579
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6930594, 1.6881070

Time for backsubstitution: 14.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1108601, upper bound: 1.1100653
time: 6.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1108542, upper bound: 1.1245988
time: 6.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.6035323, 2.5972593
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.1292329, 2.1313648
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.4254475, 2.4258032
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.0263300, 2.0291314
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2739105, 2.2803802
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.0254149, 2.0241535
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.2887731, 2.2945044
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7967978, 2.8015537
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2752190, 2.2812629
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6930737, 1.6880929

Time for backsubstitution: 14.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1100446, upper bound: 1.1108742
time: 7.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1100406, upper bound: 1.1254364
time: 5.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.6072755, 2.5935166
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.1310163, 2.1295819
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.4241104, 2.4271402
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.0264068, 2.0290546
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2748280, 2.2794619
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.0222816, 2.0272868
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.2937574, 2.2895198
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7974977, 2.8008537
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2765722, 2.2799096
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6967440, 1.6844230

Time for backsubstitution: 14.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1109934, upper bound: 1.1099104
time: 5.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1109894, upper bound: 1.1244488
time: 6.05 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 26.23 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.23
Output dim: 4, lower bound: -1.1244491, upper bound: 1.1109920
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.23
Output dim: 4, lower bound: -1.1099080, upper bound: 1.1109960
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.23
Output dim: 4, lower bound: -1.1254354, upper bound: 1.1100404
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.23
Output dim: 4, lower bound: -1.1108745, upper bound: 1.1100437
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.23
Output dim: 4, lower bound: -1.1245993, upper bound: 1.1108567
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.23
Output dim: 4, lower bound: -1.1100659, upper bound: 1.1108605
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.23
Output dim: 4, lower bound: -1.1255765, upper bound: 1.1098877
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.23
Output dim: 4, lower bound: -1.1110147, upper bound: 1.1098943
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.23
Output dim: 4, lower bound: -1.1098937, upper bound: 1.1110145
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.23
Output dim: 4, lower bound: -1.1098905, upper bound: 1.1255767
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.23
Output dim: 4, lower bound: -1.1108601, upper bound: 1.1100653
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.23
Output dim: 4, lower bound: -1.1108542, upper bound: 1.1245988
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.23
Output dim: 4, lower bound: -1.1100446, upper bound: 1.1108742
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.23
Output dim: 4, lower bound: -1.1100406, upper bound: 1.1254364
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.23
Output dim: 4, lower bound: -1.1109934, upper bound: 1.1099104
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.23
Output dim: 4, lower bound: -1.1109894, upper bound: 1.1244488

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.5948715, 2.6090302
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.1315980, 2.1336265
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.4286036, 2.4260049
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.0304718, 2.0275016
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2742786, 2.2686090
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.0277905, 2.0229335
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.2886643, 2.2927313
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8013191, 2.7978563
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2797527, 2.2763834
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6772141, 1.6907368

Time for backsubstitution: 15.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 5805

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1244426, upper bound: 1.1050911
time: 5.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1185886, upper bound: 1.1109831
time: 5.45 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.5952721, 2.6086297
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.1321864, 2.1330321
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.4290347, 2.4255743
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.0301495, 2.0278220
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2732420, 2.2696433
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.0279384, 2.0227852
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.2884932, 2.2929020
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8012133, 2.7979622
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2797213, 2.2764149
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6784158, 1.6895349

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 5805

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1099015, upper bound: 1.1050935
time: 9.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1040284, upper bound: 1.1109865
time: 5.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.5986147, 2.6052876
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.1333809, 2.1318436
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.4272666, 2.4273419
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.0305486, 2.0274248
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2751970, 2.2676907
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.0246568, 2.0260668
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.2936487, 2.2877469
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8020191, 2.7971563
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2811050, 2.2750306
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6808839, 1.6870666

Time for backsubstitution: 14.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 5805

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1254290, upper bound: 1.1041629
time: 6.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1195512, upper bound: 1.1100340
time: 7.49 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.5990152, 2.6048870
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.1339698, 2.1312492
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.4276977, 2.4269114
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.0302258, 2.0277452
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2741604, 2.2687254
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.0248051, 2.0259187
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.2934780, 2.2879174
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8019123, 2.7972622
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2810745, 2.2750621
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6820855, 1.6858647

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 5805

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1108680, upper bound: 1.1041669
time: 5.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1049824, upper bound: 1.1100380
time: 8.27 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.6001654, 2.6037374
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.1288877, 2.1363354
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.4265056, 2.4281039
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.0296392, 2.0283341
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2728024, 2.2700834
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.0250530, 2.0256708
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.2890124, 2.2923827
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7997570, 2.7994175
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2753010, 2.2808356
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6808982, 1.6870525

Time for backsubstitution: 14.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 5805

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1245928, upper bound: 1.1049622
time: 9.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1187372, upper bound: 1.1108474
time: 4.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.6005659, 2.6033366
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.1294770, 2.1357417
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.4269366, 2.4276731
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.0293169, 2.0286546
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2717667, 2.2711191
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.0252013, 2.0255227
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.2888417, 2.2925534
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7996511, 2.7995234
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2752686, 2.2808671
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6820998, 1.6858506

Time for backsubstitution: 14.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 5805

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1100594, upper bound: 1.1049675
time: 4.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1041856, upper bound: 1.1108538
time: 6.20 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.6039076, 2.5999944
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.1306701, 2.1345525
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.4251685, 2.4294410
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.0297160, 2.0282574
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2737207, 2.2691650
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.0219197, 2.0288041
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.2939968, 2.2873983
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8004570, 2.7987175
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2766533, 2.2794828
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6845679, 1.6833825

Time for backsubstitution: 14.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 5805

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1255701, upper bound: 1.1040087
time: 6.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1196825, upper bound: 1.1098838
time: 7.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.6043081, 2.5995939
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.1312604, 2.1339588
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.4255996, 2.4290104
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.0293932, 2.0285778
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2726851, 2.2702007
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.0220675, 2.0286560
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.2938261, 2.2875690
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.8003511, 2.7988234
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2766228, 2.2795138
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6857700, 1.6821804

Time for backsubstitution: 14.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 5805

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1110082, upper bound: 1.1040140
time: 7.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1051140, upper bound: 1.1098881
time: 5.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.5995941, 2.6043086
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.1339588, 2.1312602
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.4290099, 2.4255989
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.0285778, 2.0293934
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2702007, 2.2726851
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.0286560, 2.0220675
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.2875690, 2.2938261
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7988234, 2.8003511
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2795143, 2.2766223
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6821804, 1.6857700

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 5805

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1098872, upper bound: 1.1051146
time: 6.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1040140, upper bound: 1.1110077
time: 5.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.5999947, 2.6039081
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.1345525, 2.1306703
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.4294410, 2.4251680
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.0282574, 2.0297160
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2691650, 2.2737203
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.0288043, 2.0219197
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.2873983, 2.2939968
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7987175, 2.8004570
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2794828, 2.2766533
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6833825, 1.6845679

Time for backsubstitution: 14.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 5805

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1098825, upper bound: 1.1196827
time: 6.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1040081, upper bound: 1.1255691
time: 5.12 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.6033363, 2.6005659
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.1357417, 2.1294770
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.4276729, 2.4269361
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.0286546, 2.0293167
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2711191, 2.2717667
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.0255227, 2.0252011
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.2925534, 2.2888417
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7995234, 2.7996511
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2808666, 2.2752690
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6858506, 1.6820998

Time for backsubstitution: 14.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 5805

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1108536, upper bound: 1.1041849
time: 5.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1049680, upper bound: 1.1100596
time: 5.22 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.6037378, 2.6001654
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.1363354, 2.1288874
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.4281039, 2.4265051
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.0283341, 2.0296392
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2700834, 2.2728024
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.0256705, 2.0250530
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.2923822, 2.2890124
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7994175, 2.7997575
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2808361, 2.2753005
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6870527, 1.6808980

Time for backsubstitution: 14.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 5805

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1108477, upper bound: 1.1187367
time: 4.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1049621, upper bound: 1.1245946
time: 5.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.6048870, 2.5990148
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.1312490, 2.1339695
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.4269118, 2.4276977
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -2.0277452, 2.0302260
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2687254, 2.2741604
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -2.0259185, 2.0248051
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.2879171, 2.2934780
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7972622, 2.8019128
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2750626, 2.2810740
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6858649, 1.6820858

Time for backsubstitution: 15.10 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=2.308117389678955
rel_dist={4: [-1.1255975120485147, 1.1255947565116422]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5735
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 5735

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0090109, upper bound: 0.9993719
time: 5.52 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9993721, upper bound: 1.0090103
time: 4.73 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 10.49 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 10.49
Output dim: 4, lower bound: -1.0090109, upper bound: 0.9993719
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 10.49
Output dim: 4, lower bound: -0.9993721, upper bound: 1.0090103

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.4899282, 2.4934702
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.0377254, 2.0394969
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.3301888, 2.3304932
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -1.9555058, 1.9540868
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2282829, 2.2252259
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -1.9648404, 1.9654896
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.2080998, 2.2072783
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7381763, 2.7363052
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2263484, 2.2261701
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6428361, 1.6465611

Time for backsubstitution: 14.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 4560

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0088823, upper bound: 0.9993735
time: 4.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0090107, upper bound: 0.9992430
time: 6.31 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.4934702, 2.4899282
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.0394969, 2.0377257
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.3304930, 2.3301888
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -1.9540868, 1.9555058
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2252264, 2.2282829
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -1.9654899, 1.9648404
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.2072783, 2.2080998
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7363052, 2.7381763
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2261691, 2.2263484
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6465611, 1.6428361

Time for backsubstitution: 14.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4560
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 4560

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9992438, upper bound: 1.0090100
time: 5.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9993717, upper bound: 1.0088823
time: 6.35 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 26.76 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 26.76
Output dim: 4, lower bound: -1.0088823, upper bound: 0.9993735
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 26.76
Output dim: 4, lower bound: -1.0090107, upper bound: 0.9992430
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 26.76
Output dim: 4, lower bound: -0.9992438, upper bound: 1.0090100
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 26.76
Output dim: 4, lower bound: -0.9993717, upper bound: 1.0088823

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.4749060, 2.4824181
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.0320702, 2.0318089
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.3258133, 2.3245437
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -1.9537988, 1.9517555
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2251778, 2.2210140
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -1.9591484, 1.9577444
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.2071576, 2.2065973
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7349968, 2.7319551
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2170730, 2.2135553
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6323757, 1.6388636

Time for backsubstitution: 14.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 494

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0078849, upper bound: 0.9993730
time: 6.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0088812, upper bound: 0.9983760
time: 4.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.4788771, 2.4784484
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.0300379, 2.0338411
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.3242397, 2.3261180
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -1.9531746, 1.9523802
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2240705, 2.2221198
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -1.9570951, 1.9597974
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.2074189, 2.2063360
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7338257, 2.7331252
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2137341, 2.2168941
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6351385, 1.6361005

Time for backsubstitution: 14.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 494

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0080130, upper bound: 0.9992422
time: 7.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0090096, upper bound: 0.9982481
time: 5.46 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.4784479, 2.4788766
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.0338411, 2.0300376
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.3261185, 2.3242393
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -1.9523802, 1.9531746
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2221193, 2.2240710
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -1.9597974, 1.9570951
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.2063360, 2.2074189
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7331257, 2.7338262
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2168937, 2.2137341
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6361008, 1.6351385

Time for backsubstitution: 14.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 494

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9982456, upper bound: 1.0090090
time: 4.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9992426, upper bound: 1.0080121
time: 5.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.4824181, 2.4749064
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.0318089, 2.0320704
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.3245440, 2.3258133
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -1.9517555, 1.9537990
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2210140, 2.2251773
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -1.9577441, 1.9591484
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.2065978, 2.2071576
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7319546, 2.7349973
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2135558, 2.2170730
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6388636, 1.6323755

Time for backsubstitution: 15.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 494
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 494

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9983736, upper bound: 1.0088806
time: 5.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9993705, upper bound: 1.0078875
time: 4.45 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 24.97 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.97
Output dim: 4, lower bound: -1.0078849, upper bound: 0.9993730
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.97
Output dim: 4, lower bound: -1.0088812, upper bound: 0.9983760
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.97
Output dim: 4, lower bound: -1.0080130, upper bound: 0.9992422
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.97
Output dim: 4, lower bound: -1.0090096, upper bound: 0.9982481
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.97
Output dim: 4, lower bound: -0.9982456, upper bound: 1.0090090
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.97
Output dim: 4, lower bound: -0.9992426, upper bound: 1.0080121
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.97
Output dim: 4, lower bound: -0.9983736, upper bound: 1.0088806
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.97
Output dim: 4, lower bound: -0.9993705, upper bound: 1.0078875

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.4610181, 2.4713368
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.0383911, 2.0394669
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.3334799, 2.3312075
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -1.9542317, 1.9522462
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2196674, 2.2161927
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -1.9426899, 1.9389360
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.1772456, 2.1804242
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7307882, 2.7282710
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2089329, 2.2064300
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6204348, 1.6296754

Time for backsubstitution: 14.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0078807, upper bound: 0.9980903
time: 4.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9969796, upper bound: 0.9980924
time: 7.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.4638257, 2.4685297
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.0397286, 2.0381298
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.3324776, 2.3322103
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -1.9542894, 1.9521885
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2203560, 2.2155042
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -1.9403400, 1.9412861
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.1809840, 2.1766858
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7313128, 2.7277465
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2099476, 2.2054152
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6231875, 1.6269231

Time for backsubstitution: 14.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0088770, upper bound: 0.9970958
time: 6.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9979758, upper bound: 0.9970988
time: 5.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.4649882, 2.4673667
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.0363584, 2.0414991
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.3319054, 2.3327818
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -1.9536076, 1.9528706
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2185612, 2.2172990
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -1.9406371, 1.9409890
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.1775069, 2.1801629
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7296171, 2.7294416
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2055931, 2.2097688
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6231980, 1.6269124

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0080080, upper bound: 0.9979607
time: 6.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9971121, upper bound: 0.9979651
time: 7.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.4677949, 2.4645600
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.0376954, 2.0401618
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.3309031, 2.3337846
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -1.9536648, 1.9528131
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2192497, 2.2166100
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -1.9382868, 1.9433391
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.1812453, 2.1764245
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7301416, 2.7289166
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2066078, 2.2087541
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6259503, 1.6241598

Time for backsubstitution: 14.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0090046, upper bound: 0.9969641
time: 5.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9981091, upper bound: 0.9969714
time: 4.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.4645600, 2.4677954
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.0401616, 2.0376956
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.3337851, 2.3309031
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -1.9528131, 1.9536653
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2166100, 2.2192502
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -1.9433389, 1.9382868
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.1764245, 2.1812453
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7289171, 2.7301421
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2087536, 2.2066088
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6241598, 1.6259503

Time for backsubstitution: 14.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9969689, upper bound: 0.9981079
time: 4.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9969645, upper bound: 1.0090037
time: 4.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.4673676, 2.4649882
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.0414991, 2.0363584
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.3327818, 2.3319058
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -1.9528704, 1.9536076
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2172985, 2.2185612
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -1.9409890, 1.9406369
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.1801624, 2.1775069
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7294416, 2.7296171
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2097683, 2.2055936
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6269126, 1.6231980

Time for backsubstitution: 14.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9979652, upper bound: 0.9971119
time: 7.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9979608, upper bound: 1.0080079
time: 6.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.4685292, 2.4638247
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.0381298, 2.0397284
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.3322105, 2.3324771
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -1.9521885, 1.9542894
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2155037, 2.2203565
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -1.9412861, 1.9403400
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.1766858, 2.1809840
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7277460, 2.7313132
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2054148, 2.2099476
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6269231, 1.6231873

Time for backsubstitution: 14.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9970963, upper bound: 0.9979747
time: 5.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9970933, upper bound: 1.0088761
time: 5.45 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.4713368, 2.4610181
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.0394669, 2.0383911
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.3312073, 2.3334799
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -1.9522462, 1.9542320
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2161922, 2.2196679
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -1.9389358, 1.9426899
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.1804237, 2.1772456
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7282705, 2.7307887
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2064304, 2.2089329
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6296754, 1.6204348

Time for backsubstitution: 15.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 821

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9980933, upper bound: 0.9969788
time: 5.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9980903, upper bound: 1.0078798
time: 5.67 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 26.95 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.95
Output dim: 4, lower bound: -1.0078807, upper bound: 0.9980903
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 26.95
Output dim: 4, lower bound: -0.9969796, upper bound: 0.9980924
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.95
Output dim: 4, lower bound: -1.0088770, upper bound: 0.9970958
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 26.95
Output dim: 4, lower bound: -0.9979758, upper bound: 0.9970988
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.95
Output dim: 4, lower bound: -1.0080080, upper bound: 0.9979607
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 26.95
Output dim: 4, lower bound: -0.9971121, upper bound: 0.9979651
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.95
Output dim: 4, lower bound: -1.0090046, upper bound: 0.9969641
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 26.95
Output dim: 4, lower bound: -0.9981091, upper bound: 0.9969714
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 26.95
Output dim: 4, lower bound: -0.9969689, upper bound: 0.9981079
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.95
Output dim: 4, lower bound: -0.9969645, upper bound: 1.0090037
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 26.95
Output dim: 4, lower bound: -0.9979652, upper bound: 0.9971119
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.95
Output dim: 4, lower bound: -0.9979608, upper bound: 1.0080079
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 26.95
Output dim: 4, lower bound: -0.9970963, upper bound: 0.9979747
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.95
Output dim: 4, lower bound: -0.9970933, upper bound: 1.0088761
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 26.95
Output dim: 4, lower bound: -0.9980933, upper bound: 0.9969788
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.95
Output dim: 4, lower bound: -0.9980903, upper bound: 1.0078798

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.4623728, 2.4729919
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.0404072, 2.0419288
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.3349433, 2.3329945
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -1.9555683, 1.9533408
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2142258, 2.2099733
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -1.9431934, 1.9395509
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.1763477, 2.1793976
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7312269, 2.7286301
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2087679, 2.2062411
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6132255, 1.6233678

Time for backsubstitution: 15.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 5805

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0078761, upper bound: 0.9938712
time: 5.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0036628, upper bound: 0.9980852
time: 6.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.4651804, 2.4701853
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.0417442, 2.0405912
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.3339400, 2.3339972
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -1.9556260, 1.9532833
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2149143, 2.2092848
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -1.9408436, 1.9419007
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.1800861, 2.1756597
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7317524, 2.7281055
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2097826, 2.2052264
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6159782, 1.6206152

Time for backsubstitution: 15.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 5805

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0088723, upper bound: 0.9928742
time: 5.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0046590, upper bound: 0.9970880
time: 6.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.4663429, 2.4690223
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.0383744, 2.0439603
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.3333697, 2.3345685
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -1.9549441, 1.9539652
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2131186, 2.2110791
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -1.9411407, 1.9416037
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.1766086, 2.1791363
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7300558, 2.7298007
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2054291, 2.2095804
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6159887, 1.6206045

Time for backsubstitution: 14.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 5805

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0080033, upper bound: 0.9937427
time: 5.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0037899, upper bound: 0.9979556
time: 4.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.4691496, 2.4662151
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.0397115, 2.0426230
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.3323665, 2.3355713
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -1.9550018, 1.9539075
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2138071, 2.2103906
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -1.9387903, 1.9439538
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.1803470, 2.1753983
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7305813, 2.7292757
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2064438, 2.2085652
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6187415, 1.6178522

Time for backsubstitution: 14.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 5805

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0089999, upper bound: 0.9927455
time: 5.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0047866, upper bound: 0.9969623
time: 6.12 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.4662151, 2.4691501
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.0426230, 2.0397115
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.3355708, 2.3323665
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -1.9539075, 1.9550016
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2103910, 2.2138071
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -1.9439540, 1.9387903
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.1753979, 2.1803470
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7292757, 2.7305808
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2085648, 2.2064438
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6178522, 1.6187413

Time for backsubstitution: 14.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 5805

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9969598, upper bound: 1.0047860
time: 5.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9927464, upper bound: 1.0090025
time: 4.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.4690218, 2.4663429
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.0439601, 2.0383744
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.3345685, 2.3333693
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -1.9539652, 1.9549441
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2110796, 2.2131181
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -1.9416037, 1.9411407
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.1791363, 2.1766086
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7298012, 2.7300558
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2095804, 2.2054286
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6206050, 1.6159887

Time for backsubstitution: 15.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 5805

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9927489, upper bound: 1.0037897
time: 9.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9937428, upper bound: 1.0080058
time: 4.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.4701853, 2.4651799
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.0405912, 2.0417442
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.3339972, 2.3339407
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -1.9532833, 1.9556260
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2092848, 2.2149143
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -1.9419007, 1.9408436
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.1756592, 2.1800857
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7281055, 2.7317519
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2052269, 2.2097826
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6206155, 1.6159782

Time for backsubstitution: 15.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 5805

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9970887, upper bound: 1.0046590
time: 4.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9928752, upper bound: 1.0088717
time: 4.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.4729919, 2.4623728
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.0419288, 2.0404072
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.3329940, 2.3349435
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -1.9533405, 1.9555683
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2099733, 2.2142258
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -1.9395509, 1.9431934
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.1793976, 2.1763477
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.7286301, 2.7312269
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.2062416, 2.2087679
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6233678, 1.6132257

Time for backsubstitution: 14.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 5805

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9980856, upper bound: 1.0036619
time: 4.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9938722, upper bound: 1.0078751
time: 4.79 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 24.90 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.90
Output dim: 4, lower bound: -1.0078761, upper bound: 0.9938712
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.90
Output dim: 4, lower bound: -1.0036628, upper bound: 0.9980852
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.90
Output dim: 4, lower bound: -1.0088723, upper bound: 0.9928742
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.90
Output dim: 4, lower bound: -1.0046590, upper bound: 0.9970880
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.90
Output dim: 4, lower bound: -1.0080033, upper bound: 0.9937427
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.90
Output dim: 4, lower bound: -1.0037899, upper bound: 0.9979556
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.90
Output dim: 4, lower bound: -1.0089999, upper bound: 0.9927455
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.90
Output dim: 4, lower bound: -1.0047866, upper bound: 0.9969623
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.90
Output dim: 4, lower bound: -0.9969598, upper bound: 1.0047860
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.90
Output dim: 4, lower bound: -0.9927464, upper bound: 1.0090025
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.90
Output dim: 4, lower bound: -0.9927489, upper bound: 1.0037897
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.90
Output dim: 4, lower bound: -0.9937428, upper bound: 1.0080058
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.90
Output dim: 4, lower bound: -0.9970887, upper bound: 1.0046590
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.90
Output dim: 4, lower bound: -0.9928752, upper bound: 1.0088717
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.90
Output dim: 4, lower bound: -0.9980856, upper bound: 1.0036619
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.90
Output dim: 4, lower bound: -0.9938722, upper bound: 1.0078751

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.4688950, 2.4805598
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -1.9938631, 2.0012019
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.3089323, 2.3032670
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -1.9408660, 1.9404771
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2108617, 2.2061286
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -1.9445109, 1.9406877
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.1667047, 2.1683822
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.6952477, 2.6875029
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.1737514, 2.1755996
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6222935, 1.6338875

Time for backsubstitution: 14.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 523

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0002449, upper bound: 0.9938680
time: 4.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0078720, upper bound: 0.9862390
time: 4.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.4717026, 2.4777527
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -1.9952002, 1.9998646
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.3079300, 2.3042698
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -1.9409237, 1.9404197
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2115502, 2.2054400
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -1.9421606, 1.9430377
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.1704431, 2.1646438
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.6957731, 2.6869783
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.1747661, 2.1745849
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6250458, 1.6311350

Time for backsubstitution: 14.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 523

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0012418, upper bound: 0.9928736
time: 4.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0088683, upper bound: 0.9852422
time: 8.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.4728661, 2.4765902
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -1.9918299, 2.0032334
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.3073578, 2.3048415
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -1.9402418, 1.9411016
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2097535, 2.2072349
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -1.9424577, 1.9427409
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.1669655, 2.1681209
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.6940765, 2.6886735
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.1704125, 2.1789389
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6250563, 1.6311243

Time for backsubstitution: 14.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 523

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0003709, upper bound: 0.9937384
time: 8.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0079993, upper bound: 0.9861109
time: 6.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.4756727, 2.4737830
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -1.9931674, 2.0018964
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.3063555, 2.3058438
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -1.9402990, 1.9410439
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2104421, 2.2065458
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -1.9401078, 1.9450908
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.1707039, 2.1643825
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.6946011, 2.6881485
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.1714272, 2.1779237
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6278090, 1.6283717

Time for backsubstitution: 14.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 523

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0013679, upper bound: 0.9927418
time: 5.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0089959, upper bound: 0.9851165
time: 5.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.4737835, 2.4756727
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.0018964, 1.9931674
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.3058443, 2.3063555
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -1.9410439, 1.9402993
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2065454, 2.2104421
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -1.9450908, 1.9401076
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.1643825, 2.1707039
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.6881485, 2.6946015
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.1779237, 2.1714272
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6283717, 1.6278090

Time for backsubstitution: 15.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 523

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9851165, upper bound: 1.0089956
time: 5.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9927423, upper bound: 1.0013704
time: 6.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.2722282, -11.0764151, -14.2722282, -11.0764151, -2.4765902, 2.4728656
1: -10.6166239, -7.9022126, -10.6166239, -7.9022126, -2.0032334, 1.9918301
2: -10.1443129, -7.3213749, -10.1443129, -7.3213749, -2.3048420, 2.3073583
3: -12.7821178, -10.3563156, -12.7821178, -10.3563156, -1.9411016, 1.9402418
4: 5.8858538, 8.4309330, 5.8858538, 8.4309330, -2.2072349, 2.2097535
5: -8.3676176, -5.7517138, -8.3676176, -5.7517138, -1.9427409, 1.9424577
6: -12.7108383, -9.7072067, -12.7108383, -9.7072067, -2.1681209, 2.1669655
7: -6.2174892, -3.3342149, -6.2174892, -3.3342149, -2.6886730, 2.6940765
8: -3.0022964, -0.2282643, -3.0022964, -0.2282643, -2.1789393, 2.1704125
9: -5.4689426, -3.2161665, -5.4689426, -3.2161665, -1.6311245, 1.6250563

Time for backsubstitution: 14.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4569
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 523

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.9861110, upper bound: 1.0080018
time: 5.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.9851165, upper bound: 1.0003734
time: 5.63 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 26.37 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 26.37
Output dim: 4, lower bound: -1.0002449, upper bound: 0.9938680
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.37
Output dim: 4, lower bound: -1.0078720, upper bound: 0.9862390
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 26.37
Output dim: 4, lower bound: -1.0012418, upper bound: 0.9928736
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.37
Output dim: 4, lower bound: -1.0088683, upper bound: 0.9852422
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 26.37
Output dim: 4, lower bound: -1.0003709, upper bound: 0.9937384
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.37
Output dim: 4, lower bound: -1.0079993, upper bound: 0.9861109
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 26.37
Output dim: 4, lower bound: -1.0013679, upper bound: 0.9927418
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.37
Output dim: 4, lower bound: -1.0089959, upper bound: 0.9851165
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 26.37
Output dim: 4, lower bound: -0.9851165, upper bound: 1.0089956
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 26.37
Output dim: 4, lower bound: -0.9927423, upper bound: 1.0013704
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 26.37
Output dim: 4, lower bound: -0.9861110, upper bound: 1.0080018
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 26.37
Output dim: 4, lower bound: -0.9851165, upper bound: 1.0003734
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.37
Output dim: 4, lower bound: -0.9928752, upper bound: 1.0088717
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.37
Output dim: 4, lower bound: -0.9938722, upper bound: 1.0078751
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=2.249711513519287
rel_dist={4: [-1.0090186137045807, 1.0090191641306294]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2418.41 seconds
