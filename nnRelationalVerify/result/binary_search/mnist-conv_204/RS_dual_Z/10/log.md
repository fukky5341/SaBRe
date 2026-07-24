## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 0.67154946715
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-8.8134632, -6.7279902, -8.8134632, -6.7279902, -2.0854731, 2.0854731)
1: (1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.6906104, 1.6906104)
2: (-5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.5822163, 1.5822163)
3: (-10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.5740218, 1.5740218)
4: (-4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704)
5: (-8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.5837355, 1.5837355)
6: (-5.9832397, -3.9410968, -5.9832397, -3.9410968, -2.0421429, 2.0421429)
7: (-4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261)
8: (-3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.4418497, 1.4418497)
9: (-11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.9131136, 1.9131136)

## BASE Result
execution time: IAR + LP analysis = 15.28 + 31.78 = 47.06 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3552.94 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=1.4624950885772705
rel_dist={1: [-0.9108364634088741, 0.9108356417165169]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=1.3226265907287598
rel_dist={1: [-0.6734581938937723, 0.6734593550965497]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=1.2293808460235596
rel_dist={1: [-0.49714962017070174, 0.49714935310829933]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=1.2760038375854492
rel_dist={1: [-0.5884148639986186, 0.5884165441500335]}

## Binary Search Result
Binary search time: 196.62 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Relational Split (RS_dual_Z) starts
Time budget: 3356.32 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 442
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 442

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9849745, upper bound: 0.9686155
time: 3.86 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9686168, upper bound: 0.9849750
time: 3.65 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 7.70 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 7.70
Output dim: 1, lower bound: -0.9849745, upper bound: 0.9686155
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 7.70
Output dim: 1, lower bound: -0.9686168, upper bound: 0.9849750

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.9564543, 1.9421129
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5278622, 1.5338829
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.4234738, 1.4245944
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2763938, 1.2654262
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.3849473, 1.3871295
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.8531642, 1.8593891
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.3152841, 1.3158528
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6660786, 1.6643673

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 4612

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9816427, upper bound: 0.9686044
time: 3.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9849634, upper bound: 0.9652889
time: 3.66 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.9421129, 1.9564538
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5338827, 1.5278621
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.4245944, 1.4234736
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2654260, 1.2763937
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.3871295, 1.3849473
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.8593888, 1.8531640
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.3158525, 1.3152844
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6643677, 1.6660782

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 4612

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9652903, upper bound: 0.9849638
time: 3.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9686042, upper bound: 0.9816428
time: 3.63 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 22.30 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 22.30
Output dim: 1, lower bound: -0.9816427, upper bound: 0.9686044
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 22.30
Output dim: 1, lower bound: -0.9849634, upper bound: 0.9652889
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 22.30
Output dim: 1, lower bound: -0.9652903, upper bound: 0.9849638
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 22.30
Output dim: 1, lower bound: -0.9686042, upper bound: 0.9816428

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.9567618, 1.9414277
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5266412, 1.5344046
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.4219000, 1.4252770
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2774597, 1.2629244
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.3842573, 1.3874058
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.8542590, 1.8568056
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.3141638, 1.3163449
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6666784, 1.6629434

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 849

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9816040, upper bound: 0.9676216
time: 3.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9806601, upper bound: 0.9685656
time: 4.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.9557681, 1.9421129
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5278622, 1.5326618
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.4234738, 1.4230206
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2738918, 1.2654262
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.3849473, 1.3864394
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.8505807, 1.8593891
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.3152841, 1.3147324
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6646547, 1.6643673

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 849

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9849247, upper bound: 0.9643063
time: 3.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9839806, upper bound: 0.9652502
time: 3.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.9424205, 1.9557683
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5326618, 1.5283836
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.4230206, 1.4241562
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2664922, 1.2738918
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.3864393, 1.3852236
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.8604846, 1.8505805
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.3147322, 1.3157765
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6649675, 1.6646543

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 849

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9652502, upper bound: 0.9839807
time: 3.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9643076, upper bound: 0.9849252
time: 3.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.9414277, 1.9564538
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5338827, 1.5266412
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.4245944, 1.4218998
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2629243, 1.2763937
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.3871295, 1.3842573
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.8568053, 1.8531640
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.3158525, 1.3141640
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6629438, 1.6660782

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 849

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9685655, upper bound: 0.9806601
time: 3.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9676215, upper bound: 0.9816039
time: 4.04 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 22.67 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.67
Output dim: 1, lower bound: -0.9816040, upper bound: 0.9676216
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.67
Output dim: 1, lower bound: -0.9806601, upper bound: 0.9685656
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.67
Output dim: 1, lower bound: -0.9849247, upper bound: 0.9643063
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.67
Output dim: 1, lower bound: -0.9839806, upper bound: 0.9652502
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.67
Output dim: 1, lower bound: -0.9652502, upper bound: 0.9839807
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.67
Output dim: 1, lower bound: -0.9643076, upper bound: 0.9849252
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.67
Output dim: 1, lower bound: -0.9685655, upper bound: 0.9806601
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.67
Output dim: 1, lower bound: -0.9676215, upper bound: 0.9816039

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.9365625, 1.9129229
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5320532, 1.5385307
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.4209521, 1.4246058
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2756760, 1.2606210
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.3752091, 1.3810008
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.8288114, 1.8387926
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.3143076, 1.3165334
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6691465, 1.6661811

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6141

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9686913, upper bound: 0.9676021
time: 4.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9815819, upper bound: 0.9547041
time: 4.21 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.9282570, 1.9212284
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5307674, 1.5398167
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.4212286, 1.4243293
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2751560, 1.2611408
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.3778522, 1.3783575
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.8362463, 1.8313577
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.3143524, 1.3164886
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6699157, 1.6654118

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6141

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9677429, upper bound: 0.9685437
time: 4.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9642879, upper bound: 0.9556520
time: 4.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.9355698, 1.9136083
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5332737, 1.5367883
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.4225254, 1.4223495
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2721081, 1.2631229
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.3758993, 1.3800344
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.8251326, 1.8413761
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.3154281, 1.3149210
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6671228, 1.6676049

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6141

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9720117, upper bound: 0.9642867
time: 3.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9849028, upper bound: 0.9513892
time: 3.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.9272642, 1.9219139
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5319881, 1.5380739
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.4228020, 1.4220729
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2715883, 1.2636427
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.3785427, 1.3773912
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.8325675, 1.8339412
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.3154730, 1.3148762
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6678920, 1.6668358

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6141

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9710633, upper bound: 0.9652282
time: 3.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9839612, upper bound: 0.9523376
time: 3.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.9222212, 1.9272637
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5380740, 1.5325102
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.4220726, 1.4234853
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2647085, 1.2715883
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.3773911, 1.3788185
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.8350365, 1.8325675
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.3148760, 1.3159651
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6674356, 1.6678920

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6141

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9523376, upper bound: 0.9839610
time: 3.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9652281, upper bound: 0.9710631
time: 3.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.9139156, 1.9355693
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5367880, 1.5337960
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.4223492, 1.4232087
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2641888, 1.2721081
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.3800342, 1.3761754
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.8424714, 1.8251324
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.3149208, 1.3159202
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6682048, 1.6671227

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6141

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9513904, upper bound: 0.9849032
time: 4.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9642865, upper bound: 0.9720114
time: 3.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.9212284, 1.9279492
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5392947, 1.5307676
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.4236462, 1.4212289
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2611406, 1.2740903
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.3780816, 1.3778522
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.8313577, 1.8351510
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.3159965, 1.3143526
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6654119, 1.6693158

Time for backsubstitution: 14.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6141

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9556525, upper bound: 0.9806403
time: 3.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9685436, upper bound: 0.9677429
time: 3.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.9129229, 1.9362547
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5380087, 1.5320534
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.4239228, 1.4209523
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2606208, 1.2746102
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.3807247, 1.3752091
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.8387926, 1.8277161
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.3160414, 1.3143078
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6661811, 1.6685467

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6141

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9547042, upper bound: 0.9815819
time: 3.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9676020, upper bound: 0.9686914
time: 3.74 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 22.21 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.21
Output dim: 1, lower bound: -0.9686913, upper bound: 0.9676021
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.21
Output dim: 1, lower bound: -0.9815819, upper bound: 0.9547041
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.21
Output dim: 1, lower bound: -0.9677429, upper bound: 0.9685437
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.21
Output dim: 1, lower bound: -0.9642879, upper bound: 0.9556520
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.21
Output dim: 1, lower bound: -0.9720117, upper bound: 0.9642867
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.21
Output dim: 1, lower bound: -0.9849028, upper bound: 0.9513892
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.21
Output dim: 1, lower bound: -0.9710633, upper bound: 0.9652282
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.21
Output dim: 1, lower bound: -0.9839612, upper bound: 0.9523376
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.21
Output dim: 1, lower bound: -0.9523376, upper bound: 0.9839610
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.21
Output dim: 1, lower bound: -0.9652281, upper bound: 0.9710631
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.21
Output dim: 1, lower bound: -0.9513904, upper bound: 0.9849032
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.21
Output dim: 1, lower bound: -0.9642865, upper bound: 0.9720114
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.21
Output dim: 1, lower bound: -0.9556525, upper bound: 0.9806403
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.21
Output dim: 1, lower bound: -0.9685436, upper bound: 0.9677429
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.21
Output dim: 1, lower bound: -0.9547042, upper bound: 0.9815819
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.21
Output dim: 1, lower bound: -0.9676020, upper bound: 0.9686914

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.8794928, 1.8323441
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5303240, 1.5373062
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3712103, 1.3893902
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2555349, 1.2463398
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.3150938, 1.2961217
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.8224907, 1.8221133
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.2991639, 1.2951766
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6331787, 1.6154616

Time for backsubstitution: 14.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5830

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9650998, upper bound: 0.9675965
time: 3.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9686855, upper bound: 0.9640298
time: 4.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.8559837, 1.8558371
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5308287, 1.5368015
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3857367, 1.3748636
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2613938, 1.2404799
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.2903302, 1.3208810
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.8121319, 1.8324654
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.2929507, 1.3013898
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6184273, 1.6302140

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5830

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9779898, upper bound: 0.9546986
time: 3.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9815761, upper bound: 0.9511324
time: 4.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.8711710, 1.8406496
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5290384, 1.5385920
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3714869, 1.3891137
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2550151, 1.2468591
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.3177326, 1.2934786
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.8299189, 1.8146784
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.2992090, 1.2951318
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6339483, 1.6146923

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5830

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9641515, upper bound: 0.9685379
time: 4.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9477990, upper bound: 0.9649711
time: 4.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.8476782, 1.8641591
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5295429, 1.5380875
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3860133, 1.3745873
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2608740, 1.2409997
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.2929733, 1.3182422
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.8195667, 1.8250370
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.2929956, 1.3013450
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6191959, 1.6294445

Time for backsubstitution: 14.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5830

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9770483, upper bound: 0.9556468
time: 4.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9806345, upper bound: 0.9520809
time: 3.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.8785000, 1.8330295
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5315447, 1.5355636
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3727841, 1.3871338
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2519667, 1.2488415
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.3157842, 1.2951554
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.8188114, 1.8246968
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.3002843, 1.2935641
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6311550, 1.6168852

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5830

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9684344, upper bound: 0.9642809
time: 3.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9720059, upper bound: 0.9606945
time: 3.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.8549905, 1.8565226
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5320492, 1.5350589
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3873105, 1.3726075
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2578256, 1.2429817
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.2910204, 1.3199146
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.8084531, 1.8350489
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.2940711, 1.2997774
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6164036, 1.6316376

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5830

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9813247, upper bound: 0.9513834
time: 3.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9848969, upper bound: 0.9477977
time: 3.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.8701782, 1.8413348
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5302587, 1.5368494
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3730605, 1.3868573
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2514470, 1.2493608
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.3184230, 1.2925123
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.8262405, 1.8172617
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.3003291, 1.2935193
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6319246, 1.6161159

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5830

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9674860, upper bound: 0.9652225
time: 3.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9710575, upper bound: 0.9616360
time: 3.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.8466849, 1.8648443
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5307634, 1.5363446
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3875871, 1.3723309
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2573063, 1.2435014
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.2936637, 1.3172758
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.8158884, 1.8276203
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.2941159, 1.2997326
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6171722, 1.6308681

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5830

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9803832, upper bound: 0.9523318
time: 3.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9839553, upper bound: 0.9487461
time: 3.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.8651519, 1.8466849
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5363450, 1.5312854
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3723309, 1.3882694
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2445676, 1.2573063
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.3172758, 1.2939396
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.8287153, 1.8158882
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.2997323, 1.2946082
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6314678, 1.6171725

Time for backsubstitution: 14.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5830

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9477990, upper bound: 0.9839548
time: 4.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9477990, upper bound: 0.9803831
time: 4.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.8416424, 1.8701782
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5368495, 1.5307809
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3868573, 1.3737431
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2504270, 1.2514472
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.2925122, 1.3186989
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.8183570, 1.8262403
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.2935191, 1.3008214
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6167164, 1.6319249

Time for backsubstitution: 14.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5830

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9606958, upper bound: 0.9710575
time: 4.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9652223, upper bound: 0.9674858
time: 4.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.8568301, 1.8549905
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5350590, 1.5325712
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3726075, 1.3879931
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2440479, 1.2578259
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.3199146, 1.2912965
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.8361440, 1.8084531
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.2997772, 1.2945634
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6322374, 1.6164032

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5830

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9477990, upper bound: 0.9848974
time: 3.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9513833, upper bound: 0.9813246
time: 3.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.8333368, 1.8784997
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5355639, 1.5320667
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3871338, 1.3734665
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2499077, 1.2519672
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.2951553, 1.3160601
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.8257923, 1.8188117
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.2935640, 1.3007766
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6174850, 1.6311554

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5830

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9606945, upper bound: 0.9720073
time: 3.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9642807, upper bound: 0.9684342
time: 4.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.8641591, 1.8473701
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5375652, 1.5295428
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3739047, 1.3860133
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2409995, 1.2598083
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.3179662, 1.2929733
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.8250370, 1.8184717
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.3008527, 1.2929957
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6294441, 1.6185961

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5830

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9477990, upper bound: 0.9806340
time: 5.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9556467, upper bound: 0.9770483
time: 3.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.8406496, 1.8708634
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5380700, 1.5290381
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3884313, 1.3714867
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2468588, 1.2539492
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.2932029, 1.3177326
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.8146782, 1.8288238
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.2946395, 1.2992090
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6146927, 1.6333485

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5830

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9649712, upper bound: 0.9677368
time: 3.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9685377, upper bound: 0.9641515
time: 3.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.8558373, 1.8556757
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5362797, 1.5308286
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3741813, 1.3857367
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2404797, 1.2603276
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.3206050, 1.2903301
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.8324652, 1.8110366
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.3008975, 1.2929509
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6302137, 1.6178268

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5830

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9477990, upper bound: 0.9815757
time: 4.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9546984, upper bound: 0.9779898
time: 3.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.8323441, 1.8791852
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5367842, 1.5303241
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3887076, 1.3712101
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2463396, 1.2544689
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.2958460, 1.3150938
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.8221130, 1.8213952
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.2946843, 1.2991642
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6154613, 1.6325790

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5830

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9640297, upper bound: 0.9686853
time: 3.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9675961, upper bound: 0.9650998
time: 3.69 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 22.45 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.45
Output dim: 1, lower bound: -0.9650998, upper bound: 0.9675965
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.45
Output dim: 1, lower bound: -0.9686855, upper bound: 0.9640298
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.45
Output dim: 1, lower bound: -0.9779898, upper bound: 0.9546986
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.45
Output dim: 1, lower bound: -0.9815761, upper bound: 0.9511324
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.45
Output dim: 1, lower bound: -0.9641515, upper bound: 0.9685379
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.45
Output dim: 1, lower bound: -0.9477990, upper bound: 0.9649711
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.45
Output dim: 1, lower bound: -0.9770483, upper bound: 0.9556468
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.45
Output dim: 1, lower bound: -0.9806345, upper bound: 0.9520809
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.45
Output dim: 1, lower bound: -0.9684344, upper bound: 0.9642809
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.45
Output dim: 1, lower bound: -0.9720059, upper bound: 0.9606945
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.45
Output dim: 1, lower bound: -0.9813247, upper bound: 0.9513834
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.45
Output dim: 1, lower bound: -0.9848969, upper bound: 0.9477977
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.45
Output dim: 1, lower bound: -0.9674860, upper bound: 0.9652225
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.45
Output dim: 1, lower bound: -0.9710575, upper bound: 0.9616360
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.45
Output dim: 1, lower bound: -0.9803832, upper bound: 0.9523318
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.45
Output dim: 1, lower bound: -0.9839553, upper bound: 0.9487461
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.45
Output dim: 1, lower bound: -0.9477990, upper bound: 0.9839548
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.45
Output dim: 1, lower bound: -0.9477990, upper bound: 0.9803831
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.45
Output dim: 1, lower bound: -0.9606958, upper bound: 0.9710575
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.45
Output dim: 1, lower bound: -0.9652223, upper bound: 0.9674858
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.45
Output dim: 1, lower bound: -0.9477990, upper bound: 0.9848974
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.45
Output dim: 1, lower bound: -0.9513833, upper bound: 0.9813246
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.45
Output dim: 1, lower bound: -0.9606945, upper bound: 0.9720073
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.45
Output dim: 1, lower bound: -0.9642807, upper bound: 0.9684342
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.45
Output dim: 1, lower bound: -0.9477990, upper bound: 0.9806340
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.45
Output dim: 1, lower bound: -0.9556467, upper bound: 0.9770483
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.45
Output dim: 1, lower bound: -0.9649712, upper bound: 0.9677368
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.45
Output dim: 1, lower bound: -0.9685377, upper bound: 0.9641515
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.45
Output dim: 1, lower bound: -0.9477990, upper bound: 0.9815757
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.45
Output dim: 1, lower bound: -0.9546984, upper bound: 0.9779898
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.45
Output dim: 1, lower bound: -0.9640297, upper bound: 0.9686853
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.45
Output dim: 1, lower bound: -0.9675961, upper bound: 0.9650998

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.8782039, 1.8287048
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5266750, 1.5360171
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3608499, 1.3857405
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2549026, 1.2445600
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.3133962, 1.2913150
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.8183997, 1.8105547
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.2949693, 1.2833140
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6309628, 1.6091985

Time for backsubstitution: 14.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5748

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9560137, upper bound: 0.9675647
time: 3.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9650641, upper bound: 0.9584906
time: 3.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.8758540, 1.8310552
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5290349, 1.5336574
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3675606, 1.3790300
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2537553, 1.2457073
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.3102868, 1.2944242
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.8109319, 1.8180223
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.2873015, 1.2909819
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6269164, 1.6132457

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5748

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9595994, upper bound: 0.9639987
time: 3.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9686498, upper bound: 0.9549480
time: 3.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.8546948, 1.8521979
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.5271797, 1.5355121
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3753762, 1.3712142
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.2607610, 1.2387002
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4528704, 1.4528704
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.2886326, 1.3160743
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.8080409, 1.8209069
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.2887561, 1.2895273
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.6162114, 1.6239510

Time for backsubstitution: 14.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5748

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9689039, upper bound: 0.9546665
time: 3.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9779543, upper bound: 0.9455923
time: 3.72 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 22.30 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 22.30
Output dim: 1, lower bound: -0.9560137, upper bound: 0.9675647
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 22.30
Output dim: 1, lower bound: -0.9650641, upper bound: 0.9584906
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 22.30
Output dim: 1, lower bound: -0.9595994, upper bound: 0.9639987
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 22.30
Output dim: 1, lower bound: -0.9686498, upper bound: 0.9549480
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 22.30
Output dim: 1, lower bound: -0.9689039, upper bound: 0.9546665
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 22.30
Output dim: 1, lower bound: -0.9779543, upper bound: 0.9455923
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.30
Output dim: 1, lower bound: -0.9815761, upper bound: 0.9511324
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.30
Output dim: 1, lower bound: -0.9641515, upper bound: 0.9685379
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.30
Output dim: 1, lower bound: -0.9477990, upper bound: 0.9649711
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.30
Output dim: 1, lower bound: -0.9770483, upper bound: 0.9556468
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.30
Output dim: 1, lower bound: -0.9806345, upper bound: 0.9520809
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.30
Output dim: 1, lower bound: -0.9684344, upper bound: 0.9642809
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.30
Output dim: 1, lower bound: -0.9720059, upper bound: 0.9606945
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.30
Output dim: 1, lower bound: -0.9813247, upper bound: 0.9513834
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.30
Output dim: 1, lower bound: -0.9848969, upper bound: 0.9477977
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.30
Output dim: 1, lower bound: -0.9674860, upper bound: 0.9652225
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.30
Output dim: 1, lower bound: -0.9710575, upper bound: 0.9616360
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.30
Output dim: 1, lower bound: -0.9803832, upper bound: 0.9523318
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.30
Output dim: 1, lower bound: -0.9839553, upper bound: 0.9487461
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.30
Output dim: 1, lower bound: -0.9477990, upper bound: 0.9839548
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.30
Output dim: 1, lower bound: -0.9477990, upper bound: 0.9803831
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.30
Output dim: 1, lower bound: -0.9606958, upper bound: 0.9710575
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.30
Output dim: 1, lower bound: -0.9652223, upper bound: 0.9674858
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.30
Output dim: 1, lower bound: -0.9477990, upper bound: 0.9848974
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.30
Output dim: 1, lower bound: -0.9513833, upper bound: 0.9813246
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.30
Output dim: 1, lower bound: -0.9606945, upper bound: 0.9720073
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.30
Output dim: 1, lower bound: -0.9642807, upper bound: 0.9684342
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.30
Output dim: 1, lower bound: -0.9477990, upper bound: 0.9806340
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.30
Output dim: 1, lower bound: -0.9556467, upper bound: 0.9770483
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.30
Output dim: 1, lower bound: -0.9649712, upper bound: 0.9677368
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.30
Output dim: 1, lower bound: -0.9685377, upper bound: 0.9641515
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.30
Output dim: 1, lower bound: -0.9477990, upper bound: 0.9815757
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.30
Output dim: 1, lower bound: -0.9546984, upper bound: 0.9779898
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.30
Output dim: 1, lower bound: -0.9640297, upper bound: 0.9686853
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.30
Output dim: 1, lower bound: -0.9675961, upper bound: 0.9650998
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=1.509117841720581
rel_dist={1: [-0.9849775715086642, 0.9849766529340247]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 442
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 442

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7555222, upper bound: 0.7462530
time: 3.66 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7462537, upper bound: 0.7555233
time: 3.58 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 7.42 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 7.42
Output dim: 1, lower bound: -0.7555222, upper bound: 0.7462530
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 7.42
Output dim: 1, lower bound: -0.7462537, upper bound: 0.7555233

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.7677336, 1.7595387
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3879935, 1.3914340
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3261335, 1.3267739
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0693392, 1.0630721
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4353819, 1.4453433
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.2141910, 1.2154378
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6807866, 1.6843441
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1619323, 1.1622574
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4857149, 1.4847369

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 4612

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7529494, upper bound: 0.7462505
time: 3.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7555197, upper bound: 0.7436675
time: 3.63 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.7595387, 1.7677333
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3914338, 1.3879936
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3267739, 1.3261335
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0630721, 1.0693392
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4453430, 1.4353817
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.2154379, 1.2141908
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6843438, 1.6807868
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1622571, 1.1619327
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4847369, 1.4857144

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 4612

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7436683, upper bound: 0.7555201
time: 3.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7436683, upper bound: 0.7529499
time: 3.81 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 22.21 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 22.21
Output dim: 1, lower bound: -0.7529494, upper bound: 0.7462505
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 22.21
Output dim: 1, lower bound: -0.7555197, upper bound: 0.7436675
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 22.21
Output dim: 1, lower bound: -0.7436683, upper bound: 0.7555201
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 22.21
Output dim: 1, lower bound: -0.7436683, upper bound: 0.7529499

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.7676158, 1.7588532
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3867728, 1.3912088
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3245597, 1.3264894
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0688761, 1.0605704
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4353139, 1.4449940
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.2135007, 1.2152998
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6803055, 1.6817605
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1608123, 1.1620585
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4854469, 1.4833131

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 849

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7529279, upper bound: 0.7456935
time: 3.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7523925, upper bound: 0.7462288
time: 3.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.7670484, 1.7594206
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3877686, 1.3902129
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3258491, 1.3252001
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0668372, 1.0626091
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4350326, 1.4452755
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.2140529, 1.2147477
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6782031, 1.6838624
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1617337, 1.1611371
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4842911, 1.4844694

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 849

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7554983, upper bound: 0.7431178
time: 3.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7549628, upper bound: 0.7436455
time: 3.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.7594209, 1.7670481
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3902131, 1.3877684
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3252001, 1.3258491
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0626091, 1.0668374
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4452755, 1.4350324
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.2147477, 1.2140529
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6838627, 1.6782033
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1611370, 1.1617336
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4844694, 1.4842908

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 849

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7436448, upper bound: 0.7549628
time: 3.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7431171, upper bound: 0.7554981
time: 3.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.7588534, 1.7676156
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3912090, 1.3867725
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3264894, 1.3245597
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0605701, 1.0688761
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4449937, 1.4353139
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.2153001, 1.2135007
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6817603, 1.6803052
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1620585, 1.1608123
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4833136, 1.4854472

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 849

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7462281, upper bound: 0.7523925
time: 3.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7456927, upper bound: 0.7529288
time: 3.73 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 22.20 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.20
Output dim: 1, lower bound: -0.7529279, upper bound: 0.7456935
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.20
Output dim: 1, lower bound: -0.7523925, upper bound: 0.7462288
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.20
Output dim: 1, lower bound: -0.7554983, upper bound: 0.7431178
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.20
Output dim: 1, lower bound: -0.7549628, upper bound: 0.7436455
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.20
Output dim: 1, lower bound: -0.7436448, upper bound: 0.7549628
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.20
Output dim: 1, lower bound: -0.7431171, upper bound: 0.7554981
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.20
Output dim: 1, lower bound: -0.7462281, upper bound: 0.7523925
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.20
Output dim: 1, lower bound: -0.7456927, upper bound: 0.7529288

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.7438564, 1.7303486
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3916335, 1.3953351
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3236120, 1.3256998
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0670924, 1.0584896
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4372702, 1.4473741
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.2044525, 1.2077621
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6548574, 1.6605611
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1609560, 1.1622279
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4879155, 1.4862211

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6141

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7454448, upper bound: 0.7456819
time: 3.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7529151, upper bound: 0.7382078
time: 3.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.7391109, 1.7350945
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3908989, 1.3960699
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3237698, 1.3255417
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0667953, 1.0587866
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4376941, 1.4469502
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.2059629, 1.2062517
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6591060, 1.6563125
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1609815, 1.1622022
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4883552, 1.4857814

Time for backsubstitution: 14.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 6141

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7449068, upper bound: 0.7462161
time: 3.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7523810, upper bound: 0.7387458
time: 3.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.7432890, 1.7309158
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3926296, 1.3943394
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3249013, 1.3244104
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0650535, 1.0605284
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4369888, 1.4476559
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.2050047, 1.2072098
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6527555, 1.6626632
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1618775, 1.1613064
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4867592, 1.4873774

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6141

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7480151, upper bound: 0.7431064
time: 3.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7554854, upper bound: 0.7356376
time: 3.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.7385435, 1.7356620
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3918948, 1.3950742
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3250592, 1.3242524
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0647564, 1.0608255
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4374123, 1.4472318
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.2065151, 1.2056994
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6570041, 1.6584146
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1619030, 1.1612809
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4871988, 1.4869378

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6141

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7474771, upper bound: 0.7436329
time: 3.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7549513, upper bound: 0.7361756
time: 3.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.7356625, 1.7385433
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3950741, 1.3918947
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3242524, 1.3250594
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0608253, 1.0647566
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4472318, 1.4374125
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.2056994, 1.2065151
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6584146, 1.6570036
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1612808, 1.1619031
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4869380, 1.4871986

Time for backsubstitution: 14.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6141

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7361747, upper bound: 0.7549521
time: 3.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7436322, upper bound: 0.7474770
time: 3.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.7309160, 1.7432892
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3943393, 1.3926295
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3244102, 1.3249013
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0605283, 1.0650537
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4476557, 1.4369886
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.2072098, 1.2050047
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6626632, 1.6527553
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1613063, 1.1618774
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4873776, 1.4867589

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6141

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7356383, upper bound: 0.7554865
time: 3.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7431057, upper bound: 0.7480150
time: 3.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.7350950, 1.7391107
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3960700, 1.3908991
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3255417, 1.3237700
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0587864, 1.0667955
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4469500, 1.4376941
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.2062516, 1.2059629
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6563127, 1.6591060
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1622022, 1.1609817
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4857817, 1.4883549

Time for backsubstitution: 14.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6141

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7387450, upper bound: 0.7523818
time: 3.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7462152, upper bound: 0.7449069
time: 3.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.7303486, 1.7438567
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3953352, 1.3916339
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.3256996, 1.3236120
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0584893, 1.0670925
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.4473743, 1.4372702
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.2077620, 1.2044525
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6605613, 1.6548574
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1622277, 1.1609560
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4862208, 1.4879155

Time for backsubstitution: 14.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6141

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7382070, upper bound: 0.7529159
time: 3.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7456811, upper bound: 0.7454457
time: 3.80 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 22.29 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.29
Output dim: 1, lower bound: -0.7454448, upper bound: 0.7456819
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.29
Output dim: 1, lower bound: -0.7529151, upper bound: 0.7382078
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.29
Output dim: 1, lower bound: -0.7449068, upper bound: 0.7462161
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.29
Output dim: 1, lower bound: -0.7523810, upper bound: 0.7387458
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.29
Output dim: 1, lower bound: -0.7480151, upper bound: 0.7431064
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.29
Output dim: 1, lower bound: -0.7554854, upper bound: 0.7356376
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.29
Output dim: 1, lower bound: -0.7474771, upper bound: 0.7436329
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.29
Output dim: 1, lower bound: -0.7549513, upper bound: 0.7361756
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.29
Output dim: 1, lower bound: -0.7361747, upper bound: 0.7549521
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.29
Output dim: 1, lower bound: -0.7436322, upper bound: 0.7474770
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.29
Output dim: 1, lower bound: -0.7356383, upper bound: 0.7554865
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.29
Output dim: 1, lower bound: -0.7431057, upper bound: 0.7480150
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.29
Output dim: 1, lower bound: -0.7387450, upper bound: 0.7523818
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.29
Output dim: 1, lower bound: -0.7462152, upper bound: 0.7449069
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.29
Output dim: 1, lower bound: -0.7382070, upper bound: 0.7529159
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.29
Output dim: 1, lower bound: -0.7456811, upper bound: 0.7454457

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6767120, 1.6497698
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3899045, 1.3938943
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2738700, 1.2842586
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0469515, 1.0416970
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3676054, 1.3893414
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1337242, 1.1228831
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6440973, 1.6438818
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1431495, 1.1408710
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4456253, 1.4355016

Time for backsubstitution: 14.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5830

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7331966, upper bound: 0.7456765
time: 3.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7454396, upper bound: 0.7432403
time: 3.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6632776, 1.6631942
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3901930, 1.3936058
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2821710, 1.2759576
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0502989, 1.0383484
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3792369, 1.3777094
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1195736, 1.1370313
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6381779, 1.6497972
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1395990, 1.1444213
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4371958, 1.4439316

Time for backsubstitution: 14.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5830

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7504734, upper bound: 0.7382025
time: 3.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7529098, upper bound: 0.7357662
time: 3.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6719565, 1.6545157
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3891697, 1.3946291
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2740281, 1.2841005
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0466545, 1.0419939
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3680294, 1.3889170
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1352322, 1.1213727
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6483421, 1.6396332
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1431750, 1.1408453
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4460659, 1.4350619

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5830

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7331966, upper bound: 0.7462100
time: 3.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7331966, upper bound: 0.7437736
time: 4.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6585321, 1.6679497
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3894582, 1.3943406
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2823288, 1.2757998
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0500023, 1.0386455
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3796613, 1.3772855
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1210840, 1.1355233
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6424265, 1.6455524
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1396245, 1.1443958
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4376354, 1.4434917

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5830

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7499393, upper bound: 0.7387405
time: 3.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7523757, upper bound: 0.7363041
time: 3.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6761446, 1.6503372
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3909004, 1.3928984
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2751594, 1.2829692
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0449126, 1.0437357
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3673236, 1.3896229
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1342764, 1.1223309
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6419954, 1.6459839
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1440710, 1.1399496
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4444695, 1.4366579

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5830

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7455735, upper bound: 0.7431010
time: 3.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7480099, upper bound: 0.7406699
time: 3.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6627107, 1.6637616
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3911886, 1.3926102
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2834601, 1.2746682
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0482605, 1.0403874
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3789551, 1.3779910
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1201258, 1.1364790
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6360760, 1.6518993
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1405205, 1.1435000
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4360399, 1.4450879

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5830

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7530437, upper bound: 0.7356322
time: 3.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7554801, upper bound: 0.7331958
time: 3.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6713891, 1.6550832
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3901656, 1.3936332
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2753174, 1.2828112
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0446155, 1.0440326
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3677478, 1.3891985
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1357844, 1.1208205
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6462402, 1.6417353
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1440965, 1.1399239
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4449091, 1.4362185

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5830

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7450308, upper bound: 0.7436275
time: 3.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7474718, upper bound: 0.7412040
time: 3.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6579647, 1.6685171
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3904538, 1.3933450
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2836182, 1.2745104
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0479634, 1.0406845
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3793797, 1.3775671
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1216362, 1.1349711
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6403246, 1.6476545
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1405462, 1.1434743
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4364796, 1.4446480

Time for backsubstitution: 14.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5830

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7525096, upper bound: 0.7361703
time: 3.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7549460, upper bound: 0.7337339
time: 3.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6685171, 1.6579645
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3933449, 1.3904539
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2745104, 1.2836182
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0406845, 1.0479636
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3775671, 1.3793797
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1349711, 1.1216362
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6476545, 1.6403244
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1434742, 1.1405462
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4446478, 1.4364793

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5830

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7331966, upper bound: 0.7549476
time: 3.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7361694, upper bound: 0.7525104
time: 3.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6550832, 1.6713891
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3936334, 1.3901654
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2828112, 1.2753172
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0440323, 1.0446155
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3891985, 1.3677478
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1208205, 1.1357843
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6417351, 1.6462400
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1399239, 1.1440966
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4362183, 1.4449091

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5830

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7412033, upper bound: 0.7474726
time: 3.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7436268, upper bound: 0.7450307
time: 3.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6637621, 1.6627104
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3926101, 1.3911885
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2746685, 1.2834601
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0403874, 1.0482605
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3779910, 1.3789554
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1364791, 1.1201258
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6518993, 1.6360760
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1434997, 1.1405206
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4450874, 1.4360397

Time for backsubstitution: 14.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5830

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7331966, upper bound: 0.7554812
time: 3.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7356314, upper bound: 0.7530445
time: 3.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6503372, 1.6761444
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3928986, 1.3909003
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2829692, 1.2751594
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0437357, 1.0449126
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3896229, 1.3673239
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1223309, 1.1342764
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6459837, 1.6419952
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1399494, 1.1440710
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4366579, 1.4444695

Time for backsubstitution: 14.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5830

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7406692, upper bound: 0.7480107
time: 3.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7431003, upper bound: 0.7455734
time: 3.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6679497, 1.6585319
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3943408, 1.3894581
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2757998, 1.2823288
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0386455, 1.0500026
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3772852, 1.3796613
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1355233, 1.1210840
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6455526, 1.6424267
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1443957, 1.1396247
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4434919, 1.4376357

Time for backsubstitution: 14.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5830

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7363033, upper bound: 0.7523765
time: 3.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7331966, upper bound: 0.7499385
time: 4.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6545157, 1.6719565
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3946290, 1.3891698
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2841005, 1.2740281
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0419934, 1.0466545
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3889167, 1.3680294
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1213727, 1.1352321
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6396332, 1.6483421
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1408452, 1.1431751
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4350615, 1.4460657

Time for backsubstitution: 14.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5830

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7437736, upper bound: 0.7449023
time: 3.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7462099, upper bound: 0.7424631
time: 3.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6631947, 1.6632779
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3936059, 1.3901926
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2759578, 1.2821708
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0383484, 1.0502994
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3777094, 1.3792369
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1370313, 1.1195736
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6497974, 1.6381781
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1444212, 1.1395991
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4439316, 1.4371960

Time for backsubstitution: 14.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5830

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7331966, upper bound: 0.7529109
time: 4.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7382017, upper bound: 0.7504743
time: 3.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6497698, 1.6767118
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3938942, 1.3899044
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2842586, 1.2738700
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0416968, 1.0469515
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3893414, 1.3676054
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1228831, 1.1337242
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6438818, 1.6440973
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1408709, 1.1431496
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4355011, 1.4456258

Time for backsubstitution: 14.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5830

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7432395, upper bound: 0.7454403
time: 3.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7456759, upper bound: 0.7430040
time: 3.54 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 22.08 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.08
Output dim: 1, lower bound: -0.7331966, upper bound: 0.7456765
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.08
Output dim: 1, lower bound: -0.7454396, upper bound: 0.7432403
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.08
Output dim: 1, lower bound: -0.7504734, upper bound: 0.7382025
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.08
Output dim: 1, lower bound: -0.7529098, upper bound: 0.7357662
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.08
Output dim: 1, lower bound: -0.7331966, upper bound: 0.7462100
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.08
Output dim: 1, lower bound: -0.7331966, upper bound: 0.7437736
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.08
Output dim: 1, lower bound: -0.7499393, upper bound: 0.7387405
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.08
Output dim: 1, lower bound: -0.7523757, upper bound: 0.7363041
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.08
Output dim: 1, lower bound: -0.7455735, upper bound: 0.7431010
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.08
Output dim: 1, lower bound: -0.7480099, upper bound: 0.7406699
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.08
Output dim: 1, lower bound: -0.7530437, upper bound: 0.7356322
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.08
Output dim: 1, lower bound: -0.7554801, upper bound: 0.7331958
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.08
Output dim: 1, lower bound: -0.7450308, upper bound: 0.7436275
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.08
Output dim: 1, lower bound: -0.7474718, upper bound: 0.7412040
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.08
Output dim: 1, lower bound: -0.7525096, upper bound: 0.7361703
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.08
Output dim: 1, lower bound: -0.7549460, upper bound: 0.7337339
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.08
Output dim: 1, lower bound: -0.7331966, upper bound: 0.7549476
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.08
Output dim: 1, lower bound: -0.7361694, upper bound: 0.7525104
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.08
Output dim: 1, lower bound: -0.7412033, upper bound: 0.7474726
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.08
Output dim: 1, lower bound: -0.7436268, upper bound: 0.7450307
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.08
Output dim: 1, lower bound: -0.7331966, upper bound: 0.7554812
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.08
Output dim: 1, lower bound: -0.7356314, upper bound: 0.7530445
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.08
Output dim: 1, lower bound: -0.7406692, upper bound: 0.7480107
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.08
Output dim: 1, lower bound: -0.7431003, upper bound: 0.7455734
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.08
Output dim: 1, lower bound: -0.7363033, upper bound: 0.7523765
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.08
Output dim: 1, lower bound: -0.7331966, upper bound: 0.7499385
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.08
Output dim: 1, lower bound: -0.7437736, upper bound: 0.7449023
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.08
Output dim: 1, lower bound: -0.7462099, upper bound: 0.7424631
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.08
Output dim: 1, lower bound: -0.7331966, upper bound: 0.7529109
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.08
Output dim: 1, lower bound: -0.7382017, upper bound: 0.7504743
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.08
Output dim: 1, lower bound: -0.7432395, upper bound: 0.7454403
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.08
Output dim: 1, lower bound: -0.7456759, upper bound: 0.7430040

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6744161, 1.6461306
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3862555, 1.3915936
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2635095, 1.2777328
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0458272, 1.0399172
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3595443, 1.3842590
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1306942, 1.1180763
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6368058, 1.6323230
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1356688, 1.1290084
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4416757, 1.4292383

Time for backsubstitution: 14.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5748

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7341496, upper bound: 0.7456485
time: 3.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7429726, upper bound: 0.7366693
time: 3.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6730733, 1.6474736
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3876040, 1.3902456
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2673442, 1.2738981
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0451720, 1.0405729
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3625231, 1.3812802
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1289175, 1.1198530
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6325390, 1.6365902
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1312871, 1.1333899
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4393630, 1.4315512

Time for backsubstitution: 14.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5748

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7365035, upper bound: 0.7432117
time: 3.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7454094, upper bound: 0.7343304
time: 3.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6609817, 1.6595552
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3865440, 1.3913053
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2718103, 1.2694321
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0491750, 1.0365689
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3711758, 1.3726270
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1165435, 1.1322244
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6308863, 1.6382387
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1321182, 1.1325588
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4332452, 1.4376683

Time for backsubstitution: 14.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5748

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7415792, upper bound: 0.7381741
time: 3.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7504431, upper bound: 0.7292509
time: 3.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6596389, 1.6608982
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3878920, 1.3899571
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2756450, 1.2655973
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0485194, 1.0372243
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3741546, 1.3696482
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1147668, 1.1340011
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6266196, 1.6425059
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1277366, 1.1369405
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4309325, 1.4399812

Time for backsubstitution: 14.67 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=1.3692495822906494
rel_dist={1: [-0.7555267226325175, 0.7555256824713692]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 442
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 442

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6734536, upper bound: 0.6665525
time: 3.40 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6665511, upper bound: 0.6734548
time: 3.39 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 6.97 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 6.97
Output dim: 1, lower bound: -0.6734536, upper bound: 0.6665525
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 6.97
Output dim: 1, lower bound: -0.6665511, upper bound: 0.6734548

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.7048264, 1.6986804
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3413707, 1.3439510
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2936866, 1.2941670
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -1.0003213, 0.9956206
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3869758, 1.3944468
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1572719, 1.1582072
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6233277, 1.6259954
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3729553, 1.3679352
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1108154, 1.1110590
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4255934, 1.4248602

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 4612

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6715790, upper bound: 0.6665493
time: 3.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6734507, upper bound: 0.6646599
time: 3.45 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6986809, 1.7048266
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3439511, 1.3413708
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2941670, 1.2936866
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9956206, 1.0003210
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3944468, 1.3869758
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1582072, 1.1572720
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6259956, 1.6233275
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3679352, 1.3729553
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1110586, 1.1108154
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4248600, 1.4255934

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4612
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 4612

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6646609, upper bound: 0.6734518
time: 3.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6665484, upper bound: 0.6715799
time: 3.47 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 21.85 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 21.85
Output dim: 1, lower bound: -0.6715790, upper bound: 0.6665493
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 21.85
Output dim: 1, lower bound: -0.6734507, upper bound: 0.6646599
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 21.85
Output dim: 1, lower bound: -0.6646609, upper bound: 0.6734518
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 21.85
Output dim: 1, lower bound: -0.6665484, upper bound: 0.6715799

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.7045665, 1.6979952
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3401498, 1.3434769
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2921131, 1.2935603
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9993483, 0.9931189
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3868375, 1.3940976
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1565819, 1.1579313
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6223207, 1.6234119
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3717108, 1.3674507
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1096948, 1.1106297
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4250374, 1.4234364

Time for backsubstitution: 14.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 849

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6715634, upper bound: 0.6661242
time: 3.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6711538, upper bound: 0.6665337
time: 3.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.7041411, 1.6984208
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3408967, 1.3427302
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2930801, 1.2925932
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9978193, 0.9946480
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3866262, 1.3943088
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1569963, 1.1575171
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6207442, 1.6249888
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3724709, 1.3666904
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1103860, 1.1099386
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4241695, 1.4243035

Time for backsubstitution: 14.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 849

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6734352, upper bound: 0.6642378
time: 3.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6730237, upper bound: 0.6646440
time: 3.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6984210, 1.7041414
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3427302, 1.3408965
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2925932, 1.2930799
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9946479, 0.9978192
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3943088, 1.3866262
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1575172, 1.1569960
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6249886, 1.6207440
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3666906, 1.3724706
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1099385, 1.1103861
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4243040, 1.4241695

Time for backsubstitution: 14.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 849

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6646434, upper bound: 0.6730247
time: 3.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6642372, upper bound: 0.6734362
time: 3.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6979957, 1.7045667
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3434771, 1.3401498
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2935603, 1.2921128
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9931189, 0.9993483
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3940976, 1.3868375
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1579313, 1.1565819
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6234121, 1.6223209
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3674507, 1.3717105
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1106297, 1.1096950
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4234362, 1.4250367

Time for backsubstitution: 14.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 849

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6665327, upper bound: 0.6711556
time: 3.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6661231, upper bound: 0.6715642
time: 3.76 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 22.30 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.30
Output dim: 1, lower bound: -0.6715634, upper bound: 0.6661242
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 22.30
Output dim: 1, lower bound: -0.6711538, upper bound: 0.6665337
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.30
Output dim: 1, lower bound: -0.6734352, upper bound: 0.6642378
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.30
Output dim: 1, lower bound: -0.6730237, upper bound: 0.6646440
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.30
Output dim: 1, lower bound: -0.6646434, upper bound: 0.6730247
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.30
Output dim: 1, lower bound: -0.6642372, upper bound: 0.6734362
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 22.30
Output dim: 1, lower bound: -0.6665327, upper bound: 0.6711556
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.30
Output dim: 1, lower bound: -0.6661231, upper bound: 0.6715642

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6796217, 1.6694903
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3448272, 1.3476032
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2911651, 1.2927310
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9975646, 0.9911125
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3888996, 1.3964777
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1475337, 1.1500158
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.5968730, 1.6011505
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3706331, 1.3664927
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1098386, 1.1107926
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4275050, 1.4262345

Time for backsubstitution: 15.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 6141

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6658956, upper bound: 0.6661158
time: 3.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6715538, upper bound: 0.6604507
time: 3.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6791964, 1.6699159
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3455739, 1.3468565
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2921321, 1.2917640
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9960356, 0.9926416
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3886883, 1.3966889
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1479478, 1.1496017
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.5952961, 1.6027272
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3713932, 1.3657327
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1105298, 1.1101016
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4266381, 1.4271016

Time for backsubstitution: 15.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 6141

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6677627, upper bound: 0.6642293
time: 3.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6734256, upper bound: 0.6585821
time: 3.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6756363, 1.6734755
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3450229, 1.3474075
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2922506, 1.2916455
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9958127, 0.9928644
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3890064, 1.3963709
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1490805, 1.1484690
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.5984828, 1.5995407
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3715129, 1.3656130
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1105489, 1.1100824
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4269676, 1.4267719

Time for backsubstitution: 15.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 6141

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6673492, upper bound: 0.6646343
time: 3.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6730150, upper bound: 0.6589937
time: 3.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6734753, 1.6756365
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3474073, 1.3450230
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2916453, 1.2922509
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9928641, 0.9958128
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3963709, 1.3890066
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1484687, 1.1490806
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.5995409, 1.5984826
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3656130, 1.3715127
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1100823, 1.1105491
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4267721, 1.4269676

Time for backsubstitution: 15.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 6141

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6589929, upper bound: 0.6730170
time: 3.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6646338, upper bound: 0.6673504
time: 3.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6699162, 1.6791961
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3468564, 1.3455740
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2917640, 1.2921324
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9926414, 0.9960356
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3966889, 1.3886886
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1496017, 1.1479479
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6027272, 1.5952961
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3657327, 1.3713930
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1101016, 1.1105299
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4271016, 1.4266379

Time for backsubstitution: 14.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6141

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6585813, upper bound: 0.6734275
time: 3.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6642286, upper bound: 0.6677638
time: 3.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6694908, 1.6796215
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3476033, 1.3448273
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2927310, 1.2911654
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9911122, 0.9975647
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3964777, 1.3888998
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.1500158, 1.1475337
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.6011508, 1.5968728
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3664927, 1.3706329
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.1107925, 1.1098387
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.4262342, 1.4275053

Time for backsubstitution: 14.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6141
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6141

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6604496, upper bound: 0.6715544
time: 3.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6661145, upper bound: 0.6658974
time: 3.61 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 22.28 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 22.28
Output dim: 1, lower bound: -0.6658956, upper bound: 0.6661158
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.28
Output dim: 1, lower bound: -0.6715538, upper bound: 0.6604507
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 22.28
Output dim: 1, lower bound: -0.6677627, upper bound: 0.6642293
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.28
Output dim: 1, lower bound: -0.6734256, upper bound: 0.6585821
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 22.28
Output dim: 1, lower bound: -0.6673492, upper bound: 0.6646343
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.28
Output dim: 1, lower bound: -0.6730150, upper bound: 0.6589937
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.28
Output dim: 1, lower bound: -0.6589929, upper bound: 0.6730170
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 22.28
Output dim: 1, lower bound: -0.6646338, upper bound: 0.6673504
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.28
Output dim: 1, lower bound: -0.6585813, upper bound: 0.6734275
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 22.28
Output dim: 1, lower bound: -0.6642286, upper bound: 0.6677638
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.28
Output dim: 1, lower bound: -0.6604496, upper bound: 0.6715544
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 22.28
Output dim: 1, lower bound: -0.6661145, upper bound: 0.6658974

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.5990429, 1.5989799
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3433142, 1.3458740
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2476490, 1.2429891
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9799345, 0.9709714
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3279586, 1.3268130
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.0626547, 1.0757480
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.5801935, 1.5889080
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3931863
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.0884818, 1.0920986
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.3767853, 1.3818374

Time for backsubstitution: 14.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5830

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6696396, upper bound: 0.6604470
time: 3.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6715494, upper bound: 0.6584761
time: 3.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.5986171, 1.5994055
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3440611, 1.3451272
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2486160, 1.2420220
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9784052, 0.9725005
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3277473, 1.3270242
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.0630689, 1.0753338
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.5786171, 1.5904844
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3924260
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.0891727, 1.0914075
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.3759184, 1.3827047

Time for backsubstitution: 14.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5830

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6714507, upper bound: 0.6585777
time: 3.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6734215, upper bound: 0.6566652
time: 3.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.5950575, 1.6029720
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3435102, 1.3456782
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2487345, 1.2419035
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9781826, 0.9727233
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3280656, 1.3267062
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.0642016, 1.0742029
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.5818033, 1.5873008
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3970261, 1.3923063
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.0891920, 1.0913882
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.3762484, 1.3823750

Time for backsubstitution: 14.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5830

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6710402, upper bound: 0.6589894
time: 3.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6730110, upper bound: 0.6570786
time: 3.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6029720, 1.5950575
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3456783, 1.3435098
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2419035, 1.2487345
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9727232, 0.9781827
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3267062, 1.3280656
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.0742028, 1.0642016
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.5873003, 1.5818033
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3923068, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.0913881, 1.0891922
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.3823748, 1.3762481

Time for backsubstitution: 14.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5830

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6570775, upper bound: 0.6730123
time: 4.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6589885, upper bound: 0.6710413
time: 3.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.5994058, 1.5986171
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3451273, 1.3440610
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2420220, 1.2486157
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9725006, 0.9784053
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3270242, 1.3277473
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.0753338, 1.0630689
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.5904846, 1.5786169
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3924260, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.0914074, 1.0891730
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.3827047, 1.3759184

Time for backsubstitution: 14.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5830

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6566659, upper bound: 0.6734230
time: 3.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6585769, upper bound: 0.6714518
time: 3.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.5989799, 1.5990427
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3458741, 1.3433143
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2429891, 1.2476487
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9709713, 0.9799345
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3268130, 1.3279586
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.0757480, 1.0626547
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.5889077, 1.5801935
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3931861, 1.3970261
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.0920986, 1.0884819
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.3818369, 1.3767858

Time for backsubstitution: 14.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5830

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6584749, upper bound: 0.6715502
time: 3.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6604456, upper bound: 0.6696410
time: 3.92 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 22.44 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 22.44
Output dim: 1, lower bound: -0.6696396, upper bound: 0.6604470
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 22.44
Output dim: 1, lower bound: -0.6715494, upper bound: 0.6584761
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 22.44
Output dim: 1, lower bound: -0.6714507, upper bound: 0.6585777
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.44
Output dim: 1, lower bound: -0.6734215, upper bound: 0.6566652
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 22.44
Output dim: 1, lower bound: -0.6710402, upper bound: 0.6589894
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.44
Output dim: 1, lower bound: -0.6730110, upper bound: 0.6570786
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.44
Output dim: 1, lower bound: -0.6570775, upper bound: 0.6730123
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 22.44
Output dim: 1, lower bound: -0.6589885, upper bound: 0.6710413
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.44
Output dim: 1, lower bound: -0.6566659, upper bound: 0.6734230
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 22.44
Output dim: 1, lower bound: -0.6585769, upper bound: 0.6714518
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.44
Output dim: 1, lower bound: -0.6584749, upper bound: 0.6715502
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 22.44
Output dim: 1, lower bound: -0.6604456, upper bound: 0.6696410

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.5949779, 1.5967739
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3414233, 1.3414785
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2411315, 1.2316616
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9766257, 0.9712124
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3219204, 1.3189631
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.0582621, 1.0718595
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.5670578, 1.5821261
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3934331, 1.3813274
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.0773103, 1.0828311
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.3696551, 1.3781760

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5748

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6663537, upper bound: 0.6566595
time: 3.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6734140, upper bound: 0.6496220
time: 3.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.5914187, 1.6003401
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3408723, 1.3420295
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2412498, 1.2315431
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9764030, 0.9714352
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3222387, 1.3186450
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.0593946, 1.0707287
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.5702450, 1.5789425
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3935523, 1.3812077
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.0773296, 1.0828118
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.3699851, 1.3778462

Time for backsubstitution: 14.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5748

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6659482, upper bound: 0.6570728
time: 3.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6730037, upper bound: 0.6500307
time: 3.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.6003404, 1.5914185
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3420293, 1.3408724
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2315431, 1.2412500
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9714353, 0.9764031
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3186452, 1.3222387
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.0707285, 1.0593948
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.5789425, 1.5702446
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3812079, 1.3935525
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.0828118, 1.0773296
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.3778462, 1.3699849

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5748

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6500301, upper bound: 0.6730048
time: 3.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6570710, upper bound: 0.6659489
time: 3.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.5967736, 1.5949781
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3414783, 1.3414234
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2316616, 1.2411315
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9712126, 0.9766257
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3189631, 1.3219204
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.0718596, 1.0582621
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.5821259, 1.5670583
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3813272, 1.3934326
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.0828311, 1.0773103
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.3781762, 1.3696554

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5748

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6496214, upper bound: 0.6734149
time: 3.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6496230, upper bound: 0.6663540
time: 4.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.5963483, 1.5954034
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3422251, 1.3406764
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2326286, 1.2401645
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9696834, 0.9781547
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3187518, 1.3221316
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.0722735, 1.0578479
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.5805495, 1.5686347
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3820872, 1.3926723
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.0835223, 1.0766193
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.3773084, 1.3705227

Time for backsubstitution: 14.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5748
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5748

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6515581, upper bound: 0.6715438
time: 3.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6584686, upper bound: 0.6644139
time: 3.89 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 22.37 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 22.37
Output dim: 1, lower bound: -0.6663537, upper bound: 0.6566595
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 22.37
Output dim: 1, lower bound: -0.6734140, upper bound: 0.6496220
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 22.37
Output dim: 1, lower bound: -0.6659482, upper bound: 0.6570728
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 22.37
Output dim: 1, lower bound: -0.6730037, upper bound: 0.6500307
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 22.37
Output dim: 1, lower bound: -0.6500301, upper bound: 0.6730048
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 22.37
Output dim: 1, lower bound: -0.6570710, upper bound: 0.6659489
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 22.37
Output dim: 1, lower bound: -0.6496214, upper bound: 0.6734149
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 22.37
Output dim: 1, lower bound: -0.6496230, upper bound: 0.6663540
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 22.37
Output dim: 1, lower bound: -0.6515581, upper bound: 0.6715438
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 22.37
Output dim: 1, lower bound: -0.6584686, upper bound: 0.6644139

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.5818105, 1.5942934
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3403261, 1.3356515
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2396569, 1.2238319
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9760830, 0.9682692
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3189223, 1.3030460
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.0537562, 1.0710100
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.5669663, 1.5816414
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3925786, 1.3765030
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.0730011, 1.0820258
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.3492713, 1.3743454

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 916

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6714354, upper bound: 0.6496147
time: 3.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6734117, upper bound: 0.6496140
time: 3.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.5782504, 1.5978599
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3397753, 1.3362025
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2397754, 1.2237134
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9758608, 0.9684920
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3192408, 1.3027279
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.0548890, 1.0698800
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.5701525, 1.5784581
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3926978, 1.3763833
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.0730202, 1.0820066
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.3496013, 1.3740156

Time for backsubstitution: 14.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 916

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6710251, upper bound: 0.6500235
time: 3.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6730014, upper bound: 0.6500227
time: 3.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.5978599, 1.5782511
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3362026, 1.3397752
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2237132, 1.2397757
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9684920, 0.9758607
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3027279, 1.3192406
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.0698800, 1.0548890
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.5784581, 1.5701525
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3763833, 1.3926980
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.0820067, 1.0730202
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.3740153, 1.3496010

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 916

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6496150, upper bound: 0.6730021
time: 3.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6500229, upper bound: 0.6710250
time: 3.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.5942931, 1.5818105
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3356516, 1.3403260
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2238317, 1.2396569
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9682691, 0.9760832
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3030460, 1.3189223
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.0710101, 1.0537562
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.5816414, 1.5669663
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3765025, 1.3925784
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.0820258, 1.0730009
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.3743453, 1.3492715

Time for backsubstitution: 14.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 916

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6496150, upper bound: 0.6734129
time: 3.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6496141, upper bound: 0.6714364
time: 3.99 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 22.60 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 22.60
Output dim: 1, lower bound: -0.6714354, upper bound: 0.6496147
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 22.60
Output dim: 1, lower bound: -0.6734117, upper bound: 0.6496140
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 22.60
Output dim: 1, lower bound: -0.6710251, upper bound: 0.6500235
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 22.60
Output dim: 1, lower bound: -0.6730014, upper bound: 0.6500227
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 22.60
Output dim: 1, lower bound: -0.6496150, upper bound: 0.6730021
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 22.60
Output dim: 1, lower bound: -0.6500229, upper bound: 0.6710250
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 22.60
Output dim: 1, lower bound: -0.6496150, upper bound: 0.6734129
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 22.60
Output dim: 1, lower bound: -0.6496141, upper bound: 0.6714364

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.5812802, 1.5940583
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3400037, 1.3349236
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2390335, 1.2224166
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9753397, 0.9679404
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3185370, 1.3021722
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.0533435, 1.0708280
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.5657978, 1.5811243
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3919864, 1.3751595
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.0723403, 1.0817362
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.3478231, 1.3737067

Time for backsubstitution: 14.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6734107, upper bound: 0.6496139
time: 3.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6667386, upper bound: 0.6496135
time: 3.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.5777206, 1.5976248
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3394530, 1.3354746
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2391520, 1.2222981
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9751170, 0.9681633
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3188555, 1.3018541
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.0544765, 1.0696979
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.5689845, 1.5779407
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3921056, 1.3750398
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.0723596, 1.0817170
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.3481531, 1.3733768

Time for backsubstitution: 14.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6730004, upper bound: 0.6500226
time: 3.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6663302, upper bound: 0.6500221
time: 3.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.5976248, 1.5777206
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3354747, 1.3394531
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2222979, 1.2391522
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9681633, 0.9751171
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3018544, 1.3188555
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.0696981, 1.0544764
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.5779409, 1.5689840
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3750396, 1.3921058
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.0817170, 1.0723596
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.3733768, 1.3481531

Time for backsubstitution: 14.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6500214, upper bound: 0.6663322
time: 3.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6500220, upper bound: 0.6730023
time: 3.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.5940585, 1.5812802
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3349235, 1.3400036
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2224164, 1.2390337
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9679406, 0.9753395
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3021722, 1.3185370
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.0708282, 1.0533437
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.5811243, 1.5657976
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3751597, 1.3919859
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.0817361, 1.0723404
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.3737068, 1.3478234

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6496143, upper bound: 0.6667390
time: 3.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6496149, upper bound: 0.6734120
time: 3.80 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 22.36 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 22.36
Output dim: 1, lower bound: -0.6734107, upper bound: 0.6496139
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 22.36
Output dim: 1, lower bound: -0.6667386, upper bound: 0.6496135
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 22.36
Output dim: 1, lower bound: -0.6730004, upper bound: 0.6500226
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 22.36
Output dim: 1, lower bound: -0.6663302, upper bound: 0.6500221
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 22.36
Output dim: 1, lower bound: -0.6500214, upper bound: 0.6663322
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 22.36
Output dim: 1, lower bound: -0.6500220, upper bound: 0.6730023
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 22.36
Output dim: 1, lower bound: -0.6496143, upper bound: 0.6667390
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 22.36
Output dim: 1, lower bound: -0.6496149, upper bound: 0.6734120

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.5813322, 1.5941021
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3400248, 1.3349487
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2390385, 1.2224221
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9753516, 0.9679508
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3185623, 1.3022017
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.0533612, 1.0708429
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.5658422, 1.5811768
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3920074, 1.3751774
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.0723196, 1.0817128
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.3478374, 1.3737184

Time for backsubstitution: 14.69 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2559
type: RSZ, layer: 3, pos: 2137
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 2481
type: RSZ, layer: 3, pos: 226
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1494
type: RSZ, layer: 3, pos: 2567
type: RSZ, layer: 3, pos: 557
type: RSZ, layer: 3, pos: 1487
type: RSZ, layer: 3, pos: 3124
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2578
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1115
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2474
type: RSZ, layer: 3, pos: 576
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 1832
type: RSZ, layer: 3, pos: 1110
type: RSZ, layer: 3, pos: 2327
type: RSZ, layer: 3, pos: 905
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 1261
type: RSZ, layer: 3, pos: 1409
type: RSZ, layer: 3, pos: 2454

Time for candidate selection: 0.33 seconds

### Candidate
type: RSZ, layer: 3, pos: 2559

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6207333, upper bound: 0.6171031
time: 3.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6434678, upper bound: 0.6081875
time: 3.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.5777721, 1.5976686
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3394743, 1.3354999
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2391572, 1.2223036
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9751291, 0.9681736
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3188803, 1.3018837
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.0544939, 1.0697128
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.5690289, 1.5779932
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3921270, 1.3750577
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.0723389, 1.0816936
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.3481674, 1.3733885

Time for backsubstitution: 14.68 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2559
type: RSZ, layer: 3, pos: 2137
type: RSZ, layer: 3, pos: 1451
type: RSZ, layer: 3, pos: 1452
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 2481
type: RSZ, layer: 3, pos: 226
type: RSZ, layer: 3, pos: 1850
type: RSZ, layer: 3, pos: 2340
type: RSZ, layer: 3, pos: 1494
type: RSZ, layer: 3, pos: 2567
type: RSZ, layer: 3, pos: 557
type: RSZ, layer: 3, pos: 1487
type: RSZ, layer: 3, pos: 3124
type: RSZ, layer: 3, pos: 772
type: RSZ, layer: 3, pos: 2578
type: RSZ, layer: 3, pos: 417
type: RSZ, layer: 3, pos: 907
type: RSZ, layer: 3, pos: 1781
type: RSZ, layer: 3, pos: 1115
type: RSZ, layer: 3, pos: 1507
type: RSZ, layer: 3, pos: 1922
type: RSZ, layer: 3, pos: 2474
type: RSZ, layer: 3, pos: 576
type: RSZ, layer: 3, pos: 2124
type: RSZ, layer: 3, pos: 1832
type: RSZ, layer: 3, pos: 1110
type: RSZ, layer: 3, pos: 2327
type: RSZ, layer: 3, pos: 905
type: RSZ, layer: 3, pos: 1839
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 1261
type: RSZ, layer: 3, pos: 1409
type: RSZ, layer: 3, pos: 2454

Time for candidate selection: 0.33 seconds

### Candidate
type: RSZ, layer: 3, pos: 2559

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6200446, upper bound: 0.6174663
time: 3.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6431054, upper bound: 0.6087544
time: 3.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.8134632, -6.7279902, -8.8134632, -6.7279902, -1.5976686, 1.5777724
1: 1.9534850, 3.6440954, 1.9534850, 3.6440954, -1.3354999, 1.3394743
2: -5.4696188, -3.8512685, -5.4696188, -3.8512685, -1.2223039, 1.2391570
3: -10.1650553, -8.4066534, -10.1650553, -8.4066534, -0.9681735, 0.9751292
4: -4.7845268, -3.3316565, -4.7845268, -3.3316565, -1.3018837, 1.3188803
5: -8.3735094, -6.7897739, -8.3735094, -6.7897739, -1.0697129, 1.0544939
6: -5.9832397, -3.9410968, -5.9832397, -3.9410968, -1.5779934, 1.5690286
7: -4.2095613, -2.8125353, -4.2095613, -2.8125353, -1.3750577, 1.3921270
8: -3.7387199, -2.2968702, -3.7387199, -2.2968702, -1.0816935, 1.0723389
9: -11.0502615, -9.1371479, -11.0502615, -9.1371479, -1.3733883, 1.3481669

Time for backsubstitution: 14.69 seconds
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=1.3226265907287598
rel_dist={1: [-0.6734581938937723, 0.6734593550965497]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2421.64 seconds
