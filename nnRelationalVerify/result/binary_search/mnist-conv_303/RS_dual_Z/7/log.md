## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 0.68968146896
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.2382860, 2.2382860)
1: (1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.6527927, 1.6527927)
2: (-4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.6866517, 1.6866517)
3: (-11.0800304, -8.8735428, -11.0800304, -8.8735428, -2.2064877, 2.2064877)
4: (-5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7911050, 1.7911050)
5: (-9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405)
6: (-6.5653353, -4.2852068, -6.5653353, -4.2852068, -2.2801285, 2.2801285)
7: (-8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4803467, 1.4803467)
8: (0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.5804996, 1.5804996)
9: (-9.4929600, -7.3942938, -9.4929600, -7.3942938, -2.0986662, 2.0986662)

## BASE Result
execution time: IAR + LP analysis = 15.09 + 32.20 = 47.29 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3552.71 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=1.3714687824249268
rel_dist={1: [-0.9396763991955983, 0.9396741022232695]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.VERIFIED, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=1.1798031330108643
rel_dist={1: [-0.6639788454347988, 0.6639788374373623]}

## Binary search (step 2) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=4, k_high=5, k_mid=4, eps_mid=0.0156250, abs_max=1.2436916828155518
rel_dist={1: [-0.7636396766977613, 0.7636396766977596]}

## Binary Search Result
Binary search time: 149.37 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.01171875


# Relational Split (RS_dual_Z) starts
Time budget: 3403.34 seconds

## Binary search (step 0) starts
Candidate k: 8, corresponding eps: 0.0312500


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.37 seconds

### Candidate
type: RSZ, layer: 3, pos: 1725

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0319960, upper bound: 0.9699264
time: 3.48 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9699263, upper bound: 1.0319965
time: 3.41 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 7.27 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 7.27
Output dim: 1, lower bound: -1.0319960, upper bound: 0.9699264
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 7.27
Output dim: 1, lower bound: -0.9699263, upper bound: 1.0319965

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.2382860, 2.2382860
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4978988, 1.4850764
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4211621, 1.4346474
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -2.0072174, 2.0083146
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7911050, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9492509, 1.9484956
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4309001, 1.4341772
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4608173, 1.4707372
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8781881, 1.9090672

Time for backsubstitution: 5.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 662

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0138222, upper bound: 0.9626992
time: 3.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0226377, upper bound: 0.9530431
time: 3.71 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.2382860, 2.2382860
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4992464, 1.4978983
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4366488, 1.4211619
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -2.0072403, 2.0072174
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7911050, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9492958, 1.9492509
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4341769, 1.4371998
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4707370, 1.4746745
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.9285278, 1.8781881

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 662

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9530433, upper bound: 1.0226381
time: 3.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9626990, upper bound: 1.0138225
time: 3.78 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 13.01 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 13.01
Output dim: 1, lower bound: -1.0138222, upper bound: 0.9626992
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 13.01
Output dim: 1, lower bound: -1.0226377, upper bound: 0.9530431
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 13.01
Output dim: 1, lower bound: -0.9530433, upper bound: 1.0226381
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 13.01
Output dim: 1, lower bound: -0.9626990, upper bound: 1.0138225

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.2254448, 2.2165866
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4921954, 1.4866071
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4015405, 1.4463075
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -2.0116839, 1.9828691
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7869453, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9450586, 1.9593122
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4335856, 1.4290473
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4534235, 1.4673226
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8877187, 1.8866911

Time for backsubstitution: 5.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 1928

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0080507, upper bound: 0.9439184
time: 3.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9954098, upper bound: 0.9568930
time: 3.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.2382860, 2.2237761
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4978988, 1.4793737
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4211621, 1.4150258
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9817724, 2.0083146
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7911050, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9492509, 1.9443030
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4257703, 1.4341772
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4608173, 1.4633429
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8558121, 1.9090672

Time for backsubstitution: 5.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 3, pos: 1928

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0167092, upper bound: 0.9343668
time: 3.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0054330, upper bound: 0.9476472
time: 3.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.2237763, 2.2169559
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4935439, 1.4994287
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4170270, 1.4328221
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -2.0117064, 1.9817724
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7911050, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9451025, 1.9600675
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4368625, 1.4320700
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4633427, 1.4712598
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.9380584, 1.8558121

Time for backsubstitution: 5.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 3, pos: 1928

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9476470, upper bound: 1.0054330
time: 3.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9343667, upper bound: 1.0167097
time: 3.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.2382860, 2.2241454
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4992464, 1.4921956
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4366488, 1.4015404
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9817948, 2.0072174
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7911050, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9492958, 1.9450583
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4290471, 1.4371998
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4707370, 1.4672801
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.9061518, 1.8781881

Time for backsubstitution: 5.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 3, pos: 1928

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9568929, upper bound: 0.9954098
time: 3.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9439185, upper bound: 1.0080511
time: 3.82 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 13.23 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.23
Output dim: 1, lower bound: -1.0080507, upper bound: 0.9439184
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.23
Output dim: 1, lower bound: -0.9954098, upper bound: 0.9568930
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.23
Output dim: 1, lower bound: -1.0167092, upper bound: 0.9343668
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.23
Output dim: 1, lower bound: -1.0054330, upper bound: 0.9476472
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.23
Output dim: 1, lower bound: -0.9476470, upper bound: 1.0054330
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.23
Output dim: 1, lower bound: -0.9343667, upper bound: 1.0167097
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.23
Output dim: 1, lower bound: -0.9568929, upper bound: 0.9954098
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.23
Output dim: 1, lower bound: -0.9439185, upper bound: 1.0080511

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1344919, 2.0956717
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.5144300, 1.4839659
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.3996766, 1.4445333
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9839454, 1.9585214
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7675104, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9283633, 1.9472370
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4214048, 1.4116025
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4723682, 1.4755235
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8843775, 1.8829949

Time for backsubstitution: 5.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 3, pos: 1935

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9080755, upper bound: 0.8483544
time: 3.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9080755, upper bound: 0.8483544
time: 3.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1045299, 2.1577644
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4895544, 1.4944301
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.3997662, 1.4462774
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9982262, 1.9551311
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7895145, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9387703, 1.9426174
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4161406, 1.4260721
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4616241, 1.4926484
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8840227, 1.8856924

Time for backsubstitution: 5.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 1935

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9060989, upper bound: 0.8508479
time: 3.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9060989, upper bound: 0.8508479
time: 3.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1528077, 2.1028612
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.5193958, 1.4767325
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4190955, 1.4132516
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9540339, 1.9841862
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7732344, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9315701, 1.9322269
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4165525, 1.4170341
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4797482, 1.4715438
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8529310, 1.9053733

Time for backsubstitution: 5.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 3, pos: 1935

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9099016, upper bound: 0.8444460
time: 3.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9099016, upper bound: 0.8444460
time: 3.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1228456, 2.1649542
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4945202, 1.4871964
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4191852, 1.4149957
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9683146, 1.9807954
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7911050, 1.7907357
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9429631, 1.9276083
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4083261, 1.4312022
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4690042, 1.4886687
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8521156, 1.9080687

Time for backsubstitution: 5.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 3, pos: 1935

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9095630, upper bound: 0.8479485
time: 3.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9095630, upper bound: 0.8479485
time: 3.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1333323, 2.0960412
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.5158639, 1.4967875
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4151022, 1.4310479
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9839678, 1.9574246
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7907357, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9284077, 1.9479923
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4246922, 1.4146256
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4823117, 1.4794664
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.9347172, 1.8521159

Time for backsubstitution: 5.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 1935

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8479490, upper bound: 0.9095633
time: 3.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8479490, upper bound: 0.9095633
time: 3.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1028609, 2.1581335
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4909883, 1.5072517
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4151919, 1.4327919
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9982491, 1.9540343
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7911050, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9388146, 1.9433727
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4194174, 1.4290948
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4715438, 1.4965856
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.9343619, 1.8548136

Time for backsubstitution: 5.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 1935

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8444463, upper bound: 0.9099016
time: 3.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8444463, upper bound: 0.9099016
time: 3.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1516480, 2.1032310
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.5208297, 1.4895544
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4345217, 1.3997662
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9540563, 1.9830890
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7911050, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9316144, 1.9329822
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4198399, 1.4200571
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4896917, 1.4754868
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.9032707, 1.8744943

Time for backsubstitution: 5.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 1935

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8508481, upper bound: 0.9060992
time: 3.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8508481, upper bound: 0.9060992
time: 3.49 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1211767, 2.1653233
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4959540, 1.5000184
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4346113, 1.4015102
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9683371, 1.9796987
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7911050, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9430079, 1.9283636
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4116030, 1.4342246
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4789233, 1.4926059
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.9024553, 1.8771899

Time for backsubstitution: 5.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 1935

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8483546, upper bound: 0.9080757
time: 3.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8483546, upper bound: 0.9080757
time: 3.65 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 12.99 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.99
Output dim: 1, lower bound: -0.9080755, upper bound: 0.8483544
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.99
Output dim: 1, lower bound: -0.9080755, upper bound: 0.8483544
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.99
Output dim: 1, lower bound: -0.9060989, upper bound: 0.8508479
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.99
Output dim: 1, lower bound: -0.9060989, upper bound: 0.8508479
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.99
Output dim: 1, lower bound: -0.9099016, upper bound: 0.8444460
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.99
Output dim: 1, lower bound: -0.9099016, upper bound: 0.8444460
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.99
Output dim: 1, lower bound: -0.9095630, upper bound: 0.8479485
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.99
Output dim: 1, lower bound: -0.9095630, upper bound: 0.8479485
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.99
Output dim: 1, lower bound: -0.8479490, upper bound: 0.9095633
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.99
Output dim: 1, lower bound: -0.8479490, upper bound: 0.9095633
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.99
Output dim: 1, lower bound: -0.8444463, upper bound: 0.9099016
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.99
Output dim: 1, lower bound: -0.8444463, upper bound: 0.9099016
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.99
Output dim: 1, lower bound: -0.8508481, upper bound: 0.9060992
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.99
Output dim: 1, lower bound: -0.8508481, upper bound: 0.9060992
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.99
Output dim: 1, lower bound: -0.8483546, upper bound: 0.9080757
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.99
Output dim: 1, lower bound: -0.8483546, upper bound: 0.9080757

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1317091, 2.0952792
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.5116515, 1.4829047
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.3996091, 1.4451089
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9625378, 1.9515100
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7635822, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9265585, 1.9498751
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4199324, 1.4150028
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4555049, 1.4703472
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8822317, 1.8857057

Time for backsubstitution: 5.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 3, pos: 2901

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8863758, upper bound: 0.8313493
time: 3.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8943176, upper bound: 0.8245472
time: 3.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1340995, 2.0956717
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.5144300, 1.4811873
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.3996766, 1.4444659
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9769344, 1.9585214
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7675104, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9283633, 1.9454322
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4214048, 1.4101295
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4671922, 1.4755235
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8843775, 1.8808491

Time for backsubstitution: 5.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 2901

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8863758, upper bound: 0.8313493
time: 3.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8943176, upper bound: 0.8245472
time: 3.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1023626, 2.1574359
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4867759, 1.4933686
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.3996987, 1.4468529
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9768186, 1.9481196
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7859249, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9370565, 1.9452562
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4146676, 1.4294722
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4472785, 1.4880438
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8818769, 1.8884032

Time for backsubstitution: 5.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 3, pos: 2901

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8820476, upper bound: 0.8341243
time: 3.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8914606, upper bound: 0.8285038
time: 3.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1041374, 2.1577644
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4895544, 1.4916513
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.3997662, 1.4462099
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9912148, 1.9551311
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7895145, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9387703, 1.9408126
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4161406, 1.4245989
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4564481, 1.4926484
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8840227, 1.8835466

Time for backsubstitution: 5.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 2901

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8820476, upper bound: 0.8341243
time: 3.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8914606, upper bound: 0.8285038
time: 3.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1500249, 2.1024687
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.5166168, 1.4756713
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4189992, 1.4134605
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9265604, 1.9780130
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7668724, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9297838, 1.9348657
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4150801, 1.4204342
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4628844, 1.4663675
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8507853, 1.9080842

Time for backsubstitution: 5.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 2901

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8879998, upper bound: 0.8273962
time: 3.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8964063, upper bound: 0.8211418
time: 3.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1524153, 2.1028612
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.5193958, 1.4739540
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4190955, 1.4131842
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9470229, 1.9841862
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7732344, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9315701, 1.9304218
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4165525, 1.4155610
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4745717, 1.4715438
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8529310, 1.9032276

Time for backsubstitution: 5.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 2901

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8879998, upper bound: 0.8273962
time: 3.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8964063, upper bound: 0.8211418
time: 3.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1206784, 2.1646252
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4917412, 1.4861352
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4190888, 1.4152045
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9408407, 1.9746222
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7885246, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9411764, 1.9302592
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4068527, 1.4346023
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4546580, 1.4840641
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8499699, 1.9107792

Time for backsubstitution: 5.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 2901

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8851375, upper bound: 0.8324686
time: 3.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8947843, upper bound: 0.8259597
time: 3.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1224532, 2.1649542
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4945202, 1.4844179
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4191852, 1.4149282
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9613032, 1.9807954
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7911050, 1.7868075
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9429631, 1.9258032
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4083261, 1.4297290
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4638276, 1.4886687
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8521156, 1.9059227

Time for backsubstitution: 5.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 2901

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8851375, upper bound: 0.8324686
time: 3.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8947843, upper bound: 0.8259597
time: 3.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1305499, 2.0956488
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.5130849, 1.4957263
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4150348, 1.4316235
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9625602, 1.9504132
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7868075, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9266033, 1.9506302
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4232192, 1.4180257
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4654489, 1.4742899
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.9325714, 1.8548267

Time for backsubstitution: 5.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 2901

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8259602, upper bound: 0.8947846
time: 3.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8324691, upper bound: 0.8851378
time: 3.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1329403, 2.0960412
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.5158639, 1.4940090
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4151022, 1.4309804
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9769568, 1.9574246
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7907357, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9284077, 1.9461873
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4246922, 1.4131527
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4771357, 1.4794664
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.9347172, 1.8499701

Time for backsubstitution: 5.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 2901

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8259602, upper bound: 0.8947846
time: 3.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8324691, upper bound: 0.8851378
time: 3.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1006942, 2.1578045
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4882092, 1.5061905
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4151244, 1.4333675
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9768410, 1.9470229
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7911050, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9371004, 1.9460115
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4179449, 1.4324951
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4571981, 1.4919825
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.9322166, 1.8575246

Time for backsubstitution: 5.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 2901

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8211421, upper bound: 0.8964065
time: 4.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8273959, upper bound: 0.8879999
time: 4.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1024690, 2.1581335
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4909883, 1.5044730
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4151919, 1.4327245
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9912376, 1.9540343
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7911050, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9388146, 1.9415677
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4194174, 1.4276218
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4663672, 1.4965856
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.9343619, 1.8526678

Time for backsubstitution: 5.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 2901

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8211421, upper bound: 0.8964065
time: 3.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8273959, upper bound: 0.8879999
time: 3.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1488657, 2.1028383
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.5180507, 1.4884932
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4344249, 1.3999751
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9265828, 1.9769158
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7900977, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9298272, 1.9356210
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4183669, 1.4234571
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4728284, 1.4703102
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.9011250, 1.8772051

Time for backsubstitution: 5.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 3, pos: 2901

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8285040, upper bound: 0.8914609
time: 3.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8341246, upper bound: 0.8820480
time: 3.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1512561, 2.1032310
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.5208297, 1.4867756
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4345217, 1.3996987
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9470453, 1.9830890
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7911050, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9316144, 1.9311771
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4198399, 1.4185839
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4845157, 1.4754868
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.9032707, 1.8723485

Time for backsubstitution: 5.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 2901

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8285040, upper bound: 0.8914609
time: 3.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8341246, upper bound: 0.8820480
time: 3.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1190100, 2.1649942
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4931750, 1.4989572
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4345145, 1.4017191
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9408636, 1.9735255
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7911050, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9412208, 1.9310145
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4101295, 1.4376249
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4645777, 1.4880028
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.9003096, 1.8799007

Time for backsubstitution: 5.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 3, pos: 2901

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8245475, upper bound: 0.8943179
time: 3.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8313493, upper bound: 0.8863761
time: 3.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1207848, 2.1653233
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4959540, 1.4972398
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4346113, 1.4014428
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9613256, 1.9796987
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7911050, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9430079, 1.9265585
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4116030, 1.4327517
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4737473, 1.4926059
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.9024553, 1.8750439

Time for backsubstitution: 5.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 2901

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8245475, upper bound: 0.8943179
time: 3.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8313493, upper bound: 0.8863761
time: 3.84 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 13.31 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.31
Output dim: 1, lower bound: -0.8863758, upper bound: 0.8313493
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.31
Output dim: 1, lower bound: -0.8943176, upper bound: 0.8245472
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.31
Output dim: 1, lower bound: -0.8863758, upper bound: 0.8313493
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.31
Output dim: 1, lower bound: -0.8943176, upper bound: 0.8245472
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.31
Output dim: 1, lower bound: -0.8820476, upper bound: 0.8341243
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.31
Output dim: 1, lower bound: -0.8914606, upper bound: 0.8285038
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.31
Output dim: 1, lower bound: -0.8820476, upper bound: 0.8341243
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.31
Output dim: 1, lower bound: -0.8914606, upper bound: 0.8285038
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.31
Output dim: 1, lower bound: -0.8879998, upper bound: 0.8273962
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.31
Output dim: 1, lower bound: -0.8964063, upper bound: 0.8211418
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.31
Output dim: 1, lower bound: -0.8879998, upper bound: 0.8273962
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.31
Output dim: 1, lower bound: -0.8964063, upper bound: 0.8211418
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.31
Output dim: 1, lower bound: -0.8851375, upper bound: 0.8324686
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.31
Output dim: 1, lower bound: -0.8947843, upper bound: 0.8259597
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.31
Output dim: 1, lower bound: -0.8851375, upper bound: 0.8324686
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.31
Output dim: 1, lower bound: -0.8947843, upper bound: 0.8259597
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.31
Output dim: 1, lower bound: -0.8259602, upper bound: 0.8947846
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.31
Output dim: 1, lower bound: -0.8324691, upper bound: 0.8851378
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.31
Output dim: 1, lower bound: -0.8259602, upper bound: 0.8947846
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.31
Output dim: 1, lower bound: -0.8324691, upper bound: 0.8851378
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.31
Output dim: 1, lower bound: -0.8211421, upper bound: 0.8964065
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.31
Output dim: 1, lower bound: -0.8273959, upper bound: 0.8879999
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.31
Output dim: 1, lower bound: -0.8211421, upper bound: 0.8964065
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.31
Output dim: 1, lower bound: -0.8273959, upper bound: 0.8879999
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.31
Output dim: 1, lower bound: -0.8285040, upper bound: 0.8914609
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.31
Output dim: 1, lower bound: -0.8341246, upper bound: 0.8820480
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.31
Output dim: 1, lower bound: -0.8285040, upper bound: 0.8914609
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.31
Output dim: 1, lower bound: -0.8341246, upper bound: 0.8820480
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.31
Output dim: 1, lower bound: -0.8245475, upper bound: 0.8943179
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.31
Output dim: 1, lower bound: -0.8313493, upper bound: 0.8863761
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.31
Output dim: 1, lower bound: -0.8245475, upper bound: 0.8943179
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.31
Output dim: 1, lower bound: -0.8313493, upper bound: 0.8863761

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1303353, 2.0930257
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.5081344, 1.4807582
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.3996944, 1.4449483
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9664421, 1.9437838
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7565398, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9145164, 1.9543200
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4297631, 1.4120224
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4550309, 1.4686244
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8647017, 1.8729312

Time for backsubstitution: 5.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 655

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8699947, upper bound: 0.8142560
time: 3.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8692699, upper bound: 0.8169809
time: 3.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1317091, 2.0939054
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.5095048, 1.4829047
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.3994484, 1.4451089
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9548116, 1.9515100
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7635822, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9265585, 1.9378328
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4169524, 1.4150028
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4555049, 1.4698732
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8694577, 1.8857057

Time for backsubstitution: 5.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 655

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8786322, upper bound: 0.8092183
time: 3.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8777760, upper bound: 0.8108046
time: 3.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1327257, 2.0934181
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.5109124, 1.4790409
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.3997622, 1.4443054
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9808383, 1.9507947
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7604675, 1.7897334
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9163208, 1.9498770
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4312360, 1.4071493
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4667177, 1.4738009
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8668480, 1.8680747

Time for backsubstitution: 5.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 655

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8699947, upper bound: 0.8142560
time: 3.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8692699, upper bound: 0.8169809
time: 3.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1340995, 2.0942979
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.5122838, 1.4811873
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.3995161, 1.4444659
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9692078, 1.9585214
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7675104, 1.7902870
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9283633, 1.9333899
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4184248, 1.4101295
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4671922, 1.4750493
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8716035, 1.8808491

Time for backsubstitution: 5.61 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 655

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8786322, upper bound: 0.8092183
time: 3.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8777760, upper bound: 0.8108046
time: 3.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1009889, 2.1551819
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4832582, 1.4913597
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.3997843, 1.4466922
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9807224, 1.9403930
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7788820, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9247561, 1.9497008
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4244950, 1.4255817
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4468045, 1.4863212
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8643465, 1.8756292

Time for backsubstitution: 5.56 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 655

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8649690, upper bound: 0.8164329
time: 3.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8650860, upper bound: 0.8194151
time: 3.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1023626, 2.1560616
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4846292, 1.4933686
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.3995380, 1.4468529
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9690919, 1.9481196
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7859249, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9370565, 1.9332142
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4116871, 1.4294722
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4472785, 1.4875696
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8691020, 1.8884032

Time for backsubstitution: 5.56 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 655

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8748564, upper bound: 0.8118359
time: 3.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8749124, upper bound: 0.8142362
time: 3.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1027637, 2.1555109
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4860368, 1.4896424
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.3998518, 1.4460492
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9951186, 1.9474039
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7824721, 1.7846899
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9264703, 1.9452577
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4259679, 1.4207084
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4559736, 1.4909256
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8664927, 1.8707726

Time for backsubstitution: 5.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 655

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8649690, upper bound: 0.8164329
time: 3.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8650860, upper bound: 0.8194151
time: 3.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1041374, 2.1563907
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4874082, 1.4916513
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.3996058, 1.4462099
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9834886, 1.9551311
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7895145, 1.7852368
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9387703, 1.9287703
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4131606, 1.4245989
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4564481, 1.4921741
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8712482, 1.8835466

Time for backsubstitution: 5.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 655

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8748564, upper bound: 0.8118359
time: 3.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8749124, upper bound: 0.8142362
time: 3.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1486511, 2.1002150
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.5130992, 1.4735248
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4190848, 1.4132999
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9304667, 1.9702859
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7598290, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9177413, 1.9393106
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4249108, 1.4174540
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4624104, 1.4646442
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8332553, 1.8953102

Time for backsubstitution: 5.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 655

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8737468, upper bound: 0.8121318
time: 3.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8698947, upper bound: 0.8122735
time: 3.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1500249, 2.1010950
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.5144706, 1.4756713
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4188387, 1.4134605
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9188337, 1.9780130
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7668724, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9297838, 1.9228237
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4120991, 1.4204342
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4628844, 1.4658935
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8380108, 1.9080842

Time for backsubstitution: 5.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 655

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8829413, upper bound: 0.8064831
time: 3.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8783872, upper bound: 0.8062781
time: 3.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1510415, 2.1006074
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.5158782, 1.4718075
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4191809, 1.4130237
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9509287, 1.9764600
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7661915, 1.7843299
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9195275, 1.9348669
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4263837, 1.4125807
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4740977, 1.4698207
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8354011, 1.8904536

Time for backsubstitution: 5.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 655

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8737468, upper bound: 0.8121318
time: 3.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8698947, upper bound: 0.8122735
time: 3.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1524153, 2.1014874
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.5172496, 1.4739540
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4189348, 1.4131842
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9392962, 1.9841862
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7732344, 1.7848835
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9315701, 1.9183798
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4135725, 1.4155610
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4745717, 1.4710696
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8401570, 1.9032276

Time for backsubstitution: 5.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 655

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8829413, upper bound: 0.8064831
time: 3.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8783872, upper bound: 0.8062781
time: 3.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1193047, 2.1623716
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4882236, 1.4841263
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4191744, 1.4150438
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9447470, 1.9668951
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7814822, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9288759, 1.9347041
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4166796, 1.4307120
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4541841, 1.4823411
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8324399, 1.8980048

Time for backsubstitution: 5.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 655

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8719066, upper bound: 0.8159304
time: 3.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8676597, upper bound: 0.8172677
time: 3.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1206784, 2.1632514
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4895949, 1.4861352
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4189284, 1.4152045
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9331145, 1.9746222
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7885246, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9411764, 1.9182172
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4038727, 1.4346023
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4546580, 1.4835899
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8371959, 1.9107792

Time for backsubstitution: 5.60 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 655

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8824842, upper bound: 0.8103939
time: 3.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8773135, upper bound: 0.8112390
time: 3.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1210794, 2.1627002
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4910026, 1.4824090
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4192708, 1.4147675
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9652095, 1.9730697
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7866797, 1.7792172
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9306622, 1.9302483
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4181530, 1.4258387
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4633536, 1.4869454
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8345857, 1.8931482

Time for backsubstitution: 5.62 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 655

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8719066, upper bound: 0.8159304
time: 3.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8676597, upper bound: 0.8172677
time: 3.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1224532, 2.1635799
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4923739, 1.4844179
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4190245, 1.4149282
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9535766, 1.9807954
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7911050, 1.7797651
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9429631, 1.9137611
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4053452, 1.4297290
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4638276, 1.4881945
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8393416, 1.9059227

Time for backsubstitution: 5.65 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 655

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8824842, upper bound: 0.8103939
time: 3.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8773135, upper bound: 0.8112390
time: 3.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1291761, 2.0933952
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.5095677, 1.4935801
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4151204, 1.4314629
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9664645, 1.9426866
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7797651, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9145255, 1.9550748
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4334109, 1.4150543
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4649744, 1.4725673
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.9150414, 1.8420522

Time for backsubstitution: 5.59 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 655

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8112394, upper bound: 0.8773138
time: 3.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8103936, upper bound: 0.8824844
time: 3.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1305499, 2.0942750
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.5109382, 1.4957263
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4148743, 1.4316235
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9548335, 1.9504132
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7868075, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9266033, 1.9385881
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4202387, 1.4180257
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4654489, 1.4738157
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.9197974, 1.8548267

Time for backsubstitution: 5.73 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 655

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8172681, upper bound: 0.8676600
time: 3.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8159305, upper bound: 0.8719067
time: 3.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1315660, 2.0937877
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.5123467, 1.4918628
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4151876, 1.4308199
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9808607, 1.9496975
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7836928, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9163299, 1.9506321
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4348834, 1.4101810
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4766612, 1.4777439
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.9171872, 1.8371956

Time for backsubstitution: 5.73 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 655

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8112394, upper bound: 0.8773138
time: 3.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8103936, upper bound: 0.8824844
time: 3.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1329403, 2.0946674
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.5137172, 1.4940090
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4149415, 1.4309804
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9692302, 1.9574246
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7907357, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9284077, 1.9341450
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4217117, 1.4131527
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4771357, 1.4789922
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.9219432, 1.8499701

Time for backsubstitution: 5.76 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 655

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8172681, upper bound: 0.8676600
time: 3.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8159305, upper bound: 0.8719067
time: 3.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.0993199, 2.1555505
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4846916, 1.5041814
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4152100, 1.4332068
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9807448, 1.9392962
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7911050, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9247646, 1.9504561
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4281323, 1.4286137
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4567237, 1.4902594
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.9146862, 1.8447506

Time for backsubstitution: 5.76 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 655

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8062784, upper bound: 0.8783874
time: 3.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8064832, upper bound: 0.8829415
time: 4.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1006942, 2.1564302
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4860625, 1.5061905
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4149640, 1.4333675
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9691148, 1.9470229
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7911050, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9371004, 1.9339695
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4149640, 1.4324951
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4571981, 1.4915082
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.9194417, 1.8575246

Time for backsubstitution: 5.76 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 655

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8122737, upper bound: 0.8698946
time: 5.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8121320, upper bound: 0.8737470
time: 4.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1010947, 2.1558800
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4874706, 1.5024641
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4152775, 1.4325638
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9951415, 1.9463072
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7911050, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9264789, 1.9460125
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4296052, 1.4237404
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4658933, 1.4948628
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.9168320, 1.8398938

Time for backsubstitution: 5.71 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 655

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8062784, upper bound: 0.8783874
time: 3.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8064832, upper bound: 0.8829415
time: 3.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1024690, 2.1567597
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4888415, 1.5044730
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4150312, 1.4327245
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9835110, 1.9540343
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7911050, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9388146, 1.9295254
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4164374, 1.4276218
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4663672, 1.4961114
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.9215875, 1.8526678

Time for backsubstitution: 5.74 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 655

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8122737, upper bound: 0.8698946
time: 5.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8121320, upper bound: 0.8737470
time: 4.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1474919, 2.1005845
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.5145335, 1.4863467
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4345102, 1.3998145
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9304886, 1.9691887
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7830539, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9177504, 1.9400656
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4285576, 1.4204855
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4723544, 1.4685872
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8835950, 1.8644316

Time for backsubstitution: 5.84 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 3, pos: 655

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8142364, upper bound: 0.8749125
time: 4.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8118364, upper bound: 0.8748566
time: 3.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1488657, 2.1014645
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.5159039, 1.4884932
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4342642, 1.3999751
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9188561, 1.9769158
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7900977, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9298272, 1.9235787
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4153864, 1.4234571
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4728284, 1.4698360
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8883505, 1.8772051

Time for backsubstitution: 5.77 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=4, k_high=12, k_mid=8, eps_mid=0.0312500, abs_max=1.49924635887146
rel_dist={1: [-1.0866450021547305, 1.0866453172123824]}

## Binary search (step 1) starts
Candidate k: 5, corresponding eps: 0.0195312


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.41 seconds

### Candidate
type: RSZ, layer: 3, pos: 1725

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8326115, upper bound: 0.7790404
time: 4.14 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7790405, upper bound: 0.8326118
time: 3.74 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 8.29 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 8.29
Output dim: 1, lower bound: -0.8326115, upper bound: 0.7790404
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 8.29
Output dim: 1, lower bound: -0.7790405, upper bound: 0.8326118

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.9076419, 1.9065990
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.3062322, 1.2982187
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2123272, 1.2207557
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.7215066, 1.7221918
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5645666, 1.5790825
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6796601, 1.6791880
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2450318, 1.2470798
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2661448, 1.2723446
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.6266618, 1.6459613

Time for backsubstitution: 5.68 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 662

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8195566, upper bound: 0.7753398
time: 4.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8279557, upper bound: 0.7660593
time: 3.93 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.9065990, 1.9076419
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2982190, 1.3062325
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2207556, 1.2123272
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.7221918, 1.7215066
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5790825, 1.5645666
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6791880, 1.6796601
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2470803, 1.2450318
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2723446, 1.2661448
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.6459613, 1.6266620

Time for backsubstitution: 5.69 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 662

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7660588, upper bound: 0.8279560
time: 3.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7753396, upper bound: 0.8195572
time: 4.24 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 13.84 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 13.84
Output dim: 1, lower bound: -0.8195566, upper bound: 0.7753398
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 13.84
Output dim: 1, lower bound: -0.8279557, upper bound: 0.7660593
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 13.84
Output dim: 1, lower bound: -0.7660588, upper bound: 0.8279560
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 13.84
Output dim: 1, lower bound: -0.7753396, upper bound: 0.8195572

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.8877215, 1.8821850
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.3005292, 1.2970369
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1927056, 1.2206851
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.7147560, 1.6967468
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5603590, 1.5782950
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6754673, 1.6843760
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2447867, 1.2419500
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2587507, 1.2674377
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.6242275, 1.6235852

Time for backsubstitution: 5.68 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 1928

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8156641, upper bound: 0.7603735
time: 3.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8068988, upper bound: 0.7706758
time: 4.25 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.9076419, 1.8866787
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.3062322, 1.2925160
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2123272, 1.2011341
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6960611, 1.7221918
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5645666, 1.5748746
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6796601, 1.6749952
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2399020, 1.2470798
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2661448, 1.2649503
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.6042862, 1.6459613

Time for backsubstitution: 5.67 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 1928

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8237622, upper bound: 0.7530288
time: 4.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8139396, upper bound: 0.7617252
time: 3.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.8866787, 1.8832278
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2925160, 1.3050504
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2011340, 1.2122567
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.7154417, 1.6960611
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5748749, 1.5637791
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6749952, 1.6848481
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2468343, 1.2399018
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2649503, 1.2612379
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.6435261, 1.6042860

Time for backsubstitution: 5.69 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 1928

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7617250, upper bound: 0.8139399
time: 4.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7530305, upper bound: 0.8237621
time: 3.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.9065990, 1.8877215
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2982190, 1.3005295
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2207556, 1.1927056
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6967468, 1.7215066
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5790825, 1.5603592
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6791880, 1.6754673
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2419496, 1.2450318
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2723446, 1.2587507
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.6235852, 1.6266620

Time for backsubstitution: 5.67 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 1928

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7706755, upper bound: 0.8069006
time: 3.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7603735, upper bound: 0.8156641
time: 4.32 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 14.13 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.13
Output dim: 1, lower bound: -0.8156641, upper bound: 0.7603735
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.13
Output dim: 1, lower bound: -0.8068988, upper bound: 0.7706758
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.13
Output dim: 1, lower bound: -0.8237622, upper bound: 0.7530288
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.13
Output dim: 1, lower bound: -0.8139396, upper bound: 0.7617252
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.13
Output dim: 1, lower bound: -0.7617250, upper bound: 0.8139399
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.13
Output dim: 1, lower bound: -0.7530305, upper bound: 0.8237621
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.13
Output dim: 1, lower bound: -0.7706755, upper bound: 0.8069006
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.13
Output dim: 1, lower bound: -0.7603735, upper bound: 0.8156641

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7855325, 1.7612700
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.3134356, 1.2943957
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1908753, 1.2189109
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6870179, 1.6711278
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5409241, 1.5620160
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8257337
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6587725, 1.6705685
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2306318, 1.2245052
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2736669, 1.2756386
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.6207533, 1.6198890

Time for backsubstitution: 5.67 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 1935

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7614719, upper bound: 0.7056692
time: 4.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7614719, upper bound: 0.7056692
time: 4.15 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7668066, 1.7821980
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2978883, 1.3112493
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1909316, 1.2188761
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6895604, 1.6690087
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5441551, 1.5588598
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8244672
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6616592, 1.6676812
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2273417, 1.2296534
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2669516, 1.2836301
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.6205311, 1.6204586

Time for backsubstitution: 5.70 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 3, pos: 1935

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7583109, upper bound: 0.7109269
time: 3.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7583109, upper bound: 0.7109269
time: 3.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.8038483, 1.7657638
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.3184013, 1.2898748
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2102945, 1.1993599
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6683230, 1.6967921
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5466480, 1.5586386
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8266816
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6619787, 1.6611872
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2275991, 1.2299368
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2810464, 1.2731514
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.6010990, 1.6422675

Time for backsubstitution: 5.71 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 1935

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7676746, upper bound: 0.7017414
time: 3.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7676746, upper bound: 0.7017414
time: 3.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7851224, 1.7848082
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.3028541, 1.3067288
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2103503, 1.1993040
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6704421, 1.6946731
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5498791, 1.5554399
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8257556
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6648655, 1.6583004
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2224569, 1.2350850
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2743316, 1.2798815
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.6005898, 1.6428370

Time for backsubstitution: 5.70 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 1935

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7645334, upper bound: 0.7047929
time: 4.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7645334, upper bound: 0.7047929
time: 4.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7848082, 1.7623129
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.3067288, 1.3024092
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1993039, 1.2104826
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6877031, 1.6704421
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5554399, 1.5475326
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8257561, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6583004, 1.6710405
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2326860, 1.2224572
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2798815, 1.2694390
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.6401119, 1.6005898

Time for backsubstitution: 5.72 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 1935

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7047921, upper bound: 0.7645339
time: 3.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7047921, upper bound: 0.7645339
time: 3.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7657638, 1.7829223
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2898746, 1.3179560
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1993599, 1.2104478
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6902461, 1.6683230
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5586386, 1.5443439
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8266811, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6611872, 1.6681533
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2293892, 1.2275991
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2731514, 1.2774153
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.6398301, 1.6010993

Time for backsubstitution: 5.73 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 1935

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7017414, upper bound: 0.7676750
time: 3.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7017414, upper bound: 0.7676750
time: 3.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.8031240, 1.7668066
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.3116946, 1.2978883
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2187231, 1.1909316
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6690083, 1.6961064
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5611629, 1.5441551
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8268137, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6615067, 1.6616592
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2296534, 1.2278888
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2872610, 1.2669516
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.6204586, 1.6229682

Time for backsubstitution: 5.73 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 1935

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7109263, upper bound: 0.7583112
time: 4.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7109263, upper bound: 0.7583112
time: 3.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7840791, 1.7855330
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2948403, 1.3134356
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2187788, 1.1908754
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6711278, 1.6939874
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5643625, 1.5409241
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8277397, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6643934, 1.6587725
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2245054, 1.2330306
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2805309, 1.2736666
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.6198893, 1.6234777

Time for backsubstitution: 5.72 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 1935

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7056686, upper bound: 0.7614722
time: 4.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7056686, upper bound: 0.7614722
time: 4.20 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 14.35 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.35
Output dim: 1, lower bound: -0.7614719, upper bound: 0.7056692
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.35
Output dim: 1, lower bound: -0.7614719, upper bound: 0.7056692
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.35
Output dim: 1, lower bound: -0.7583109, upper bound: 0.7109269
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.35
Output dim: 1, lower bound: -0.7583109, upper bound: 0.7109269
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.35
Output dim: 1, lower bound: -0.7676746, upper bound: 0.7017414
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.35
Output dim: 1, lower bound: -0.7676746, upper bound: 0.7017414
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.35
Output dim: 1, lower bound: -0.7645334, upper bound: 0.7047929
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.35
Output dim: 1, lower bound: -0.7645334, upper bound: 0.7047929
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.35
Output dim: 1, lower bound: -0.7047921, upper bound: 0.7645339
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.35
Output dim: 1, lower bound: -0.7047921, upper bound: 0.7645339
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.35
Output dim: 1, lower bound: -0.7017414, upper bound: 0.7676750
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.35
Output dim: 1, lower bound: -0.7017414, upper bound: 0.7676750
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.35
Output dim: 1, lower bound: -0.7109263, upper bound: 0.7583112
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.35
Output dim: 1, lower bound: -0.7109263, upper bound: 0.7583112
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.35
Output dim: 1, lower bound: -0.7056686, upper bound: 0.7614722
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.35
Output dim: 1, lower bound: -0.7056686, upper bound: 0.7614722

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7836466, 1.7608776
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.3106570, 1.2926905
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1908078, 1.2192454
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6710086, 1.6641164
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5369959, 1.5820203
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8182135
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6569672, 1.6715403
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2291589, 1.2260778
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2611856, 1.2704623
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.6186075, 1.6207786

Time for backsubstitution: 5.70 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 2901

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7458326, upper bound: 0.6983173
time: 4.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7555461, upper bound: 0.6874464
time: 3.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7851405, 1.7612700
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.3134356, 1.2916172
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1908753, 1.2188435
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6800065, 1.6711278
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5409241, 1.5580878
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8257337
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6587725, 1.6687634
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2306318, 1.2230322
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2684903, 1.2756386
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.6207533, 1.6177433

Time for backsubstitution: 5.75 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 2901

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7458326, upper bound: 0.6983173
time: 4.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7555461, upper bound: 0.6874464
time: 3.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7653050, 1.7818055
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2951097, 1.3095441
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1908641, 1.2192105
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6737332, 1.6619973
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5402269, 1.5856576
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8169470
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6598539, 1.6686537
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2258687, 1.2311082
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2560444, 1.2784538
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.6183853, 1.6213481

Time for backsubstitution: 5.67 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 2901

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7410209, upper bound: 0.7018348
time: 4.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7522602, upper bound: 0.6936980
time: 3.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7664142, 1.7821980
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2978883, 1.3084707
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1909316, 1.2188087
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6825490, 1.6690087
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5441551, 1.5549316
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8244672
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6616592, 1.6658764
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2273417, 1.2281804
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2617755, 1.2836301
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.6205311, 1.6183128

Time for backsubstitution: 5.66 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 2901

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7410209, upper bound: 0.7018348
time: 3.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7522602, upper bound: 0.6936980
time: 4.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.8019624, 1.7653713
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.3156223, 1.2881696
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2101982, 1.1994652
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6485229, 1.6906190
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5402861, 1.5762234
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8191609
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6601925, 1.6621594
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2261262, 1.2315094
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2685657, 1.2679749
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5989532, 1.6431570

Time for backsubstitution: 5.70 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 2901

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7517682, upper bound: 0.6942104
time: 4.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7612216, upper bound: 0.6840121
time: 4.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.8034563, 1.7657638
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.3184013, 1.2870963
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2102945, 1.1992924
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6613116, 1.6967921
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5466480, 1.5547104
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8266816
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6619787, 1.6593821
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2275991, 1.2284636
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2758698, 1.2731514
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.6010990, 1.6401217

Time for backsubstitution: 5.64 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 2901

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7517682, upper bound: 0.6942104
time: 4.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7612216, upper bound: 0.6840121
time: 4.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7836208, 1.7844157
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.3000751, 1.3050239
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2102542, 1.1994910
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6519847, 1.6884999
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5435171, 1.5799913
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8182354
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6630793, 1.6592805
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2209845, 1.2365398
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2634239, 1.2747049
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5984440, 1.6437266

Time for backsubstitution: 5.69 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 2901

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7476592, upper bound: 0.6970134
time: 4.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7587099, upper bound: 0.6879167
time: 3.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7847300, 1.7848082
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.3028541, 1.3039503
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2103503, 1.1992365
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6634307, 1.6946731
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5498791, 1.5515118
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8257556
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6648655, 1.6564956
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2224569, 1.2336118
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2691550, 1.2798815
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.6005898, 1.6406913

Time for backsubstitution: 5.63 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 2901

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7476592, upper bound: 0.6970134
time: 4.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7587099, upper bound: 0.6879167
time: 3.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7829223, 1.7619205
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.3039503, 1.3007040
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1992364, 1.2108170
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6716943, 1.6634307
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5515118, 1.5675368
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8206878, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6564951, 1.6720123
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2312131, 1.2240298
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2674007, 1.2642624
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.6379662, 1.6014793

Time for backsubstitution: 5.68 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 2901

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6879164, upper bound: 0.7587102
time: 3.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6970126, upper bound: 0.7476598
time: 4.22 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7844157, 1.7623129
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.3067288, 1.2996306
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1993039, 1.2104151
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6806922, 1.6704421
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5554399, 1.5436044
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8182359, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6583004, 1.6692355
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2326860, 1.2209840
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2747049, 1.2694390
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.6401119, 1.5984440

Time for backsubstitution: 5.69 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 2901

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6879164, upper bound: 0.7587102
time: 3.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6970126, upper bound: 0.7476598
time: 4.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7642622, 1.7825298
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2870965, 1.3162508
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1992924, 1.2107821
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6744189, 1.6613116
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5547104, 1.5711422
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8264709, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6593819, 1.6691256
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2279167, 1.2290540
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2622437, 1.2722390
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.6376843, 1.6019890

Time for backsubstitution: 5.67 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 2901

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6840113, upper bound: 0.7612222
time: 3.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6942103, upper bound: 0.7517686
time: 3.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7653713, 1.7829223
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2898746, 1.3151774
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1993599, 1.2103803
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6832347, 1.6683230
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5586386, 1.5404158
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8191609, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6611872, 1.6663482
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2293892, 1.2261260
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2679749, 1.2774153
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.6398301, 1.5989535

Time for backsubstitution: 5.72 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 2901

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6840113, upper bound: 0.7612222
time: 3.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6942103, upper bound: 0.7517686
time: 3.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.8012376, 1.7664142
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.3089156, 1.2961831
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2186267, 1.1910367
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6492085, 1.6899333
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5548015, 1.5617399
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8217454, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6597204, 1.6626315
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2281804, 1.2294614
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2747803, 1.2617753
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.6183128, 1.6238577

Time for backsubstitution: 5.65 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 2901

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6936978, upper bound: 0.7522602
time: 4.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7018349, upper bound: 0.7410213
time: 4.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.8027320, 1.7668066
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.3116946, 1.2951097
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2187231, 1.1908641
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6619973, 1.6961064
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5611629, 1.5402269
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8192935, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6615067, 1.6598542
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2296534, 1.2264156
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2820849, 1.2669516
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.6204586, 1.6208224

Time for backsubstitution: 5.69 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 2901

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6936978, upper bound: 0.7522602
time: 4.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7018349, upper bound: 0.7410213
time: 4.14 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7825780, 1.7851405
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2920613, 1.3117306
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2186828, 1.1910626
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6526699, 1.6878142
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5580006, 1.5654755
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8275285, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6626072, 1.6597526
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2230325, 1.2344856
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2696238, 1.2684903
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.6177435, 1.6243675

Time for backsubstitution: 5.73 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 3, pos: 2901

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6874459, upper bound: 0.7555467
time: 4.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6983165, upper bound: 0.7458327
time: 3.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7836871, 1.7855330
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2948403, 1.3106570
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2187788, 1.1908079
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6641164, 1.6939874
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5643625, 1.5369959
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8202195, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6643934, 1.6569676
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2245054, 1.2315576
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2753549, 1.2736666
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.6198893, 1.6213319

Time for backsubstitution: 5.70 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 2901

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6874459, upper bound: 0.7555467
time: 4.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6983165, upper bound: 0.7458327
time: 3.88 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 14.10 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.10
Output dim: 1, lower bound: -0.7458326, upper bound: 0.6983173
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.10
Output dim: 1, lower bound: -0.7555461, upper bound: 0.6874464
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.10
Output dim: 1, lower bound: -0.7458326, upper bound: 0.6983173
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.10
Output dim: 1, lower bound: -0.7555461, upper bound: 0.6874464
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.10
Output dim: 1, lower bound: -0.7410209, upper bound: 0.7018348
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.10
Output dim: 1, lower bound: -0.7522602, upper bound: 0.6936980
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.10
Output dim: 1, lower bound: -0.7410209, upper bound: 0.7018348
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.10
Output dim: 1, lower bound: -0.7522602, upper bound: 0.6936980
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.10
Output dim: 1, lower bound: -0.7517682, upper bound: 0.6942104
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.10
Output dim: 1, lower bound: -0.7612216, upper bound: 0.6840121
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.10
Output dim: 1, lower bound: -0.7517682, upper bound: 0.6942104
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.10
Output dim: 1, lower bound: -0.7612216, upper bound: 0.6840121
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.10
Output dim: 1, lower bound: -0.7476592, upper bound: 0.6970134
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.10
Output dim: 1, lower bound: -0.7587099, upper bound: 0.6879167
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.10
Output dim: 1, lower bound: -0.7476592, upper bound: 0.6970134
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.10
Output dim: 1, lower bound: -0.7587099, upper bound: 0.6879167
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.10
Output dim: 1, lower bound: -0.6879164, upper bound: 0.7587102
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.10
Output dim: 1, lower bound: -0.6970126, upper bound: 0.7476598
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.10
Output dim: 1, lower bound: -0.6879164, upper bound: 0.7587102
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.10
Output dim: 1, lower bound: -0.6970126, upper bound: 0.7476598
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.10
Output dim: 1, lower bound: -0.6840113, upper bound: 0.7612222
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.10
Output dim: 1, lower bound: -0.6942103, upper bound: 0.7517686
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.10
Output dim: 1, lower bound: -0.6840113, upper bound: 0.7612222
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.10
Output dim: 1, lower bound: -0.6942103, upper bound: 0.7517686
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.10
Output dim: 1, lower bound: -0.6936978, upper bound: 0.7522602
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.10
Output dim: 1, lower bound: -0.7018349, upper bound: 0.7410213
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.10
Output dim: 1, lower bound: -0.6936978, upper bound: 0.7522602
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.10
Output dim: 1, lower bound: -0.7018349, upper bound: 0.7410213
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.10
Output dim: 1, lower bound: -0.6874459, upper bound: 0.7555467
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.10
Output dim: 1, lower bound: -0.6983165, upper bound: 0.7458327
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.10
Output dim: 1, lower bound: -0.6874459, upper bound: 0.7555467
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.10
Output dim: 1, lower bound: -0.6983165, upper bound: 0.7458327

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7822723, 1.7589540
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.3076539, 1.2905440
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1908011, 1.2190849
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6705513, 1.6563897
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5299530, 1.5746317
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8072772
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6449251, 1.6698024
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2341855, 1.2230976
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2607117, 1.2692077
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.6028609, 1.6080041

Time for backsubstitution: 5.66 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 655

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7305416, upper bound: 0.6829900
time: 4.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7298523, upper bound: 0.6843390
time: 4.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7808943, 1.7595038
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.3085108, 1.2896943
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1906474, 1.2192388
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6632824, 1.6635594
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5278249, 1.5749779
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8050799
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6552296, 1.6594982
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2261784, 1.2313278
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2607780, 1.2699881
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.6058331, 1.6089501

Time for backsubstitution: 5.71 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 655

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7401138, upper bound: 0.6729951
time: 4.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7394723, upper bound: 0.6743117
time: 4.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7837663, 1.7593465
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.3104329, 1.2894707
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1908689, 1.2186830
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6795492, 1.6634007
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5338812, 1.5506992
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8147979
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6467299, 1.6670258
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2356584, 1.2200520
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2680163, 1.2743840
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.6050067, 1.6049688

Time for backsubstitution: 5.67 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 655

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7305416, upper bound: 0.6829900
time: 4.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7298523, upper bound: 0.6843390
time: 4.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7823882, 1.7598963
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.3112893, 1.2886209
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1907151, 1.2188368
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6722798, 1.6705704
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5317526, 1.5510449
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8126006
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6570344, 1.6567214
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2276514, 1.2282820
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2680821, 1.2751644
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.6079793, 1.6059148

Time for backsubstitution: 5.67 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 3, pos: 655

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7401138, upper bound: 0.6729951
time: 3.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7394723, upper bound: 0.6743117
time: 3.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7639308, 1.7789831
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2921062, 1.3073978
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1908574, 1.2190499
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6732779, 1.6542706
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5331845, 1.5782728
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8060107
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6478119, 1.6669159
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2308934, 1.2281282
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2555699, 1.2771971
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.6026387, 1.6085737

Time for backsubstitution: 5.66 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 655

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7248794, upper bound: 0.6856308
time: 4.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7250108, upper bound: 0.6879460
time: 4.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7633257, 1.7804317
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2929635, 1.3065410
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1907034, 1.2192038
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6660070, 1.6614380
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5310569, 1.5786152
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8038139
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6581163, 1.6566114
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2228882, 1.2363603
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2556367, 1.2779796
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.6056113, 1.6095200

Time for backsubstitution: 5.67 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 655

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7361264, upper bound: 0.6778744
time: 4.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7362405, upper bound: 0.6802908
time: 4.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7650404, 1.7793756
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2948852, 1.3063242
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1909249, 1.2186481
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6820936, 1.6612816
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5371122, 1.5475473
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8135314
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6496167, 1.6641386
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2323663, 1.2252002
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2613010, 1.2823734
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.6047845, 1.6055384

Time for backsubstitution: 5.70 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 655

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7248794, upper bound: 0.6856308
time: 4.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7250108, upper bound: 0.6879460
time: 4.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7644348, 1.7808242
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2957420, 1.3054676
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1907709, 1.2188020
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6748228, 1.6684489
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5349851, 1.5478888
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8113351
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6599212, 1.6538341
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2243612, 1.2334325
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2613673, 1.2831559
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.6077571, 1.6064844

Time for backsubstitution: 5.68 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 655

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7361264, upper bound: 0.6778744
time: 4.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7362405, upper bound: 0.6802908
time: 4.27 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.8005886, 1.7634473
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.3126197, 1.2860231
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2101915, 1.1993046
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6480665, 1.6828918
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5332422, 1.5688348
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8082247
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6481500, 1.6604218
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2311528, 1.2285292
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2680917, 1.2667201
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5832067, 1.6303833

Time for backsubstitution: 5.66 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 3, pos: 655

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7382888, upper bound: 0.6771636
time: 4.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7351849, upper bound: 0.6769944
time: 4.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7992101, 1.7639971
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.3134761, 1.2851734
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2100377, 1.1994585
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6407962, 1.6900616
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5311141, 1.5691810
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8060274
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6584544, 1.6501174
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2231457, 1.2367592
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2681575, 1.2675009
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5861793, 1.6313293

Time for backsubstitution: 5.78 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 655

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7474927, upper bound: 0.6659537
time: 4.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7447730, upper bound: 0.6658601
time: 4.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.8020825, 1.7638397
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.3153987, 1.2849498
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2102876, 1.1991320
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6608553, 1.6890664
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5396051, 1.5473223
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8157458
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6499367, 1.6576445
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2326257, 1.2254834
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2753959, 1.2718966
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5853524, 1.6273479

Time for backsubstitution: 5.72 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 655

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7382888, upper bound: 0.6771636
time: 4.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7351849, upper bound: 0.6769944
time: 3.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.8007040, 1.7643895
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.3162551, 1.2841001
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2101338, 1.1992857
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6535850, 1.6962357
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5374765, 1.5476680
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8135486
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6602411, 1.6473401
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2246187, 1.2337136
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2754622, 1.2726772
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5883250, 1.6282940

Time for backsubstitution: 5.73 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 655

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7474927, upper bound: 0.6659537
time: 4.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7447730, upper bound: 0.6658601
time: 4.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7822471, 1.7815938
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2970719, 1.3028774
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2102475, 1.1993306
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6515303, 1.6807728
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5364733, 1.5726066
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8072991
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6510367, 1.6575427
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2260087, 1.2335596
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2629499, 1.2734482
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5826974, 1.6309528

Time for backsubstitution: 5.64 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 655

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7336243, upper bound: 0.6813661
time: 4.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7319738, upper bound: 0.6820490
time: 4.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7816415, 1.7830420
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2979283, 1.3020208
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2100937, 1.1994843
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6442580, 1.6879401
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5343461, 1.5729485
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8051023
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6613412, 1.6472383
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2180035, 1.2417920
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2630162, 1.2742310
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5856695, 1.6318991

Time for backsubstitution: 5.66 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 655

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7445180, upper bound: 0.6726645
time: 4.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7428629, upper bound: 0.6731753
time: 4.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7833562, 1.7819862
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2998509, 1.3018041
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2103436, 1.1990759
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6629763, 1.6869473
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5428362, 1.5441265
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8148198
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6528230, 1.6547577
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2274816, 1.2306316
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2686810, 1.2786245
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5848432, 1.6279175

Time for backsubstitution: 5.74 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 655

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7336243, upper bound: 0.6813661
time: 4.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7319738, upper bound: 0.6820490
time: 4.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7827511, 1.7834344
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.3007073, 1.3009474
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2101898, 1.1992298
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6557040, 1.6941147
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5407085, 1.5444689
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8126235
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6631274, 1.6444533
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2194769, 1.2388639
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2687469, 1.2794073
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5878158, 1.6288636

Time for backsubstitution: 5.73 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 655

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7445180, upper bound: 0.6726645
time: 4.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7428629, upper bound: 0.6731753
time: 4.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7815480, 1.7599411
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.3009472, 1.2985578
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1992297, 1.2106564
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6711340, 1.6557040
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5444689, 1.5583668
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8075552, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6444530, 1.6702745
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2364647, 1.2210495
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2669263, 1.2638547
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.6261382, 1.5887048

Time for backsubstitution: 5.74 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 655

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6731745, upper bound: 0.7428634
time: 3.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6726644, upper bound: 0.7445184
time: 3.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7800999, 1.7605467
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.3018041, 1.2977006
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1990759, 1.2108103
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6639681, 1.6629763
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5441265, 1.5604939
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8097515, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6547575, 1.6599703
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2282326, 1.2290545
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2661443, 1.2637885
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.6251922, 1.5857327

Time for backsubstitution: 5.75 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 655

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6820489, upper bound: 0.7319744
time: 4.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6813658, upper bound: 0.7336250
time: 4.18 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7830420, 1.7603335
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.3037257, 1.2974844
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1992974, 1.2102545
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6801319, 1.6627150
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5483966, 1.5344343
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8051023, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6462579, 1.6674976
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2379382, 1.2180040
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2742310, 1.2690313
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.6282840, 1.5856695

Time for backsubstitution: 5.74 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 3, pos: 655

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6731745, upper bound: 0.7428634
time: 4.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6726644, upper bound: 0.7445184
time: 3.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7815938, 1.7609391
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.3045826, 1.2966273
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1991436, 1.2104084
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6729655, 1.6699872
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5480547, 1.5365615
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8072991, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6565623, 1.6571932
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2297060, 1.2260087
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2734485, 1.2689648
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.6273379, 1.5826972

Time for backsubstitution: 5.71 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 655

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6820489, upper bound: 0.7319744
time: 4.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6813658, upper bound: 0.7336250
time: 4.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7628880, 1.7797775
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2841001, 1.3141046
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1992857, 1.2106216
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6738605, 1.6535850
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5476680, 1.5619712
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8133368, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6473398, 1.6673877
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2331660, 1.2260737
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2617698, 1.2718310
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.6258559, 1.5892146

Time for backsubstitution: 5.73 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 655

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6658596, upper bound: 0.7447732
time: 4.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6659531, upper bound: 0.7474929
time: 4.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7623382, 1.7811561
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2849498, 1.3132479
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1991320, 1.2107754
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6666923, 1.6608553
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5473218, 1.5640993
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8155341, 1.8287287
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6576443, 1.6570835
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2249367, 1.2340808
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2609892, 1.2717648
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.6249099, 1.5862422

Time for backsubstitution: 5.78 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 655

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6769937, upper bound: 0.7351855
time: 3.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6771629, upper bound: 0.7382894
time: 3.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7639971, 1.7801700
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2868791, 1.3130310
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1993532, 1.2102197
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6826768, 1.6605959
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5515957, 1.5312452
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8060274, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6491446, 1.6646106
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2346394, 1.2231457
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2675009, 1.2770073
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.6280017, 1.5861790

Time for backsubstitution: 5.69 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 655

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6658596, upper bound: 0.7447732
time: 4.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6659531, upper bound: 0.7474929
time: 4.01 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 14.08 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.08
Output dim: 1, lower bound: -0.7305416, upper bound: 0.6829900
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.08
Output dim: 1, lower bound: -0.7298523, upper bound: 0.6843390
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.08
Output dim: 1, lower bound: -0.7401138, upper bound: 0.6729951
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.08
Output dim: 1, lower bound: -0.7394723, upper bound: 0.6743117
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.08
Output dim: 1, lower bound: -0.7305416, upper bound: 0.6829900
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.08
Output dim: 1, lower bound: -0.7298523, upper bound: 0.6843390
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.08
Output dim: 1, lower bound: -0.7401138, upper bound: 0.6729951
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.08
Output dim: 1, lower bound: -0.7394723, upper bound: 0.6743117
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.08
Output dim: 1, lower bound: -0.7248794, upper bound: 0.6856308
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.08
Output dim: 1, lower bound: -0.7250108, upper bound: 0.6879460
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.08
Output dim: 1, lower bound: -0.7361264, upper bound: 0.6778744
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.08
Output dim: 1, lower bound: -0.7362405, upper bound: 0.6802908
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.08
Output dim: 1, lower bound: -0.7248794, upper bound: 0.6856308
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.08
Output dim: 1, lower bound: -0.7250108, upper bound: 0.6879460
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.08
Output dim: 1, lower bound: -0.7361264, upper bound: 0.6778744
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.08
Output dim: 1, lower bound: -0.7362405, upper bound: 0.6802908
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.08
Output dim: 1, lower bound: -0.7382888, upper bound: 0.6771636
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.08
Output dim: 1, lower bound: -0.7351849, upper bound: 0.6769944
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.08
Output dim: 1, lower bound: -0.7474927, upper bound: 0.6659537
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.08
Output dim: 1, lower bound: -0.7447730, upper bound: 0.6658601
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.08
Output dim: 1, lower bound: -0.7382888, upper bound: 0.6771636
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.08
Output dim: 1, lower bound: -0.7351849, upper bound: 0.6769944
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.08
Output dim: 1, lower bound: -0.7474927, upper bound: 0.6659537
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.08
Output dim: 1, lower bound: -0.7447730, upper bound: 0.6658601
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.08
Output dim: 1, lower bound: -0.7336243, upper bound: 0.6813661
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.08
Output dim: 1, lower bound: -0.7319738, upper bound: 0.6820490
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.08
Output dim: 1, lower bound: -0.7445180, upper bound: 0.6726645
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.08
Output dim: 1, lower bound: -0.7428629, upper bound: 0.6731753
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.08
Output dim: 1, lower bound: -0.7336243, upper bound: 0.6813661
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.08
Output dim: 1, lower bound: -0.7319738, upper bound: 0.6820490
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.08
Output dim: 1, lower bound: -0.7445180, upper bound: 0.6726645
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.08
Output dim: 1, lower bound: -0.7428629, upper bound: 0.6731753
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.08
Output dim: 1, lower bound: -0.6731745, upper bound: 0.7428634
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.08
Output dim: 1, lower bound: -0.6726644, upper bound: 0.7445184
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.08
Output dim: 1, lower bound: -0.6820489, upper bound: 0.7319744
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.08
Output dim: 1, lower bound: -0.6813658, upper bound: 0.7336250
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.08
Output dim: 1, lower bound: -0.6731745, upper bound: 0.7428634
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.08
Output dim: 1, lower bound: -0.6726644, upper bound: 0.7445184
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.08
Output dim: 1, lower bound: -0.6820489, upper bound: 0.7319744
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.08
Output dim: 1, lower bound: -0.6813658, upper bound: 0.7336250
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.08
Output dim: 1, lower bound: -0.6658596, upper bound: 0.7447732
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.08
Output dim: 1, lower bound: -0.6659531, upper bound: 0.7474929
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.08
Output dim: 1, lower bound: -0.6769937, upper bound: 0.7351855
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.08
Output dim: 1, lower bound: -0.6771629, upper bound: 0.7382894
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.08
Output dim: 1, lower bound: -0.6658596, upper bound: 0.7447732
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.08
Output dim: 1, lower bound: -0.6659531, upper bound: 0.7474929
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.08
Output dim: 1, lower bound: -0.6942103, upper bound: 0.7517686
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.08
Output dim: 1, lower bound: -0.6936978, upper bound: 0.7522602
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.08
Output dim: 1, lower bound: -0.7018349, upper bound: 0.7410213
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.08
Output dim: 1, lower bound: -0.6936978, upper bound: 0.7522602
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.08
Output dim: 1, lower bound: -0.7018349, upper bound: 0.7410213
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.08
Output dim: 1, lower bound: -0.6874459, upper bound: 0.7555467
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.08
Output dim: 1, lower bound: -0.6983165, upper bound: 0.7458327
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.08
Output dim: 1, lower bound: -0.6874459, upper bound: 0.7555467
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.08
Output dim: 1, lower bound: -0.6983165, upper bound: 0.7458327
Binary search (step 1): status=Status.UNKNOWN, k_low=4, k_high=7, k_mid=5, eps_mid=0.0195312, abs_max=1.3075802326202393
rel_dist={1: [-0.8569169395709344, 0.8569169395709273]}

## Binary search (step 2) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.39 seconds

### Candidate
type: RSZ, layer: 3, pos: 1725

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7415714, upper bound: 0.6972841
time: 3.77 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6972841, upper bound: 0.7415722
time: 3.70 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 7.89 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 7.89
Output dim: 1, lower bound: -0.7415714, upper bound: 0.6972841
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 7.89
Output dim: 1, lower bound: -0.6972841, upper bound: 0.7415722

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7950673, 1.7942333
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2423437, 1.2359328
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1427157, 1.1494584
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6262693, 1.6268182
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.4890380, 1.5006504
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8264575
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5897963, 1.5894186
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1830754, 1.1847141
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2012539, 1.2062137
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5428200, 1.5582592

Time for backsubstitution: 5.67 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 662

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7307993, upper bound: 0.6942542
time: 3.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7380734, upper bound: 0.6869296
time: 3.60 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7942333, 1.7950673
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2359331, 1.2423439
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1494584, 1.1427157
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6268182, 1.6262693
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5006504, 1.4890380
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8264570, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5894186, 1.5897963
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1847138, 1.1830757
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2062137, 1.2012539
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5582595, 1.5428200

Time for backsubstitution: 5.66 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 662

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6869288, upper bound: 0.7380734
time: 3.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6942544, upper bound: 0.7307999
time: 3.65 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 13.09 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 13.09
Output dim: 1, lower bound: -0.7307993, upper bound: 0.6942542
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 13.09
Output dim: 1, lower bound: -0.7380734, upper bound: 0.6869296
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 13.09
Output dim: 1, lower bound: -0.6869288, upper bound: 0.7380734
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 13.09
Output dim: 1, lower bound: -0.6942544, upper bound: 0.7307999

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7751470, 1.7707181
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2366407, 1.2338467
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1230941, 1.1454778
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6157799, 1.6013727
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.4848299, 1.4991786
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8246412
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5856035, 1.5927308
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1818528, 1.1795843
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.1938596, 1.2008092
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5363970, 1.5358832

Time for backsubstitution: 5.70 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 1928

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7268323, upper bound: 0.6805638
time: 3.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7204401, upper bound: 0.6902267
time: 3.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7715521, 1.7743125
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2402580, 1.2302301
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1387351, 1.1298368
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6008244, 1.6163287
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.4875660, 1.4964426
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8253999
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5931079, 1.5852261
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1779456, 1.1834917
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.1958497, 1.1988194
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5204439, 1.5518363

Time for backsubstitution: 5.69 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 1928

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7338258, upper bound: 0.6752054
time: 3.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7252779, upper bound: 0.6829230
time: 3.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7743130, 1.7715521
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2302301, 1.2402577
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1298368, 1.1387349
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6163287, 1.6008244
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.4964428, 1.4875662
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8253994, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5852258, 1.5931082
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1834912, 1.1779459
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.1988196, 1.1958497
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5518360, 1.5204439

Time for backsubstitution: 5.69 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 1928

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6829222, upper bound: 0.7252782
time: 3.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6752057, upper bound: 0.7338259
time: 3.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7707181, 1.7751470
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2338464, 1.2366409
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1454778, 1.1230941
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6013727, 1.6157804
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.4991789, 1.4848301
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8246412, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5927303, 1.5856037
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1795840, 1.1818533
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2008095, 1.1938598
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5358829, 1.5363970

Time for backsubstitution: 5.67 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 1928

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6902261, upper bound: 0.7204405
time: 3.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6805638, upper bound: 0.7268330
time: 4.10 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 13.62 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.62
Output dim: 1, lower bound: -0.7268323, upper bound: 0.6805638
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.62
Output dim: 1, lower bound: -0.7204401, upper bound: 0.6902267
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.62
Output dim: 1, lower bound: -0.7338258, upper bound: 0.6752054
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.62
Output dim: 1, lower bound: -0.7252779, upper bound: 0.6829230
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.62
Output dim: 1, lower bound: -0.6829222, upper bound: 0.7252782
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.62
Output dim: 1, lower bound: -0.6752057, upper bound: 0.7338259
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.62
Output dim: 1, lower bound: -0.6902261, upper bound: 0.7204405
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.62
Output dim: 1, lower bound: -0.6805638, upper bound: 0.7268330

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.6692128, 1.6498032
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2464375, 1.2312055
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1212752, 1.1437036
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5880418, 1.5753298
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.4653950, 1.4822688
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.7579889, 1.7380409
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5689087, 1.5783458
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1670408, 1.1621394
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2074327, 1.2090104
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5328784, 1.5321870

Time for backsubstitution: 5.76 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 1935

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6871888, upper bound: 0.6400472
time: 3.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6871888, upper bound: 0.6400472
time: 3.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.6542320, 1.6665449
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2339997, 1.2446885
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1213200, 1.1436757
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5900760, 1.5736346
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.4679799, 1.4797440
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.7587490, 1.7370276
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5712180, 1.5760360
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1644087, 1.1662581
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2020607, 1.2154036
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5327010, 1.5326426

Time for backsubstitution: 5.72 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 1935

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6842600, upper bound: 0.6459863
time: 3.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6842600, upper bound: 0.6459863
time: 3.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.6671247, 1.6533976
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2500539, 1.2275889
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1369328, 1.1280627
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5730858, 1.5906243
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.4681311, 1.4795671
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.7569580, 1.7387991
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5764136, 1.5708406
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1646147, 1.1660469
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2104316, 1.2070205
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5171552, 1.5481400

Time for backsubstitution: 5.66 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 3, pos: 1935

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6923314, upper bound: 0.6353526
time: 3.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6923314, upper bound: 0.6353526
time: 3.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.6506371, 1.6686335
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2376165, 1.2410722
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1369607, 1.1280179
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5747809, 1.5885901
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.4706821, 1.4770079
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.7579899, 1.7380581
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5787234, 1.5685313
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1605015, 1.1686842
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2040505, 1.2124045
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5167475, 1.5483656

Time for backsubstitution: 5.68 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 3, pos: 1935

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6900216, upper bound: 0.6403371
time: 3.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6900216, upper bound: 0.6403371
time: 3.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.6686335, 1.6506371
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2410722, 1.2376165
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1280179, 1.1369607
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5885901, 1.5747814
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.4770079, 1.4706821
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.7380581, 1.7579904
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5685310, 1.5787232
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1686840, 1.1605010
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2124047, 1.2040505
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5483656, 1.5167477

Time for backsubstitution: 5.67 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 1935

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6403365, upper bound: 0.6900217
time: 3.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6403365, upper bound: 0.6900217
time: 3.45 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.6533976, 1.6671247
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2275891, 1.2500539
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1280627, 1.1369328
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5906243, 1.5730863
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.4795671, 1.4681311
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.7387991, 1.7569585
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5708408, 1.5764134
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1660471, 1.1646144
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2070203, 1.2104316
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5481400, 1.5171552

Time for backsubstitution: 5.65 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 1935

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6353512, upper bound: 0.6923316
time: 3.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6353512, upper bound: 0.6923316
time: 3.48 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.6665449, 1.6542320
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2446885, 1.2339997
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1436756, 1.1213200
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5736341, 1.5900760
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.4797440, 1.4679799
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.7370281, 1.7587485
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5760360, 1.5712183
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1662579, 1.1644084
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2154036, 1.2020607
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5326424, 1.5327008

Time for backsubstitution: 5.69 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 1935

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6459855, upper bound: 0.6842603
time: 3.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6459855, upper bound: 0.6842603
time: 3.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.6498027, 1.6692133
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2312055, 1.2464375
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1437037, 1.1212752
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5753298, 1.5880418
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.4822688, 1.4653950
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.7380409, 1.7579889
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5783458, 1.5689089
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1621399, 1.1670408
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2090106, 1.2074327
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5321870, 1.5328784

Time for backsubstitution: 5.73 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 1935

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6400457, upper bound: 0.6871894
time: 3.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6400457, upper bound: 0.6871894
time: 3.61 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 13.19 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 13.19
Output dim: 1, lower bound: -0.6871888, upper bound: 0.6400472
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 13.19
Output dim: 1, lower bound: -0.6871888, upper bound: 0.6400472
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 13.19
Output dim: 1, lower bound: -0.6842600, upper bound: 0.6459863
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 13.19
Output dim: 1, lower bound: -0.6842600, upper bound: 0.6459863
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.19
Output dim: 1, lower bound: -0.6923314, upper bound: 0.6353526
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.19
Output dim: 1, lower bound: -0.6923314, upper bound: 0.6353526
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.19
Output dim: 1, lower bound: -0.6900216, upper bound: 0.6403371
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.19
Output dim: 1, lower bound: -0.6900216, upper bound: 0.6403371
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.19
Output dim: 1, lower bound: -0.6403365, upper bound: 0.6900217
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.19
Output dim: 1, lower bound: -0.6403365, upper bound: 0.6900217
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.19
Output dim: 1, lower bound: -0.6353512, upper bound: 0.6923316
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.19
Output dim: 1, lower bound: -0.6353512, upper bound: 0.6923316
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 13.19
Output dim: 1, lower bound: -0.6459855, upper bound: 0.6842603
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 13.19
Output dim: 1, lower bound: -0.6459855, upper bound: 0.6842603
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 13.19
Output dim: 1, lower bound: -0.6400457, upper bound: 0.6871894
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 13.19
Output dim: 1, lower bound: -0.6400457, upper bound: 0.6871894

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.6655374, 1.6530051
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2472754, 1.2256689
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1368654, 1.1281333
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5558434, 1.5836129
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.4642029, 1.4928489
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.7515993, 1.7312789
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5746088, 1.5712576
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1631417, 1.1643167
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.1953025, 1.2018442
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5150094, 1.5436242

Time for backsubstitution: 5.76 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 2901

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6790789, upper bound: 0.6310373
time: 9.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6864944, upper bound: 0.6210311
time: 3.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.6667323, 1.6533976
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2500539, 1.2248104
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1369328, 1.1279953
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5660748, 1.5906243
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.4681311, 1.4756389
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.7494378, 1.7387991
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5764136, 1.5690355
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1646147, 1.1645739
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2052550, 1.2070205
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5171552, 1.5459943

Time for backsubstitution: 5.72 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 2901

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6790789, upper bound: 0.6310373
time: 8.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6864944, upper bound: 0.6210311
time: 3.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.6490498, 1.6682410
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2348380, 1.2391524
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1368933, 1.1281540
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5586128, 1.5815787
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.4667535, 1.4958634
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.7565169, 1.7305379
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5769186, 1.5689542
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1590281, 1.1673675
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.1907277, 1.2072282
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5146017, 1.5438499

Time for backsubstitution: 5.76 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 2901

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6761108, upper bound: 0.6345283
time: 3.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6846960, upper bound: 0.6255352
time: 3.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.6502452, 1.6686335
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2376165, 1.2382936
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1369607, 1.1279504
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5677700, 1.5885901
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.4706821, 1.4730797
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.7504702, 1.7380581
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5787234, 1.5667262
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1605015, 1.1672111
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.1988740, 1.2124045
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5167475, 1.5462198

Time for backsubstitution: 5.70 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 2901

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6761108, upper bound: 0.6345283
time: 3.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6846960, upper bound: 0.6255352
time: 3.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.6670461, 1.6502447
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2382936, 1.2356966
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1279504, 1.1372149
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5743809, 1.5677700
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.4730797, 1.4858999
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.7325001, 1.7504697
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5667267, 1.5791397
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1672115, 1.1614647
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2013845, 1.1988742
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5462198, 1.5170302

Time for backsubstitution: 5.77 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 2901

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6255347, upper bound: 0.6846966
time: 3.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6345280, upper bound: 0.6761111
time: 3.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.6682410, 1.6506371
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2410722, 1.2348378
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1280179, 1.1368933
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5815787, 1.5747814
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.4770079, 1.4667535
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.7305379, 1.7579904
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5685310, 1.5769181
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1686840, 1.1590281
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2072282, 1.2040505
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5483656, 1.5146019

Time for backsubstitution: 5.68 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 2901

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6255347, upper bound: 0.6846966
time: 3.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6345280, upper bound: 0.6761111
time: 3.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.6521182, 1.6667323
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2248101, 1.2481339
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1279953, 1.1371870
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5765605, 1.5660748
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.4756389, 1.4887838
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.7371264, 1.7494378
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5690355, 1.5768304
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1645741, 1.1654840
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.1972594, 1.2052553
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5459943, 1.5174377

Time for backsubstitution: 5.69 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 2901

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6210304, upper bound: 0.6864946
time: 3.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6310371, upper bound: 0.6790794
time: 3.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.6530056, 1.6671247
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2275891, 1.2472754
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1280627, 1.1368654
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5836129, 1.5730863
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.4795671, 1.4642029
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.7312789, 1.7569585
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5708408, 1.5746083
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1660471, 1.1631415
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2018442, 1.2104316
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5481400, 1.5150094

Time for backsubstitution: 5.72 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 2901

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6210304, upper bound: 0.6864946
time: 3.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6310371, upper bound: 0.6790794
time: 3.44 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 12.85 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.85
Output dim: 1, lower bound: -0.6790789, upper bound: 0.6310373
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.85
Output dim: 1, lower bound: -0.6864944, upper bound: 0.6210311
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.85
Output dim: 1, lower bound: -0.6790789, upper bound: 0.6310373
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.85
Output dim: 1, lower bound: -0.6864944, upper bound: 0.6210311
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.85
Output dim: 1, lower bound: -0.6761108, upper bound: 0.6345283
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.85
Output dim: 1, lower bound: -0.6846960, upper bound: 0.6255352
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.85
Output dim: 1, lower bound: -0.6761108, upper bound: 0.6345283
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.85
Output dim: 1, lower bound: -0.6846960, upper bound: 0.6255352
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.85
Output dim: 1, lower bound: -0.6255347, upper bound: 0.6846966
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.85
Output dim: 1, lower bound: -0.6345280, upper bound: 0.6761111
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.85
Output dim: 1, lower bound: -0.6255347, upper bound: 0.6846966
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.85
Output dim: 1, lower bound: -0.6345280, upper bound: 0.6761111
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.85
Output dim: 1, lower bound: -0.6210304, upper bound: 0.6864946
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.85
Output dim: 1, lower bound: -0.6310371, upper bound: 0.6790794
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.85
Output dim: 1, lower bound: -0.6210304, upper bound: 0.6864946
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.85
Output dim: 1, lower bound: -0.6310371, upper bound: 0.6790794
Binary search (step 2): status=Status.VERIFIED, k_low=4, k_high=4, k_mid=4, eps_mid=0.0156250, abs_max=1.2436916828155518
rel_dist={1: [-0.7636376522603388, 0.7636396744031151]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.015625
execution time: 1965.77 seconds
