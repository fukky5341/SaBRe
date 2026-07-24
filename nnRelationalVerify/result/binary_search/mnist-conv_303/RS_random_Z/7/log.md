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
execution time: IAR + LP analysis = 15.28 + 32.61 = 47.89 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3552.11 seconds, max iter: 100)

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
rel_dist={1: [-0.7636376522603388, 0.7636396744031151]}

## Binary Search Result
Binary search time: 153.19 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.01171875


# Relational Split (RS_random_Z) starts
Time budget: 3398.92 seconds

## Binary search (step 0) starts
Candidate k: 8, corresponding eps: 0.0312500


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 1683

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 172

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0825194, upper bound: 1.0754136
time: 4.17 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0754146, upper bound: 1.0825198
time: 3.95 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 8.13 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 8.13
Output dim: 1, lower bound: -1.0825194, upper bound: 1.0754136
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 8.13
Output dim: 1, lower bound: -1.0754146, upper bound: 1.0825198

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.2067490, 2.2382860
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4929333, 1.4913237
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4366674, 1.4310677
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -2.0106559, 2.0096469
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7911050, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9067130, 1.9258606
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4182510, 1.4306822
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4682999, 1.4547992
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.9246042, 1.9284945

Time for backsubstitution: 5.56 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1928

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3104

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0737881, upper bound: 1.0581964
time: 4.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0644067, upper bound: 1.0663704
time: 4.14 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.2382860, 2.2067490
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4913235, 1.4929335
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4310679, 1.4366673
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -2.0096469, 2.0106559
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7911050, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9258609, 1.9067132
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4306822, 1.4182513
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4547992, 1.4682996
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.9284947, 1.9246039

Time for backsubstitution: 5.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 430

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1725

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0207515, upper bound: 0.9659513
time: 3.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9594855, upper bound: 1.0281978
time: 3.93 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 13.17 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 13.17
Output dim: 1, lower bound: -1.0737881, upper bound: 1.0581964
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 13.17
Output dim: 1, lower bound: -1.0644067, upper bound: 1.0663704
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 13.17
Output dim: 1, lower bound: -1.0207515, upper bound: 0.9659513
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 13.17
Output dim: 1, lower bound: -0.9594855, upper bound: 1.0281978

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1751926, 2.2310932
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4906597, 1.4836826
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4372714, 1.4323828
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9856815, 1.9829898
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7170622, 1.7553663
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9054768, 1.9245598
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.3290634, 1.3566618
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4091053, 1.4045427
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8894324, 1.9120154

Time for backsubstitution: 5.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1509

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0728061, upper bound: 1.0428827
time: 4.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0583510, upper bound: 1.0572896
time: 4.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1847265, 2.2215593
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4852924, 1.4890504
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4379823, 1.4316541
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9839988, 1.9846683
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7478502, 1.7116024
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9050658, 1.9246247
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.3442311, 1.3418665
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4184630, 1.3956048
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.9081254, 1.8933225

Time for backsubstitution: 5.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 402

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1977

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0544846, upper bound: 1.0646890
time: 4.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0627256, upper bound: 1.0524937
time: 4.07 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.2382860, 2.2063806
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4899759, 1.4787638
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4155807, 1.4346660
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -2.0096235, 2.0117302
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7624397, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9258165, 1.9059138
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4243827, 1.4152286
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4409418, 1.4643619
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8781559, 1.9051447

Time for backsubstitution: 5.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1509

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0203038, upper bound: 0.9506660
time: 3.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0053735, upper bound: 0.9649710
time: 3.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.2382860, 2.2067490
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4913235, 1.4915855
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4310679, 1.4211806
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -2.0096469, 2.0106335
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7856650, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9258609, 1.9066691
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4276595, 1.4182513
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4508615, 1.4682996
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.9284947, 1.8742654

Time for backsubstitution: 5.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1683

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1509

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9585032, upper bound: 1.0127857
time: 3.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9441685, upper bound: 1.0277711
time: 3.68 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 12.72 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 12.72
Output dim: 1, lower bound: -1.0728061, upper bound: 1.0428827
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 12.72
Output dim: 1, lower bound: -1.0583510, upper bound: 1.0572896
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 12.72
Output dim: 1, lower bound: -1.0544846, upper bound: 1.0646890
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 12.72
Output dim: 1, lower bound: -1.0627256, upper bound: 1.0524937
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 12.72
Output dim: 1, lower bound: -1.0203038, upper bound: 0.9506660
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 12.72
Output dim: 1, lower bound: -1.0053735, upper bound: 0.9649710
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 12.72
Output dim: 1, lower bound: -0.9585032, upper bound: 1.0127857
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 12.72
Output dim: 1, lower bound: -0.9441685, upper bound: 1.0277711

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1645856, 2.2235279
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.5059981, 1.4927006
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.3392801, 1.3563125
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9515486, 1.9591675
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7287464, 1.7643383
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.8904979, 1.9178405
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.3235502, 1.3454089
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4090157, 1.4060369
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8697205, 1.8823938

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 578

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2382

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0664466, upper bound: 1.0323835
time: 4.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0623828, upper bound: 1.0365048
time: 4.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1676269, 2.2204864
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4996781, 1.4990194
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.3612008, 1.3343916
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9618592, 1.9488568
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7260342, 1.7670510
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.8987577, 1.9095809
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.3178101, 1.3511488
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4105997, 1.4044528
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8598108, 1.8923035

Time for backsubstitution: 5.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 430

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2901

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0361567, upper bound: 1.0523662
time: 3.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0526497, upper bound: 1.0409615
time: 4.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1826305, 2.2198825
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4850411, 1.4889112
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4341321, 1.4398761
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9434962, 1.9571834
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7505064, 1.7012448
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.8780739, 1.8784239
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.3120618, 1.3075848
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4128063, 1.3599169
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.9054236, 1.8887491

Time for backsubstitution: 5.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 914

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0513693, upper bound: 1.0610335
time: 4.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0508385, upper bound: 1.0617537
time: 4.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1830497, 2.2215593
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4851532, 1.4890504
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4379823, 1.4278035
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9839988, 1.9441657
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7374926, 1.7116024
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9050658, 1.8976333
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.3442311, 1.3096972
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.3827748, 1.3956048
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.9035516, 1.8933225

Time for backsubstitution: 5.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 402

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 165

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0252220, upper bound: 1.0042190
time: 4.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0242996, upper bound: 1.0043283
time: 3.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.2382860, 2.1988168
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.5053134, 1.4877825
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.3175893, 1.3585954
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9754915, 1.9879084
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7741246, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9108381, 1.8991947
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4188704, 1.4039764
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4408510, 1.4658546
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8584428, 1.8755221

Time for backsubstitution: 5.63 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 578

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2320

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0076570, upper bound: 0.9428726
time: 3.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0131836, upper bound: 0.9406625
time: 3.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.2382860, 2.1957755
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4989944, 1.4941020
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.3395102, 1.3366745
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9858022, 1.9775977
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7714119, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9190974, 1.8909352
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4131303, 1.4097161
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4424345, 1.4642706
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8485332, 1.8854318

Time for backsubstitution: 5.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 423

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2139

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9668907, upper bound: 0.9268198
time: 3.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9668907, upper bound: 0.9268198
time: 3.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.2382860, 2.1991835
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.5066614, 1.5006044
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.3330760, 1.3451098
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9755139, 1.9868116
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7911050, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9108820, 1.8999500
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4221473, 1.4069991
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4507701, 1.4697933
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.9087806, 1.8446431

Time for backsubstitution: 5.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 3104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1501

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9571147, upper bound: 0.9768055
time: 3.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9225769, upper bound: 1.0107378
time: 3.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.2382860, 2.1961422
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.5003428, 1.5069239
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.3549967, 1.3231890
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9858246, 1.9765010
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7911050, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9191418, 1.8916905
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4164071, 1.4127388
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4523542, 1.4682093
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8988709, 1.8545527

Time for backsubstitution: 5.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 578

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2139

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9130412, upper bound: 0.9828119
time: 3.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9130412, upper bound: 0.9828119
time: 3.51 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 12.51 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.51
Output dim: 1, lower bound: -1.0664466, upper bound: 1.0323835
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.51
Output dim: 1, lower bound: -1.0623828, upper bound: 1.0365048
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.51
Output dim: 1, lower bound: -1.0361567, upper bound: 1.0523662
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.51
Output dim: 1, lower bound: -1.0526497, upper bound: 1.0409615
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.51
Output dim: 1, lower bound: -1.0513693, upper bound: 1.0610335
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.51
Output dim: 1, lower bound: -1.0508385, upper bound: 1.0617537
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.51
Output dim: 1, lower bound: -1.0252220, upper bound: 1.0042190
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.51
Output dim: 1, lower bound: -1.0242996, upper bound: 1.0043283
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.51
Output dim: 1, lower bound: -1.0076570, upper bound: 0.9428726
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.51
Output dim: 1, lower bound: -1.0131836, upper bound: 0.9406625
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.51
Output dim: 1, lower bound: -0.9668907, upper bound: 0.9268198
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.51
Output dim: 1, lower bound: -0.9668907, upper bound: 0.9268198
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.51
Output dim: 1, lower bound: -0.9571147, upper bound: 0.9768055
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.51
Output dim: 1, lower bound: -0.9225769, upper bound: 1.0107378
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.51
Output dim: 1, lower bound: -0.9130412, upper bound: 0.9828119
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.51
Output dim: 1, lower bound: -0.9130412, upper bound: 0.9828119

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1638579, 2.2250705
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.5066028, 1.4899082
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.3393273, 1.3561378
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9376512, 1.9587917
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7240415, 1.7515917
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.8773019, 1.9087138
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.3478563, 1.3423023
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4030948, 1.4274578
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8616235, 1.8777375

Time for backsubstitution: 5.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 627

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 655

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0501881, upper bound: 1.0124028
time: 5.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0463379, upper bound: 1.0157109
time: 4.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1645856, 2.2227998
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.5032058, 1.4927006
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.3391054, 1.3563125
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9515486, 1.9452696
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7160001, 1.7643383
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.8904979, 1.9046440
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.3204439, 1.3454089
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4090157, 1.4001160
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8650639, 1.8823938

Time for backsubstitution: 5.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 655

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 914

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0589044, upper bound: 1.0333089
time: 4.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0584148, upper bound: 1.0338972
time: 4.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1650634, 2.2154832
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4947662, 1.4968567
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.3612370, 1.3341818
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9630489, 1.9403949
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7152314, 1.7470543
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.8862913, 1.9136038
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.3213530, 1.3445084
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4099319, 1.4007611
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8424459, 1.8796945

Time for backsubstitution: 5.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 2536

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1935

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9415270, upper bound: 0.9629513
time: 4.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9415270, upper bound: 0.9629513
time: 4.21 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1676269, 2.2179229
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4975157, 1.4990194
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.3609910, 1.3343916
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9533968, 1.9488568
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7260342, 1.7562487
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.8987577, 1.8971152
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.3111696, 1.3511488
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4105997, 1.4037850
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8472018, 1.8923035

Time for backsubstitution: 5.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1977

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2382

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0462013, upper bound: 1.0305106
time: 4.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0421440, upper bound: 1.0345683
time: 4.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1784177, 2.2110858
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4836013, 1.4868550
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4319201, 1.4350154
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9409261, 1.9478908
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7459068, 1.6987739
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.8757427, 1.8796496
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.3091497, 1.3032207
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4069407, 1.3583701
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.9094665, 1.8885729

Time for backsubstitution: 5.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 578

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1269

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0493394, upper bound: 1.0588816
time: 4.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0493394, upper bound: 1.0588816
time: 4.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1826305, 2.2156692
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4829853, 1.4889112
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4292717, 1.4398761
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9434962, 1.9546132
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7480354, 1.7012448
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.8780739, 1.8760924
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.3076973, 1.3075848
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4128063, 1.3540511
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.9052474, 1.8887491

Time for backsubstitution: 5.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 2901

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 423

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0380825, upper bound: 1.0484131
time: 4.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0359321, upper bound: 1.0488560
time: 6.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1720901, 2.2087500
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4827046, 1.4793558
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4318283, 1.4301698
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9832134, 1.9440441
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7287436, 1.7044632
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9015996, 1.8936460
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.3647022, 1.2982278
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.3746307, 1.4067401
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8924127, 1.8517516

Time for backsubstitution: 5.59 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 2460

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1269

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0232727, upper bound: 1.0023784
time: 4.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0233545, upper bound: 1.0022959
time: 4.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1830497, 2.2105999
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4754419, 1.4890504
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4379823, 1.4219937
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9839988, 1.9433808
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7303410, 1.7116024
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9050658, 1.8945541
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.3327618, 1.3096972
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.3827748, 1.3874608
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.9035516, 1.8821831

Time for backsubstitution: 5.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 578

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0229630, upper bound: 1.0041909
time: 4.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0241523, upper bound: 1.0029809
time: 3.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.2382860, 2.2131734
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4843302, 1.4698222
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.3193953, 1.3594325
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.8589125, 1.8316407
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7742009, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9979181, 2.0533552
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4508982, 1.4344623
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4398870, 1.4651377
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.6932335, 1.8035216

Time for backsubstitution: 5.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 423

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9945760, upper bound: 0.9326651
time: 3.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9944719, upper bound: 0.9320580
time: 3.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.2382860, 2.2115414
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4873533, 1.4662004
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.3184264, 1.3604829
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.8192244, 1.8711636
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7757211, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -2.0727592, 1.9862747
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4493561, 1.4400558
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4402285, 1.4648914
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.7893367, 1.7103126

Time for backsubstitution: 5.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 891

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 402

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9891506, upper bound: 0.9322707
time: 3.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0035599, upper bound: 0.9104135
time: 3.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.2382860, 2.1932290
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4974580, 1.5031478
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.3383100, 1.3365705
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9912877, 1.9725461
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7619085, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9271541, 1.8778408
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4084907, 1.4198985
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4399195, 1.4609840
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8411531, 1.9311328

Time for backsubstitution: 5.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1501

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2536

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9623892, upper bound: 0.9199934
time: 3.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9599576, upper bound: 0.9225204
time: 3.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.2382860, 2.1957755
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4989944, 1.4925656
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.3394063, 1.3366745
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9807506, 1.9775977
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7714119, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9060030, 1.8909352
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4131303, 1.4050763
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4391479, 1.4642706
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8485332, 1.8780515

Time for backsubstitution: 5.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 912

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 914

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9447181, upper bound: 0.9061558
time: 3.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9447181, upper bound: 0.9061558
time: 3.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.9449167, 1.9303973
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.5000901, 1.4838071
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2438605, 1.2519295
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9870615, 2.0032406
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.6436136, 1.6203053
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.8815885, 1.8745570
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4078131, 1.3910663
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4072909, 1.4494388
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.9005463, 1.8341801

Time for backsubstitution: 5.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8982190, upper bound: 0.9174163
time: 3.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8982190, upper bound: 0.9174163
time: 3.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.9733548, 1.9046199
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4898639, 1.4955323
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2398956, 1.2570987
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9919424, 1.9982758
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5947869, 1.6753583
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.8854890, 1.8710756
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4077988, 1.3926651
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4304171, 1.4267783
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8983233, 1.8364079

Time for backsubstitution: 5.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 627

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2901

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9030410, upper bound: 0.9988584
time: 3.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9116494, upper bound: 0.9903455
time: 3.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.2367206, 2.1935966
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4988065, 1.5159695
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.3537965, 1.3230851
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9913096, 1.9714494
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7851343, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9271984, 1.8785961
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4117675, 1.4229212
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4498396, 1.4649224
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8914914, 1.9002538

Time for backsubstitution: 5.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 912

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2382

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9066103, upper bound: 0.9695605
time: 4.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9022600, upper bound: 0.9729031
time: 4.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.2382860, 2.1961422
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.5003428, 1.5053873
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.3548927, 1.3231890
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9807725, 1.9765010
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7911050, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9060469, 1.8916905
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4164071, 1.4080989
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4490676, 1.4682093
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8988709, 1.8471725

Time for backsubstitution: 5.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 402

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 627

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9013639, upper bound: 0.9827298
time: 3.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9129748, upper bound: 0.9766552
time: 3.60 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 12.84 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.84
Output dim: 1, lower bound: -1.0501881, upper bound: 1.0124028
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.84
Output dim: 1, lower bound: -1.0463379, upper bound: 1.0157109
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.84
Output dim: 1, lower bound: -1.0589044, upper bound: 1.0333089
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.84
Output dim: 1, lower bound: -1.0584148, upper bound: 1.0338972
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.84
Output dim: 1, lower bound: -0.9415270, upper bound: 0.9629513
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.84
Output dim: 1, lower bound: -0.9415270, upper bound: 0.9629513
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.84
Output dim: 1, lower bound: -1.0462013, upper bound: 1.0305106
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.84
Output dim: 1, lower bound: -1.0421440, upper bound: 1.0345683
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.84
Output dim: 1, lower bound: -1.0493394, upper bound: 1.0588816
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.84
Output dim: 1, lower bound: -1.0493394, upper bound: 1.0588816
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.84
Output dim: 1, lower bound: -1.0380825, upper bound: 1.0484131
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.84
Output dim: 1, lower bound: -1.0359321, upper bound: 1.0488560
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.84
Output dim: 1, lower bound: -1.0232727, upper bound: 1.0023784
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.84
Output dim: 1, lower bound: -1.0233545, upper bound: 1.0022959
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.84
Output dim: 1, lower bound: -1.0229630, upper bound: 1.0041909
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.84
Output dim: 1, lower bound: -1.0241523, upper bound: 1.0029809
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.84
Output dim: 1, lower bound: -0.9945760, upper bound: 0.9326651
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.84
Output dim: 1, lower bound: -0.9944719, upper bound: 0.9320580
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.84
Output dim: 1, lower bound: -0.9891506, upper bound: 0.9322707
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.84
Output dim: 1, lower bound: -1.0035599, upper bound: 0.9104135
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.84
Output dim: 1, lower bound: -0.9623892, upper bound: 0.9199934
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.84
Output dim: 1, lower bound: -0.9599576, upper bound: 0.9225204
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.84
Output dim: 1, lower bound: -0.9447181, upper bound: 0.9061558
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.84
Output dim: 1, lower bound: -0.9447181, upper bound: 0.9061558
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.84
Output dim: 1, lower bound: -0.8982190, upper bound: 0.9174163
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.84
Output dim: 1, lower bound: -0.8982190, upper bound: 0.9174163
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.84
Output dim: 1, lower bound: -0.9030410, upper bound: 0.9988584
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.84
Output dim: 1, lower bound: -0.9116494, upper bound: 0.9903455
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.84
Output dim: 1, lower bound: -0.9066103, upper bound: 0.9695605
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.84
Output dim: 1, lower bound: -0.9022600, upper bound: 0.9729031
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.84
Output dim: 1, lower bound: -0.9013639, upper bound: 0.9827298
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.84
Output dim: 1, lower bound: -0.9129748, upper bound: 0.9766552

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1672387, 2.2382860
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.5076997, 1.4781823
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.3369224, 1.3488737
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9259229, 1.9559522
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7099729, 1.7280695
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.8894038, 1.8988051
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.3157794, 1.3378906
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.3572955, 1.4198401
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8623414, 1.8809993

Time for backsubstitution: 5.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1725

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1935

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9567008, upper bound: 0.9164907
time: 4.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9567008, upper bound: 0.9164907
time: 4.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1733365, 2.2284517
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4948766, 1.4857459
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.3320634, 1.3535924
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9408460, 1.9470634
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7203388, 1.7375228
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.8673930, 1.9155929
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.3359573, 1.3102255
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.3947899, 1.3816586
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8638864, 1.8784556

Time for backsubstitution: 5.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2832

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9737824, upper bound: 0.9585826
time: 4.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9737824, upper bound: 0.9585826
time: 4.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1598206, 2.2134514
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.5020494, 1.4907866
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.3380501, 1.3526083
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9491863, 1.9361854
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7123506, 1.7618673
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.8881671, 1.9058700
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.3193629, 1.3410444
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4027462, 1.3981652
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8692617, 1.8823721

Time for backsubstitution: 5.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 423

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 430

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0166551, upper bound: 0.9949677
time: 4.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0166612, upper bound: 0.9941655
time: 4.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1645856, 2.2180352
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.5012913, 1.4927006
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.3354018, 1.3563125
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9515486, 1.9429073
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7135279, 1.7643383
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.8904979, 1.9023132
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.3160794, 1.3454089
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4090157, 1.3938463
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8650422, 1.8823938

Time for backsubstitution: 5.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2832

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9869745, upper bound: 0.9770839
time: 4.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9869745, upper bound: 0.9770839
time: 4.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1629486, 2.2151427
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4913144, 1.4951210
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.3611398, 1.3344917
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9364963, 1.9338861
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7104173, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.8842847, 1.9154916
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.3196011, 1.3438008
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.3913612, 1.3966041
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8399787, 1.8820848

Time for backsubstitution: 5.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 655

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2832

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8848703, upper bound: 0.8855744
time: 3.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8848703, upper bound: 0.8855744
time: 3.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1647234, 2.2154832
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4947662, 1.4934046
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.3612370, 1.3340844
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9565401, 1.9403949
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7152314, 1.7422400
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.8862913, 1.9115968
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.3213530, 1.3427560
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4057748, 1.4007611
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8424459, 1.8772278

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 1928

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 662

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9357450, upper bound: 0.9585840
time: 4.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9379243, upper bound: 0.9580141
time: 4.19 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1668992, 2.2194655
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4981208, 1.4962268
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.3610384, 1.3342168
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9394994, 1.9484811
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7213287, 1.7435019
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.8855617, 1.8879883
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.3354757, 1.3480422
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4046788, 1.4252064
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8391049, 1.8876472

Time for backsubstitution: 5.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 2320

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 627

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0363595, upper bound: 1.0304368
time: 4.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0461565, upper bound: 1.0264435
time: 4.27 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1676269, 2.2171950
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4947238, 1.4990194
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.3608165, 1.3343916
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9533968, 1.9349594
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7132874, 1.7562487
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.8987577, 1.8839188
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.3080633, 1.3511488
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4105997, 1.3978646
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8425457, 1.8923035

Time for backsubstitution: 5.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 2832

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 165

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0038147, upper bound: 0.9939251
time: 4.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0038147, upper bound: 0.9943062
time: 4.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1784163, 2.2110813
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4836190, 1.4868510
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4319158, 1.4349940
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9409184, 1.9478803
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7459009, 1.6987538
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.8757317, 1.8795884
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.3090446, 1.3032010
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4069407, 1.3583698
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.9094446, 1.8885458

Time for backsubstitution: 5.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 2832

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 655

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0355126, upper bound: 1.0395305
time: 5.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0291040, upper bound: 1.0424517
time: 4.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1784177, 2.2110846
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4835980, 1.4868550
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4319201, 1.4350115
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9409261, 1.9478841
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7458870, 1.6987739
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.8757427, 1.8796387
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.3091304, 1.3032207
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4069407, 1.3583701
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.9094393, 1.8885729

Time for backsubstitution: 5.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 430

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 662

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0316893, upper bound: 1.0552876
time: 4.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0416245, upper bound: 1.0483551
time: 5.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1798587, 2.2120512
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4677761, 1.4736109
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4138179, 1.4188768
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9423933, 1.9543719
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7467704, 1.6995852
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.8515894, 1.8371689
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.3073211, 1.3071971
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4066811, 1.3568513
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8929961, 1.8812509

Time for backsubstitution: 5.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 1250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2382

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0317659, upper bound: 1.0378746
time: 4.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0275592, upper bound: 1.0420916
time: 3.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1826305, 2.2120004
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4829853, 1.4737024
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4082723, 1.4398761
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9434962, 1.9535098
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7480354, 1.6999798
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.8780739, 1.8487754
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.3073096, 1.3075848
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4128063, 1.3460822
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8975260, 1.8887491

Time for backsubstitution: 5.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 1928

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1935

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9360741, upper bound: 0.9578375
time: 4.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9360741, upper bound: 0.9578375
time: 4.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1720881, 2.2087448
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4827225, 1.4793518
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4318244, 1.4301481
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9832058, 1.9440331
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7287362, 1.7044425
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9015887, 1.8935850
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.3645971, 1.2982080
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.3746305, 1.4067394
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8923900, 1.8517244

Time for backsubstitution: 5.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 1935

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 423

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0116250, upper bound: 0.9909379
time: 4.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0115050, upper bound: 0.9910423
time: 4.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1720901, 2.2087479
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4827006, 1.4793558
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4318283, 1.4301658
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9832134, 1.9440370
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7287223, 1.7044632
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9015996, 1.8936353
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.3646820, 1.2982278
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.3746307, 1.4067398
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8923852, 1.8517516

Time for backsubstitution: 5.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1725

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1501

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0222075, upper bound: 0.9663580
time: 3.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9876158, upper bound: 1.0010405
time: 3.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1832652, 2.2104506
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4761140, 1.4899504
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4354997, 1.4193695
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9842172, 1.9439721
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7295251, 1.7112849
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9055707, 1.8953505
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.3318152, 1.3093202
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.3872299, 1.3925366
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.9033349, 1.8821549

Time for backsubstitution: 5.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 914

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 423

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0113753, upper bound: 0.9928102
time: 4.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0104919, upper bound: 0.9929116
time: 4.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.1828990, 2.2108169
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4763424, 1.4897227
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.4353576, 1.4195116
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9845901, 1.9435992
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7300239, 1.7107863
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9058621, 1.8950591
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.3323846, 1.3087509
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.3878512, 1.3919153
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.9035237, 1.8819661

Time for backsubstitution: 5.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 914

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 423

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0125612, upper bound: 0.9907969
time: 4.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0124605, upper bound: 0.9917240
time: 4.14 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.2382860, 2.2099166
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4685478, 1.4539492
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.3051319, 1.3396218
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.8578095, 1.8314009
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7729344, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9719496, 2.0167279
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4505773, 1.4340744
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4336114, 1.4696426
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.6802988, 1.7951169

Time for backsubstitution: 5.59 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 2382

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1977

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9729186, upper bound: 0.9294404
time: 3.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9811971, upper bound: 0.9286561
time: 3.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.2382860, 2.2102225
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4843302, 1.4540398
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2995844, 1.3594325
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.8589125, 1.8305378
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7742009, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9979181, 2.0273867
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4505100, 1.4344623
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4398870, 1.4588616
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.6848288, 1.8035216

Time for backsubstitution: 5.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 3104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9447353, upper bound: 0.8651537
time: 3.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9447353, upper bound: 0.8651537
time: 3.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.2382860, 2.2230473
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4664350, 1.4466643
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.3255813, 1.3684852
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.8149652, 1.8662317
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7793198, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -2.0715342, 1.9855061
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4304867, 1.4337492
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4431107, 1.4636939
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.7380424, 1.6547253

Time for backsubstitution: 5.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 2832

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2139

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9510839, upper bound: 0.8952896
time: 3.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9510839, upper bound: 0.8952896
time: 3.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.2382860, 2.2089894
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4691281, 1.4452827
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.3245783, 1.3676379
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.8142924, 1.8690391
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7760582, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -2.0719905, 1.9837768
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4430494, 1.4224310
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4390309, 1.4655352
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.7337494, 1.6606829

Time for backsubstitution: 5.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 1501

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1465

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0035541, upper bound: 0.9022195
time: 3.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9950694, upper bound: 0.9104073
time: 3.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.2343879, 2.1919181
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4881239, 1.4920223
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.3284984, 1.3298886
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9911251, 1.9714537
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7496738, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9195600, 1.8714437
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.3966761, 1.3991573
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4334264, 1.4559333
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8416328, 1.9273114

Time for backsubstitution: 5.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1928

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1269

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9598993, upper bound: 0.9180092
time: 3.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9599900, upper bound: 0.9180057
time: 3.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.2382860, 2.1892278
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4863324, 1.4999111
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.3401756, 1.3267589
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9901948, 1.9726996
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7468839, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9207568, 1.8755541
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.3877497, 1.4130807
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4395752, 1.4544907
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8373318, 1.9307494

Time for backsubstitution: 5.68 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1977

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1501

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9574901, upper bound: 0.8877933
time: 3.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9298009, upper bound: 0.9214960
time: 3.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.2382860, 2.1869059
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4976978, 1.4906509
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.3384366, 1.3329705
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9783878, 1.9655991
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7660489, 1.7908349
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9036717, 1.8910749
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4094334, 1.4007115
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4328794, 1.4604149
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8527305, 1.8780293

Time for backsubstitution: 5.56 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 627

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 165

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9057611, upper bound: 0.8739020
time: 3.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9057611, upper bound: 0.8739020
time: 3.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -2.2382860, 2.1910100
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.4970798, 1.4925656
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.3357022, 1.3366745
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.9807506, 1.9752355
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.7689404, 1.7911050
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.9060030, 1.8886037
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.4087658, 1.4050763
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.4391479, 1.4580021
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.8485110, 1.8780515

Time for backsubstitution: 5.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 402

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 578

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9414660, upper bound: 0.9060282
time: 3.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9446677, upper bound: 0.8976859
time: 3.84 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 19.40 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 19.40
Output dim: 1, lower bound: -0.9567008, upper bound: 0.9164907
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 19.40
Output dim: 1, lower bound: -0.9567008, upper bound: 0.9164907
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 19.40
Output dim: 1, lower bound: -0.9737824, upper bound: 0.9585826
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 19.40
Output dim: 1, lower bound: -0.9737824, upper bound: 0.9585826
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 19.40
Output dim: 1, lower bound: -1.0166551, upper bound: 0.9949677
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 19.40
Output dim: 1, lower bound: -1.0166612, upper bound: 0.9941655
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 19.40
Output dim: 1, lower bound: -0.9869745, upper bound: 0.9770839
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 19.40
Output dim: 1, lower bound: -0.9869745, upper bound: 0.9770839
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 19.40
Output dim: 1, lower bound: -0.8848703, upper bound: 0.8855744
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 19.40
Output dim: 1, lower bound: -0.8848703, upper bound: 0.8855744
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 19.40
Output dim: 1, lower bound: -0.9357450, upper bound: 0.9585840
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 19.40
Output dim: 1, lower bound: -0.9379243, upper bound: 0.9580141
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 19.40
Output dim: 1, lower bound: -1.0363595, upper bound: 1.0304368
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 19.40
Output dim: 1, lower bound: -1.0461565, upper bound: 1.0264435
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 19.40
Output dim: 1, lower bound: -1.0038147, upper bound: 0.9939251
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 19.40
Output dim: 1, lower bound: -1.0038147, upper bound: 0.9943062
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 19.40
Output dim: 1, lower bound: -1.0355126, upper bound: 1.0395305
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 19.40
Output dim: 1, lower bound: -1.0291040, upper bound: 1.0424517
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 19.40
Output dim: 1, lower bound: -1.0316893, upper bound: 1.0552876
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 19.40
Output dim: 1, lower bound: -1.0416245, upper bound: 1.0483551
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 19.40
Output dim: 1, lower bound: -1.0317659, upper bound: 1.0378746
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 19.40
Output dim: 1, lower bound: -1.0275592, upper bound: 1.0420916
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 19.40
Output dim: 1, lower bound: -0.9360741, upper bound: 0.9578375
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 19.40
Output dim: 1, lower bound: -0.9360741, upper bound: 0.9578375
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 19.40
Output dim: 1, lower bound: -1.0116250, upper bound: 0.9909379
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 19.40
Output dim: 1, lower bound: -1.0115050, upper bound: 0.9910423
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 19.40
Output dim: 1, lower bound: -1.0222075, upper bound: 0.9663580
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 19.40
Output dim: 1, lower bound: -0.9876158, upper bound: 1.0010405
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 19.40
Output dim: 1, lower bound: -1.0113753, upper bound: 0.9928102
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 19.40
Output dim: 1, lower bound: -1.0104919, upper bound: 0.9929116
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 19.40
Output dim: 1, lower bound: -1.0125612, upper bound: 0.9907969
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 19.40
Output dim: 1, lower bound: -1.0124605, upper bound: 0.9917240
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 19.40
Output dim: 1, lower bound: -0.9729186, upper bound: 0.9294404
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 19.40
Output dim: 1, lower bound: -0.9811971, upper bound: 0.9286561
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 19.40
Output dim: 1, lower bound: -0.9447353, upper bound: 0.8651537
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 19.40
Output dim: 1, lower bound: -0.9447353, upper bound: 0.8651537
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 19.40
Output dim: 1, lower bound: -0.9510839, upper bound: 0.8952896
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 19.40
Output dim: 1, lower bound: -0.9510839, upper bound: 0.8952896
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 19.40
Output dim: 1, lower bound: -1.0035541, upper bound: 0.9022195
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 19.40
Output dim: 1, lower bound: -0.9950694, upper bound: 0.9104073
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 19.40
Output dim: 1, lower bound: -0.9598993, upper bound: 0.9180092
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 19.40
Output dim: 1, lower bound: -0.9599900, upper bound: 0.9180057
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 19.40
Output dim: 1, lower bound: -0.9574901, upper bound: 0.8877933
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 19.40
Output dim: 1, lower bound: -0.9298009, upper bound: 0.9214960
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 19.40
Output dim: 1, lower bound: -0.9057611, upper bound: 0.8739020
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 19.40
Output dim: 1, lower bound: -0.9057611, upper bound: 0.8739020
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 19.40
Output dim: 1, lower bound: -0.9414660, upper bound: 0.9060282
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 19.40
Output dim: 1, lower bound: -0.9446677, upper bound: 0.8976859
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.40
Output dim: 1, lower bound: -0.8982190, upper bound: 0.9174163
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.40
Output dim: 1, lower bound: -0.8982190, upper bound: 0.9174163
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.40
Output dim: 1, lower bound: -0.9030410, upper bound: 0.9988584
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.40
Output dim: 1, lower bound: -0.9116494, upper bound: 0.9903455
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.40
Output dim: 1, lower bound: -0.9066103, upper bound: 0.9695605
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.40
Output dim: 1, lower bound: -0.9022600, upper bound: 0.9729031
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.40
Output dim: 1, lower bound: -0.9013639, upper bound: 0.9827298
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.40
Output dim: 1, lower bound: -0.9129748, upper bound: 0.9766552
Binary search (step 0): status=Status.UNKNOWN, k_low=4, k_high=12, k_mid=8, eps_mid=0.0312500, abs_max=1.49924635887146
rel_dist={1: [-1.0866450021547305, 1.0866453172123824]}

## Binary search (step 1) starts
Candidate k: 5, corresponding eps: 0.0195312


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 891

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2320

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8505539, upper bound: 0.8513323
time: 4.59 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8513302, upper bound: 0.8505558
time: 4.28 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 8.88 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 8.88
Output dim: 1, lower bound: -0.8505539, upper bound: 0.8513323
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 8.88
Output dim: 1, lower bound: -0.8513302, upper bound: 0.8505558

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.9199824, 1.9210024
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2875600, 1.2894487
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2293832, 1.2287266
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5905290, 1.5657244
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.6024323, 1.6033826
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.7497954, 1.7916899
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2865517, 1.2830560
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2757695, 1.2759161
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5206246, 1.5806892

Time for backsubstitution: 5.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 891

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1250

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8204169, upper bound: 0.8208405
time: 3.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8204169, upper bound: 0.8208405
time: 3.78 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.9210024, 1.9199824
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2894487, 1.2875597
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2287266, 1.2293832
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5657244, 1.5905290
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.6033826, 1.6024323
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.7916899, 1.7497954
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2830560, 1.2865520
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2759161, 1.2757695
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5806890, 1.5206249

Time for backsubstitution: 5.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 172

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 627

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8484123, upper bound: 0.8504633
time: 5.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8512397, upper bound: 0.8476259
time: 4.67 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 16.15 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 16.15
Output dim: 1, lower bound: -0.8204169, upper bound: 0.8208405
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 16.15
Output dim: 1, lower bound: -0.8204169, upper bound: 0.8208405
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 16.15
Output dim: 1, lower bound: -0.8484123, upper bound: 0.8504633
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 16.15
Output dim: 1, lower bound: -0.8512397, upper bound: 0.8476259

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.9119525, 1.9241252
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2866960, 1.2887075
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2281141, 1.2219782
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5904899, 1.5657024
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.6004543, 1.6026402
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.7651477, 1.7832198
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2803395, 1.3018878
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2748911, 1.2702620
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5200884, 1.5804801

Time for backsubstitution: 5.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 430

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 578

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8193663, upper bound: 0.8207177
time: 3.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8202941, upper bound: 0.8197900
time: 3.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.9199824, 1.9129729
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2875600, 1.2885847
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2226348, 1.2287266
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5905290, 1.5656857
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.6016898, 1.6033826
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.7413254, 1.7916899
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2865517, 1.2768435
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2701154, 1.2759161
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5204155, 1.5806892

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 172

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 423

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8109354, upper bound: 0.8113547
time: 4.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8109319, upper bound: 0.8113591
time: 4.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.9206247, 1.9194813
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2892170, 1.2874453
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2287230, 1.2293425
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5646811, 1.5898125
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.6017170, 1.6010127
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.7911859, 1.7494326
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2822361, 1.2858720
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2758188, 1.2756681
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5810332, 1.5214663

Time for backsubstitution: 5.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 891

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8318373, upper bound: 0.8055293
time: 3.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8034020, upper bound: 0.8337610
time: 4.17 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.9205012, 1.9196043
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2893343, 1.2873285
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2286859, 1.2293798
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5650082, 1.5894852
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.6019630, 1.6007662
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.7913270, 1.7492914
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2823763, 1.2857318
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2758148, 1.2756722
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5815306, 1.5209692

Time for backsubstitution: 5.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 662

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 423

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8417676, upper bound: 0.8368379
time: 4.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8417296, upper bound: 0.8381447
time: 4.76 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 15.21 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.21
Output dim: 1, lower bound: -0.8193663, upper bound: 0.8207177
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.21
Output dim: 1, lower bound: -0.8202941, upper bound: 0.8197900
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.21
Output dim: 1, lower bound: -0.8109354, upper bound: 0.8113547
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.21
Output dim: 1, lower bound: -0.8109319, upper bound: 0.8113591
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.21
Output dim: 1, lower bound: -0.8318373, upper bound: 0.8055293
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.21
Output dim: 1, lower bound: -0.8034020, upper bound: 0.8337610
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.21
Output dim: 1, lower bound: -0.8417676, upper bound: 0.8368379
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.21
Output dim: 1, lower bound: -0.8417296, upper bound: 0.8381447

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.9120345, 1.9239779
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2873688, 1.2895224
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2255788, 1.2193544
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5907078, 1.5661535
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5998254, 1.6023228
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.7656546, 1.7839084
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2793937, 1.3012979
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2793455, 1.2751048
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5198717, 1.5803812

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1501

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 912

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8018619, upper bound: 0.8031463
time: 3.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8018619, upper bound: 0.8031463
time: 3.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.9118056, 1.9242067
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2875109, 1.2893803
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2254901, 1.2194431
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5909410, 1.5659199
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.6001372, 1.6020114
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.7658367, 1.7837267
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2797494, 1.3009422
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2797339, 1.2747164
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5199895, 1.5802631

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1977

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3104

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8138727, upper bound: 0.8072997
time: 3.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8059522, upper bound: 0.8132843
time: 3.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.9172525, 1.9100513
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2718644, 1.2728319
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2050948, 1.2077274
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5894260, 1.5651219
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.6004233, 1.6018724
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.7170792, 1.7607813
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2861645, 1.2764494
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2635272, 1.2760587
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5103316, 1.5734420

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 402

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1509

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8101480, upper bound: 0.8021882
time: 3.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8014573, upper bound: 0.8104683
time: 3.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.9170609, 1.9102426
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2718072, 1.2728891
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2016292, 1.2111933
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5899653, 1.5645826
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.6001797, 1.6021166
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.7104173, 1.7674432
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2861574, 1.2764564
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2702579, 1.2693281
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5131631, 1.5706108

Time for backsubstitution: 5.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 914

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7370210, upper bound: 0.7399604
time: 4.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7366639, upper bound: 0.7405759
time: 4.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.9225559, 1.9194016
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2890291, 1.2868092
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2273481, 1.2290606
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5645723, 1.5900867
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.6015110, 1.6012964
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.7905469, 1.7465434
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2795644, 1.2842340
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2756107, 1.2756686
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5833800, 1.5214028

Time for backsubstitution: 5.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2139

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8125943, upper bound: 0.7877292
time: 4.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8125943, upper bound: 0.7877292
time: 3.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.9205446, 1.9214125
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2885809, 1.2872570
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2284412, 1.2279675
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5649548, 1.5897043
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.6020002, 1.6008067
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.7882967, 1.7487936
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2805982, 1.2832000
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2758191, 1.2754600
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5809700, 1.5238137

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2139

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7299906, upper bound: 0.7599180
time: 3.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7299044, upper bound: 0.7600008
time: 3.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.9177713, 1.9166832
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2736382, 1.2715752
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2111521, 1.2083801
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5639048, 1.5889213
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.6006961, 1.5992556
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.7670813, 1.7183843
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2819898, 1.2853382
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2692273, 1.2758152
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5714507, 1.5137205

Time for backsubstitution: 5.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 1928

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1683

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8059681, upper bound: 0.7981689
time: 3.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8036350, upper bound: 0.8017080
time: 3.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.9175801, 1.9168744
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2735815, 1.2716324
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2076862, 1.2118460
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5644445, 1.5883820
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.6004524, 1.5994992
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.7604189, 1.7250457
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2819822, 1.2853451
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2759581, 1.2690847
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5742817, 1.5108893

Time for backsubstitution: 5.59 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 2382

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1509

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8408561, upper bound: 0.8243528
time: 4.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8322001, upper bound: 0.8376519
time: 4.43 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 14.76 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.76
Output dim: 1, lower bound: -0.8018619, upper bound: 0.8031463
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.76
Output dim: 1, lower bound: -0.8018619, upper bound: 0.8031463
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.76
Output dim: 1, lower bound: -0.8138727, upper bound: 0.8072997
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.76
Output dim: 1, lower bound: -0.8059522, upper bound: 0.8132843
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.76
Output dim: 1, lower bound: -0.8101480, upper bound: 0.8021882
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.76
Output dim: 1, lower bound: -0.8014573, upper bound: 0.8104683
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.76
Output dim: 1, lower bound: -0.7370210, upper bound: 0.7399604
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.76
Output dim: 1, lower bound: -0.7366639, upper bound: 0.7405759
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.76
Output dim: 1, lower bound: -0.8125943, upper bound: 0.7877292
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.76
Output dim: 1, lower bound: -0.8125943, upper bound: 0.7877292
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.76
Output dim: 1, lower bound: -0.7299906, upper bound: 0.7599180
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.76
Output dim: 1, lower bound: -0.7299044, upper bound: 0.7600008
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.76
Output dim: 1, lower bound: -0.8059681, upper bound: 0.7981689
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.76
Output dim: 1, lower bound: -0.8036350, upper bound: 0.8017080
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.76
Output dim: 1, lower bound: -0.8408561, upper bound: 0.8243528
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.76
Output dim: 1, lower bound: -0.8322001, upper bound: 0.8376519

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.9116168, 1.9259810
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2872548, 1.2883074
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2249336, 1.2190886
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5905309, 1.5656023
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5998421, 1.5993326
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.7653079, 1.7819357
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2817287, 1.3009076
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2783532, 1.2782171
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5159266, 1.5784400

Time for backsubstitution: 5.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 662

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1725

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7804127, upper bound: 0.7223601
time: 3.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7195626, upper bound: 0.7822751
time: 3.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.9120345, 1.9235601
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2861533, 1.2895224
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2255788, 1.2187091
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5901566, 1.5661535
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5968342, 1.6023228
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.7656546, 1.7835622
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2790031, 1.3012979
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2793455, 1.2741125
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5179303, 1.5803812

Time for backsubstitution: 5.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 2536

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 891

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7866397, upper bound: 0.7585088
time: 3.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7574495, upper bound: 0.7880176
time: 3.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.8876023, 1.9062510
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2803276, 1.2804587
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2273030, 1.2217003
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5618544, 1.5363798
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5082321, 1.5293496
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.7648430, 1.7827158
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2006102, 1.2305799
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2266300, 1.2271876
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.4675748, 1.5403340

Time for backsubstitution: 5.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 914

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 627

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8105197, upper bound: 0.8072043
time: 3.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8137783, upper bound: 0.8053193
time: 4.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.8940473, 1.9000034
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2785890, 1.2838137
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2277474, 1.2212558
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5614004, 1.5374315
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5275002, 1.5101068
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.7645802, 1.7827330
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2093878, 1.2208567
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2321470, 1.2216125
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.4792578, 1.5278480

Time for backsubstitution: 5.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1509

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8051825, upper bound: 0.8034258
time: 3.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7958198, upper bound: 0.8126859
time: 3.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.9054842, 1.9001844
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2842581, 1.2812767
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1070045, 1.1238211
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5591588, 1.5412984
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.6110911, 1.6108449
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.7026167, 1.7514815
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2785337, 1.2651975
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2638814, 1.2774031
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.4860041, 1.5429208

Time for backsubstitution: 5.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 3104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2832

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7529172, upper bound: 0.7496435
time: 3.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7529172, upper bound: 0.7496435
time: 3.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.9073853, 1.8982835
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2803090, 1.2853708
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1207052, 1.1101205
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5656028, 1.5348542
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.6093955, 1.6125400
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.7077789, 1.7463193
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2749121, 1.2687850
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2648714, 1.2764130
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.4798110, 1.5491145

Time for backsubstitution: 5.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 430

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 578

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8005786, upper bound: 0.8103420
time: 4.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8013372, upper bound: 0.8090817
time: 4.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.9167314, 1.9149988
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2820597, 1.2723374
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2049458, 1.2101619
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5871968, 1.5643451
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.6018343, 1.6016512
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.7098103, 1.7674179
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2854562, 1.2763159
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2699294, 1.2899747
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5145779, 1.5696950

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 891

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1977

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7198614, upper bound: 0.7372020
time: 4.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7339774, upper bound: 0.7278854
time: 4.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.9170609, 1.9099131
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2712555, 1.2728891
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2006602, 1.2111933
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5897279, 1.5645826
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5997143, 1.6021166
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.7103972, 1.7674432
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2860169, 1.2764564
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2702579, 1.2689993
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5122547, 1.5706108

Time for backsubstitution: 5.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1269

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 912

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7243721, upper bound: 0.7244589
time: 4.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7243721, upper bound: 0.7244589
time: 4.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.9147520, 1.9159160
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2871099, 1.2913618
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2264304, 1.2289269
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5654392, 1.5851903
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5951943, 1.6371679
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.7845078, 1.7340951
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2749248, 1.2880783
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2734587, 1.2730341
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5716298, 1.5399508

Time for backsubstitution: 5.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 402

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2832

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7667986, upper bound: 0.7394632
time: 3.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7667986, upper bound: 0.7394632
time: 3.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.9190702, 1.9194016
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2890291, 1.2848902
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2272143, 1.2290606
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5596762, 1.5900867
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.6015110, 1.5949802
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.7780986, 1.7465434
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2795644, 1.2795939
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2729762, 1.2756686
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5833800, 1.5096524

Time for backsubstitution: 5.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 1159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 172

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8078423, upper bound: 0.7787395
time: 4.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8056490, upper bound: 0.7830553
time: 3.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.9202347, 1.9261880
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2986889, 1.2867062
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2316954, 1.2269362
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5621872, 1.5894670
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.6033616, 1.6003408
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.7875776, 1.7487679
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2799156, 1.2830787
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2754471, 1.2960632
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5824614, 1.5229812

Time for backsubstitution: 5.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 655

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2382

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7285832, upper bound: 0.7522549
time: 4.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7223648, upper bound: 0.7584833
time: 3.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.9205446, 1.9211025
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2880306, 1.2872570
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2274098, 1.2279675
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5647173, 1.5897043
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.6015348, 1.6008067
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.7882710, 1.7487936
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2804768, 1.2832000
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2758191, 1.2750881
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5801377, 1.5238137

Time for backsubstitution: 5.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 423

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1465

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7299004, upper bound: 0.7547566
time: 4.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7246742, upper bound: 0.7599969
time: 3.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.9172583, 1.9164503
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2664886, 1.2596612
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2057543, 1.1971638
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5626469, 1.5835299
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5749640, 1.5993755
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.7470107, 1.7120733
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2673752, 1.2466614
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2616313, 1.2742875
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5353365, 1.4868860

Time for backsubstitution: 5.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 2382

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1935

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7259968, upper bound: 0.7178725
time: 3.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7259968, upper bound: 0.7178725
time: 3.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.9175382, 1.9166832
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2617240, 1.2715752
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2111521, 1.2029822
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5639048, 1.5876637
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.6006961, 1.5735233
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.7607703, 1.7183843
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2819898, 1.2707238
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2692273, 1.2682192
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5714507, 1.4776061

Time for backsubstitution: 5.60 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 912

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1928

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7992639, upper bound: 0.7914862
time: 4.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7908458, upper bound: 0.7978278
time: 4.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.9059391, 1.9071345
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2860143, 1.2799716
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1101806, 1.1280414
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5342169, 1.5645983
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.6110377, 1.6083899
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.7457523, 1.7155414
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2743177, 1.2740927
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2763114, 1.2704282
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5499434, 1.4805145

Time for backsubstitution: 5.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 912

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1683

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8050922, upper bound: 0.7864955
time: 3.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8027512, upper bound: 0.7900551
time: 3.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.9078403, 1.9052336
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2819207, 1.2839208
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1238813, 1.1143405
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5406609, 1.5581543
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.6093426, 1.6100850
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.7509146, 1.7103791
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2707300, 1.2777143
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2773015, 1.2694383
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5439072, 1.4867082

Time for backsubstitution: 5.72 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 2139

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1977

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8219012, upper bound: 0.8351313
time: 4.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8297326, upper bound: 0.8224434
time: 4.71 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 15.05 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 1, lower bound: -0.7804127, upper bound: 0.7223601
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 1, lower bound: -0.7195626, upper bound: 0.7822751
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 1, lower bound: -0.7866397, upper bound: 0.7585088
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 1, lower bound: -0.7574495, upper bound: 0.7880176
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 1, lower bound: -0.8105197, upper bound: 0.8072043
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 1, lower bound: -0.8137783, upper bound: 0.8053193
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 1, lower bound: -0.8051825, upper bound: 0.8034258
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 1, lower bound: -0.7958198, upper bound: 0.8126859
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 1, lower bound: -0.7529172, upper bound: 0.7496435
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 1, lower bound: -0.7529172, upper bound: 0.7496435
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 1, lower bound: -0.8005786, upper bound: 0.8103420
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 1, lower bound: -0.8013372, upper bound: 0.8090817
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 1, lower bound: -0.7198614, upper bound: 0.7372020
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 1, lower bound: -0.7339774, upper bound: 0.7278854
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 1, lower bound: -0.7243721, upper bound: 0.7244589
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 1, lower bound: -0.7243721, upper bound: 0.7244589
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 1, lower bound: -0.7667986, upper bound: 0.7394632
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 1, lower bound: -0.7667986, upper bound: 0.7394632
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 1, lower bound: -0.8078423, upper bound: 0.7787395
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 1, lower bound: -0.8056490, upper bound: 0.7830553
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 1, lower bound: -0.7285832, upper bound: 0.7522549
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 1, lower bound: -0.7223648, upper bound: 0.7584833
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 1, lower bound: -0.7299004, upper bound: 0.7547566
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 1, lower bound: -0.7246742, upper bound: 0.7599969
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 1, lower bound: -0.7259968, upper bound: 0.7178725
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 1, lower bound: -0.7259968, upper bound: 0.7178725
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 1, lower bound: -0.7992639, upper bound: 0.7914862
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 1, lower bound: -0.7908458, upper bound: 0.7978278
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 1, lower bound: -0.8050922, upper bound: 0.7864955
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 1, lower bound: -0.8027512, upper bound: 0.7900551
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 1, lower bound: -0.8219012, upper bound: 0.8351313
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 1, lower bound: -0.8297326, upper bound: 0.8224434

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.9122920, 1.9256122
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2859073, 1.2789459
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2093999, 1.2119832
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5904899, 1.5662467
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5625563, 1.5765622
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.7652988, 1.7814851
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2766585, 1.2978849
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2682109, 1.2742746
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.4694788, 1.5534096

Time for backsubstitution: 5.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1269

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1977

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7644445, upper bound: 0.7195783
time: 4.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7729451, upper bound: 0.7189586
time: 3.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.9112482, 1.9266551
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2775183, 1.2869596
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2178283, 1.2035547
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5910726, 1.5655611
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5770712, 1.5620463
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.7648268, 1.7819262
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2787066, 1.2958369
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2744107, 1.2681415
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.4887779, 1.5319920

Time for backsubstitution: 5.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 1501

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1977

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7084760, upper bound: 0.7751140
time: 3.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7155247, upper bound: 0.7703512
time: 3.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.9139657, 1.9234810
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2859659, 1.2888870
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2242041, 1.2184272
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5900478, 1.5664268
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5966287, 1.6026068
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.7650146, 1.7806726
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2763312, 1.2996595
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2791376, 1.2741129
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5202780, 1.5803182

Time for backsubstitution: 5.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 430

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 165

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7507961, upper bound: 0.7213072
time: 3.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7504974, upper bound: 0.7216020
time: 3.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.9119549, 1.9254918
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2855182, 1.2893348
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2252972, 1.2173342
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5904303, 1.5660448
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5971184, 1.6021171
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.7627645, 1.7829227
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2773650, 1.2986257
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2793460, 1.2739043
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5178671, 1.5827281

Time for backsubstitution: 5.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1683

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3104

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7515483, upper bound: 0.7758254
time: 3.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7450081, upper bound: 0.7822917
time: 4.27 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.8872128, 1.9057384
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2800958, 1.2803440
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2272990, 1.2216592
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5608120, 1.5356646
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5066190, 1.5279825
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.7643800, 1.7823939
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1997890, 1.2298996
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2265334, 1.2270870
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.4679632, 1.5412195

Time for backsubstitution: 5.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 912

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 430

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7926717, upper bound: 0.7919433
time: 4.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7973574, upper bound: 0.7888353
time: 4.15 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.8870897, 1.9058616
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2802122, 1.2802272
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2272618, 1.2216964
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5611391, 1.5353372
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5068655, 1.5277362
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.7645211, 1.7822528
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1999292, 1.2297597
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2265294, 1.2270911
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.4684601, 1.5407224

Time for backsubstitution: 5.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 1725

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 912

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7961763, upper bound: 0.7876368
time: 4.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7961763, upper bound: 0.7876368
time: 4.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.8823485, 1.8902061
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2910147, 1.2922320
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1315930, 1.1388015
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5312123, 1.5136874
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5380859, 1.5189972
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.7495174, 1.7728324
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2015879, 1.2096040
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2318575, 1.2223132
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.4558842, 1.4983635

Time for backsubstitution: 5.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 2460

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 912

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7876269, upper bound: 0.7858445
time: 4.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7876269, upper bound: 0.7858445
time: 4.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.8842497, 1.8883052
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2870073, 1.2961810
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1452935, 1.1251013
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5376563, 1.5072434
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5363903, 1.5206926
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.7546792, 1.7676702
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1981347, 1.2131915
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2328477, 1.2213233
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.4497733, 1.5045571

Time for backsubstitution: 5.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 2832

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2536

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7924342, upper bound: 0.8077834
time: 4.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7909187, upper bound: 0.8093012
time: 3.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.9061317, 1.9001727
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2838161, 1.2812583
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1069745, 1.1230042
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5591536, 1.5404806
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.6110907, 1.6109080
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.7028728, 1.7514682
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2733903, 1.2651668
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2638500, 1.2720656
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.4857306, 1.5429201

Time for backsubstitution: 5.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2536

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 914

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7476567, upper bound: 0.7454344
time: 4.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7483777, upper bound: 0.7454269
time: 4.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.9054728, 1.9001844
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2842395, 1.2812767
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1070045, 1.1237910
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5591588, 1.5412941
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.6110911, 1.6108446
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.7026038, 1.7514815
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2785029, 1.2651975
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2638814, 1.2773714
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.4860034, 1.5429208

Time for backsubstitution: 5.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 2901

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 662

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7209377, upper bound: 0.7174935
time: 3.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7209733, upper bound: 0.7174191
time: 3.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.9073591, 1.8980284
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2807877, 1.2860494
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1212926, 1.1106193
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5658998, 1.5353844
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.6086855, 1.6121421
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.7083211, 1.7470427
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2739654, 1.2680604
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2693174, 1.2812476
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.4797239, 1.5490630

Time for backsubstitution: 5.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 655

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7856783, upper bound: 0.7932114
time: 4.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7834323, upper bound: 0.7951301
time: 4.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.9071302, 1.8982573
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2809298, 1.2858493
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1212041, 1.1107080
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5661325, 1.5351512
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.6089969, 1.6118302
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.7085028, 1.7468610
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2743020, 1.2678387
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2697058, 1.2808592
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.4798417, 1.5490274

Time for backsubstitution: 5.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1269

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2536

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7979519, upper bound: 0.8038625
time: 4.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7963594, upper bound: 0.8057042
time: 4.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.9147182, 1.9134581
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2817636, 1.2722237
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2026739, 1.2134726
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5433412, 1.5286162
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5962787, 1.5912924
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6804748, 1.7319231
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2561054, 1.2471862
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2542269, 1.2572286
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5119798, 1.5658109

Time for backsubstitution: 5.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2382

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2832

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 430

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6897647, upper bound: 0.7090692
time: 4.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6902419, upper bound: 0.7090684
time: 4.26 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.9151907, 1.9131961
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2819462, 1.2721539
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2083471, 1.2078903
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5516410, 1.5204878
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5914745, 1.5960975
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6807914, 1.7380824
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2575846, 1.2463694
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2372119, 1.2742260
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5106928, 1.5674610

Time for backsubstitution: 5.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2382

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1501

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7325515, upper bound: 0.7122466
time: 4.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7212138, upper bound: 0.7263402
time: 3.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.9166427, 1.9119163
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2710981, 1.2716744
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2000155, 1.2084913
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5892739, 1.5640314
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5990686, 1.5991263
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.7099857, 1.7654052
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2883530, 1.2760665
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2692664, 1.2721121
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5083094, 1.5686698

Time for backsubstitution: 5.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1501

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 402

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7066707, upper bound: 0.7171612
time: 3.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7177921, upper bound: 0.7059838
time: 3.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.9170609, 1.9094954
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2700400, 1.2728891
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2006602, 1.2105486
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5891762, 1.5645826
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5967231, 1.6021166
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.7103972, 1.7670317
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2856274, 1.2764564
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2702579, 1.2680073
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5103130, 1.5706108

Time for backsubstitution: 5.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 891

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 662

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6767943, upper bound: 0.6749973
time: 4.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6767943, upper bound: 0.6749973
time: 4.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.9153976, 1.9159026
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2865846, 1.2913470
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2264266, 1.2281379
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5654292, 1.5843668
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5951943, 1.6372333
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.7847638, 1.7340827
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2702456, 1.2880471
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2734265, 1.2679665
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5716524, 1.5399504

Time for backsubstitution: 5.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1501

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 655

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7519956, upper bound: 0.7218055
time: 3.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7492402, upper bound: 0.7250337
time: 3.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.9147387, 1.9159160
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2870953, 1.2913618
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2264304, 1.2289231
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5654392, 1.5851800
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5951943, 1.6371675
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.7844954, 1.7340951
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2748928, 1.2880783
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2734587, 1.2730019
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5716290, 1.5399508

Time for backsubstitution: 5.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 1683

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1935

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 2382

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7653502, upper bound: 0.7316282
time: 3.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7589363, upper bound: 0.7379859
time: 4.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.8829050, 1.9113822
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2822847, 1.2772446
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2272043, 1.2254996
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5628123, 1.5924304
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5783253, 1.5651948
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.7521563, 1.7328815
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2581484, 1.2634163
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2604446, 1.2553444
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5642643, 1.5006156

Time for backsubstitution: 5.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 912

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1501

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8062605, upper bound: 0.7584226
time: 4.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7866016, upper bound: 0.7775223
time: 3.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.9118843, 1.8824058
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2812786, 1.2782512
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2237046, 1.2290506
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5621819, 1.5930603
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5749130, 1.5686071
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.7690048, 1.7209148
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2659180, 1.2581789
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2519996, 1.2637823
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5706224, 1.4981842

Time for backsubstitution: 5.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 3104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1977

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7955302, upper bound: 0.7817409
time: 4.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8044178, upper bound: 0.7688068
time: 3.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.9195061, 1.9268785
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2980199, 1.2839139
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2316604, 1.2267623
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5482893, 1.5840201
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5956368, 1.5875940
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.7743783, 1.7381067
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2939332, 1.2799671
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2695267, 1.3072305
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5756545, 1.5183253

Time for backsubstitution: 5.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 2139

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 912

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7112759, upper bound: 0.7407218
time: 3.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7112759, upper bound: 0.7407218
time: 3.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.9202347, 1.9254594
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2958965, 1.2867062
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2315216, 1.2269362
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5621872, 1.5755696
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5906148, 1.6003408
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.7875776, 1.7355685
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2768037, 1.2830787
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2754471, 1.2901428
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5778050, 1.5229812

Time for backsubstitution: 5.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 2536

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2139

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 1269

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7214593, upper bound: 0.7579236
time: 4.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7218413, upper bound: 0.7575448
time: 3.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.9173174, 1.9239550
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2792788, 1.2765079
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2234802, 1.2268478
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5627966, 1.5872350
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5896859, 1.5879533
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.7846112, 1.7512865
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2799385, 1.2798197
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2727230, 1.2756374
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5851078, 1.5277059

Time for backsubstitution: 5.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 172

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 430

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7070421, upper bound: 0.7325017
time: 4.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7079182, upper bound: 0.7198005
time: 3.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.9233975, 1.9178753
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2772818, 1.2785051
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.2262905, 1.2240374
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5622482, 1.5877833
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5886807, 1.5889585
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.7907639, 1.7451334
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2770965, 1.2826617
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2763674, 1.2719929
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5840302, 1.5287836

Time for backsubstitution: 5.54 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=4, k_high=7, k_mid=5, eps_mid=0.0195312, abs_max=1.3075802326202393
rel_dist={1: [-0.8569169395709344, 0.8569169395709273]}

## Binary search (step 2) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 402
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 1928

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 402

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7498615, upper bound: 0.7580006
time: 4.02 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7579985, upper bound: 0.7498615
time: 4.08 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 8.12 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 8.12
Output dim: 1, lower bound: -0.7498615, upper bound: 0.7580006
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 8.12
Output dim: 1, lower bound: -0.7579985, upper bound: 0.7498615

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7911811, 1.7998924
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2240772, 1.2254233
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1675153, 1.1670145
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6188450, 1.6185088
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5248528, 1.5232224
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8104897, 1.8094325
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5895176, 1.5903823
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1715720, 1.1773105
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2099633, 1.2090428
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5542703, 1.5492606

Time for backsubstitution: 5.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 2901

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1465

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7498564, upper bound: 0.7540987
time: 4.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7459484, upper bound: 0.7579976
time: 4.49 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7998924, 1.7911811
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2254233, 1.2240770
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1670146, 1.1675153
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6185088, 1.6188450
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5232224, 1.5248528
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8094330, 1.8104897
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5903821, 1.5895176
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1773107, 1.1715722
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2090428, 1.2099633
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5492606, 1.5542705

Time for backsubstitution: 5.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 2139

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 172

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7531851, upper bound: 0.7417098
time: 4.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7498369, upper bound: 0.7450348
time: 4.37 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 14.53 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 14.53
Output dim: 1, lower bound: -0.7498564, upper bound: 0.7540987
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 14.53
Output dim: 1, lower bound: -0.7459484, upper bound: 0.7579976
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 14.53
Output dim: 1, lower bound: -0.7531851, upper bound: 0.7417098
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 14.53
Output dim: 1, lower bound: -0.7498369, upper bound: 0.7450348

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7879524, 1.8015275
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2149258, 1.2146745
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1635857, 1.1653330
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6169229, 1.6161475
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5128040, 1.5103691
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8048000, 1.8079200
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5858583, 1.5916452
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1704664, 1.1739314
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2068679, 1.2088630
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5590262, 1.5531540

Time for backsubstitution: 5.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1725

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 891

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7357543, upper bound: 0.7178697
time: 3.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7105609, upper bound: 0.7406124
time: 3.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7928162, 1.7966638
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2133281, 1.2162721
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1658340, 1.1630849
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6164842, 1.6165867
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5119991, 1.5111730
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8089771, 1.8037434
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5907807, 1.5867231
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1681929, 1.1762049
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2097833, 1.2059476
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5581641, 1.5540161

Time for backsubstitution: 5.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 627

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1725

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7229723, upper bound: 0.6924214
time: 3.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6791715, upper bound: 0.7357590
time: 3.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7625661, 1.7770383
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2183034, 1.2161522
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1670322, 1.1647332
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6219225, 1.6217542
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.4999681, 1.4988687
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.7891064, 1.7938848
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5475302, 1.5561304
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1583610, 1.1582949
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.1961262, 1.1902964
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5472927, 1.5542481

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 578

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 165

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7451605, upper bound: 0.7294561
time: 4.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7360704, upper bound: 0.7337749
time: 4.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7857494, 1.7538550
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2174985, 1.2169571
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1642318, 1.1675329
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6214175, 1.6222587
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.4972386, 1.5015986
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.7928286, 1.7901640
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5571041, 1.5466657
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1645761, 1.1526225
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.1893761, 1.1970470
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5492382, 1.5523028

Time for backsubstitution: 5.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1683

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7199286, upper bound: 0.7089319
time: 3.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7143049, upper bound: 0.7141293
time: 4.29 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 13.75 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.75
Output dim: 1, lower bound: -0.7357543, upper bound: 0.7178697
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.75
Output dim: 1, lower bound: -0.7105609, upper bound: 0.7406124
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.75
Output dim: 1, lower bound: -0.7229723, upper bound: 0.6924214
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.75
Output dim: 1, lower bound: -0.6791715, upper bound: 0.7357590
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.75
Output dim: 1, lower bound: -0.7451605, upper bound: 0.7294561
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.75
Output dim: 1, lower bound: -0.7360704, upper bound: 0.7337749
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.75
Output dim: 1, lower bound: -0.7199286, upper bound: 0.7089319
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.75
Output dim: 1, lower bound: -0.7143049, upper bound: 0.7141293

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7894821, 1.8014483
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2147377, 1.2141278
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1624014, 1.1650231
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6168146, 1.6163449
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5125985, 1.5105553
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8041172, 1.8068333
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5852194, 1.5892060
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1680007, 1.1722929
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2067013, 1.2088630
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5603337, 1.5524702

Time for backsubstitution: 5.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 2139

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 165

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7249497, upper bound: 0.6973861
time: 3.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7220086, upper bound: 0.7078759
time: 3.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7878733, 1.8016558
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2143795, 1.2144864
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1632757, 1.1641486
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6171203, 1.6160393
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5129900, 1.5101638
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8037138, 1.8072371
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5834193, 1.5910063
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1688285, 1.1714659
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2068682, 1.2086961
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5583420, 1.5543988

Time for backsubstitution: 5.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1935

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 914

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7062121, upper bound: 0.7357648
time: 3.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7058570, upper bound: 0.7361628
time: 3.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7932820, 1.7962952
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2121277, 1.2091484
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1481214, 1.1531692
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6164527, 1.6171036
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.4776158, 1.4884021
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8244600, 1.7992773
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5906992, 1.5862641
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1634865, 1.1731365
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2019482, 1.2018626
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5053916, 1.5166829

Time for backsubstitution: 5.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 2139

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6622528, upper bound: 0.6329974
time: 3.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6617434, upper bound: 0.6335142
time: 3.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7924476, 1.7963014
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2057166, 1.2150717
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1548641, 1.1453722
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6170015, 1.6165552
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.4892287, 1.4767892
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8045111, 1.8192263
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5903220, 1.5866418
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1651244, 1.1714981
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2056983, 1.1969025
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5187993, 1.5012436

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 2382

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2536

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6766383, upper bound: 0.7315535
time: 3.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6746180, upper bound: 0.7327866
time: 3.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7524238, 1.7659707
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2136989, 1.2068994
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1604047, 1.1621850
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6210771, 1.6212406
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.4875002, 1.4914484
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.7672024, 1.7872634
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5440640, 1.5523355
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1640429, 1.1480069
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.1901784, 1.1948807
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5385284, 1.5361800

Time for backsubstitution: 5.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2139

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1269

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7443315, upper bound: 0.7287351
time: 4.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7445926, upper bound: 0.7284965
time: 4.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7625661, 1.7668958
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2090507, 1.2161522
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1670322, 1.1581056
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6219225, 1.6209087
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.4925480, 1.4988687
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.7891064, 1.7719808
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5475302, 1.5526640
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1480727, 1.1582949
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.1961262, 1.1843486
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5472927, 1.5454836

Time for backsubstitution: 5.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 2832

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2382

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7350076, upper bound: 0.7276549
time: 4.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7301600, upper bound: 0.7326770
time: 6.44 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7852850, 1.7535546
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2086508, 1.2049608
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1586888, 1.1572523
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6201935, 1.6177855
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.4722633, 1.4966850
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.7898159, 1.7872767
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5417976, 1.5429640
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1547928, 1.1254895
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.1804388, 1.1922183
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5083876, 1.5171485

Time for backsubstitution: 5.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 1159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6900379, upper bound: 0.6820645
time: 3.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6904121, upper bound: 0.6820647
time: 3.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7854490, 1.7538550
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2055023, 1.2169571
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1642318, 1.1619902
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6214175, 1.6210346
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.4972386, 1.4766240
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.7928286, 1.7871509
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5534024, 1.5466657
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1645761, 1.1428387
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.1893761, 1.1881096
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5492382, 1.5114522

Time for backsubstitution: 5.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1465

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2320

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7076054, upper bound: 0.7099912
time: 4.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7101463, upper bound: 0.7099490
time: 4.05 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 13.72 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.72
Output dim: 1, lower bound: -0.7249497, upper bound: 0.6973861
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.72
Output dim: 1, lower bound: -0.7220086, upper bound: 0.7078759
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.72
Output dim: 1, lower bound: -0.7062121, upper bound: 0.7357648
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.72
Output dim: 1, lower bound: -0.7058570, upper bound: 0.7361628
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 13.72
Output dim: 1, lower bound: -0.6622528, upper bound: 0.6329974
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 13.72
Output dim: 1, lower bound: -0.6617434, upper bound: 0.6335142
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.72
Output dim: 1, lower bound: -0.6766383, upper bound: 0.7315535
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.72
Output dim: 1, lower bound: -0.6746180, upper bound: 0.7327866
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.72
Output dim: 1, lower bound: -0.7443315, upper bound: 0.7287351
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.72
Output dim: 1, lower bound: -0.7445926, upper bound: 0.7284965
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.72
Output dim: 1, lower bound: -0.7350076, upper bound: 0.7276549
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.72
Output dim: 1, lower bound: -0.7301600, upper bound: 0.7326770
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.72
Output dim: 1, lower bound: -0.6900379, upper bound: 0.6820645
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.72
Output dim: 1, lower bound: -0.6904121, upper bound: 0.6820647
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.72
Output dim: 1, lower bound: -0.7076054, upper bound: 0.7099912
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.72
Output dim: 1, lower bound: -0.7101463, upper bound: 0.7099490

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7795906, 1.7864628
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2099442, 1.2054749
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1565907, 1.1635494
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6159682, 1.6159072
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5001287, 1.5031343
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.7830544, 1.8016505
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5826573, 1.5885975
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1726422, 1.1610084
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2002606, 1.2120619
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5479946, 1.5249150

Time for backsubstitution: 5.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 1725

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1935

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6830373, upper bound: 0.6569354
time: 3.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6830373, upper bound: 0.6569354
time: 3.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7894821, 1.7915568
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2060845, 1.2141278
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1624014, 1.1592126
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6168146, 1.6154985
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5051770, 1.5105553
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8041172, 1.7857704
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5852194, 1.5866437
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1567168, 1.1722929
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2067013, 1.2024224
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5603337, 1.5401309

Time for backsubstitution: 5.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 578

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2832

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6850843, upper bound: 0.6714053
time: 3.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6850843, upper bound: 0.6714053
time: 3.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7843204, 1.7960505
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2126317, 1.2124295
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1598842, 1.1594330
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6145506, 1.6096377
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5085926, 1.5068307
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.7994614, 1.7980795
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5804567, 1.5892797
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1653678, 1.1672797
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2006402, 1.2046278
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5602906, 1.5542374

Time for backsubstitution: 5.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 578

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7053272, upper bound: 0.7356850
time: 3.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7060748, upper bound: 0.7350283
time: 3.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7878733, 1.7981024
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2123227, 1.2144864
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1585600, 1.1641486
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6171203, 1.6134696
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5096569, 1.5101638
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8037138, 1.8029842
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5834193, 1.5880439
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1646421, 1.1714659
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2068682, 1.2024684
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5581810, 1.5543988

Time for backsubstitution: 5.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 627

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2382

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7047433, upper bound: 0.7299488
time: 3.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6996953, upper bound: 0.7350112
time: 3.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7853889, 1.7905879
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.1954784, 1.2039378
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1415215, 1.1335945
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6165195, 1.6156087
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.4755945, 1.4617598
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.7819481, 1.7998710
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5838089, 1.5807271
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1443715, 1.1462829
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.1974716, 1.1893973
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5159914, 1.4962845

Time for backsubstitution: 5.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 627

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 662

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6663382, upper bound: 0.7279123
time: 3.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6735233, upper bound: 0.7206484
time: 3.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7867341, 1.7892427
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.1945827, 1.2046900
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1430862, 1.1320297
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6160545, 1.6160736
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.4741988, 1.4631557
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.7851562, 1.7966633
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5844073, 1.5801291
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1399093, 1.1508200
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.1981933, 1.1886761
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5138404, 1.4985108

Time for backsubstitution: 5.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 891

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2320

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6631360, upper bound: 0.7255034
time: 3.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6665356, upper bound: 0.7254296
time: 3.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7524214, 1.7659669
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2137058, 1.2068954
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1604009, 1.1621723
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6210699, 1.6212320
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.4874873, 1.4914281
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.7671609, 1.7872052
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5440531, 1.5522993
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1639800, 1.1479869
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.1901779, 1.1948802
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5385032, 1.5361526

Time for backsubstitution: 5.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 3104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1501

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7432330, upper bound: 0.7118915
time: 4.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7271699, upper bound: 0.7276387
time: 4.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7524195, 1.7659686
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2136948, 1.2069063
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1603918, 1.1621811
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6210680, 1.6212339
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.4874797, 1.4914353
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.7671437, 1.7872219
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5440273, 1.5523243
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1640229, 1.1479442
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.1901779, 1.1948802
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5385013, 1.5361550

Time for backsubstitution: 5.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1928

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7408758, upper bound: 0.7085682
time: 4.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7336919, upper bound: 0.7244791
time: 4.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7618375, 1.7673020
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2079573, 1.2133601
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1669688, 1.1579313
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6080251, 1.6137724
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.4838209, 1.4861224
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.7882767, 1.7682209
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5343337, 1.5415020
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1586714, 1.1551881
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.1902056, 1.1920984
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5409164, 1.5408273

Time for backsubstitution: 5.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 1683

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1977

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7267370, upper bound: 0.7265591
time: 4.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7338842, upper bound: 0.7226205
time: 4.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7625661, 1.7661667
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2062588, 1.2161522
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1668577, 1.1581056
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6219225, 1.6070113
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.4798002, 1.4988687
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.7891064, 1.7711520
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5475302, 1.5394671
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1449652, 1.1582949
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.1961262, 1.1784282
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5426369, 1.5454836

Time for backsubstitution: 5.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1935

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2832

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6946934, upper bound: 0.6957203
time: 4.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6946934, upper bound: 0.6957203
time: 4.44 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7781382, 1.7553296
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2077429, 1.2041512
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1557603, 1.1503718
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6201534, 1.6177592
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.4702511, 1.4957280
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.7888889, 1.7859001
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5396152, 1.5343182
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1487670, 1.1374364
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.1807537, 1.1887128
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5079522, 1.5169749

Time for backsubstitution: 5.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1935

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1501

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6885682, upper bound: 0.6653567
time: 3.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6733040, upper bound: 0.6807922
time: 4.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7852850, 1.7464077
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2086508, 1.2040529
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1518083, 1.1572523
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6201935, 1.6177454
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.4713063, 1.4966850
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.7884388, 1.7872767
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5331521, 1.5429640
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1547928, 1.1194637
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.1769333, 1.1922183
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5082140, 1.5171485

Time for backsubstitution: 5.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 430

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1977

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6635601, upper bound: 0.6614063
time: 4.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6635601, upper bound: 0.6614063
time: 4.14 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7981653, 1.7689610
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.1835845, 1.1978066
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1657047, 1.1628958
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.4878879, 1.4676290
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.4986115, 1.4782953
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8240995
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6368341, 1.6668582
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1999545, 1.1732411
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.1882663, 1.1877413
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.3689101, 1.3755410

Time for backsubstitution: 5.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 2139

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 423

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7001369, upper bound: 0.7025274
time: 4.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7001397, upper bound: 0.7025244
time: 4.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.8006659, 1.7681448
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.1863496, 1.1962960
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1651373, 1.1634212
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.4680438, 1.4885397
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.4993715, 1.4775348
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6742544, 1.6327066
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1991830, 1.1761169
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.1883836, 1.1876183
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.4169614, 1.3334944

Time for backsubstitution: 5.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 914

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7067594, upper bound: 0.7058933
time: 4.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7058562, upper bound: 0.7062812
time: 4.10 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 13.83 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.83
Output dim: 1, lower bound: -0.6830373, upper bound: 0.6569354
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.83
Output dim: 1, lower bound: -0.6830373, upper bound: 0.6569354
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.83
Output dim: 1, lower bound: -0.6850843, upper bound: 0.6714053
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.83
Output dim: 1, lower bound: -0.6850843, upper bound: 0.6714053
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.83
Output dim: 1, lower bound: -0.7053272, upper bound: 0.7356850
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.83
Output dim: 1, lower bound: -0.7060748, upper bound: 0.7350283
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.83
Output dim: 1, lower bound: -0.7047433, upper bound: 0.7299488
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.83
Output dim: 1, lower bound: -0.6996953, upper bound: 0.7350112
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.83
Output dim: 1, lower bound: -0.6663382, upper bound: 0.7279123
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.83
Output dim: 1, lower bound: -0.6735233, upper bound: 0.7206484
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.83
Output dim: 1, lower bound: -0.6631360, upper bound: 0.7255034
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.83
Output dim: 1, lower bound: -0.6665356, upper bound: 0.7254296
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.83
Output dim: 1, lower bound: -0.7432330, upper bound: 0.7118915
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.83
Output dim: 1, lower bound: -0.7271699, upper bound: 0.7276387
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.83
Output dim: 1, lower bound: -0.7408758, upper bound: 0.7085682
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.83
Output dim: 1, lower bound: -0.7336919, upper bound: 0.7244791
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.83
Output dim: 1, lower bound: -0.7267370, upper bound: 0.7265591
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.83
Output dim: 1, lower bound: -0.7338842, upper bound: 0.7226205
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.83
Output dim: 1, lower bound: -0.6946934, upper bound: 0.6957203
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.83
Output dim: 1, lower bound: -0.6946934, upper bound: 0.6957203
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.83
Output dim: 1, lower bound: -0.6885682, upper bound: 0.6653567
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.83
Output dim: 1, lower bound: -0.6733040, upper bound: 0.6807922
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.83
Output dim: 1, lower bound: -0.6635601, upper bound: 0.6614063
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.83
Output dim: 1, lower bound: -0.6635601, upper bound: 0.6614063
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.83
Output dim: 1, lower bound: -0.7001369, upper bound: 0.7025274
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.83
Output dim: 1, lower bound: -0.7001397, upper bound: 0.7025244
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.83
Output dim: 1, lower bound: -0.7067594, upper bound: 0.7058933
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.83
Output dim: 1, lower bound: -0.7058562, upper bound: 0.7062812

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7844853, 1.7960324
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2133045, 1.2132163
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1573322, 1.1568096
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6147685, 1.6100416
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5080261, 1.5065131
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.7963476, 1.7948370
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5809617, 1.5899303
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1644211, 1.1666174
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2050941, 1.2093933
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5600734, 1.5541146

Time for backsubstitution: 5.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2382

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2536

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7025718, upper bound: 0.7310848
time: 3.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7013374, upper bound: 0.7331069
time: 3.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7843022, 1.7962155
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2134194, 1.2131026
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1572609, 1.1568806
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6149545, 1.6098547
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5082755, 1.5062642
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.7962189, 1.7949657
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5811076, 1.5897846
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1647053, 1.1663327
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2054050, 1.2090819
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5601678, 1.5540202

Time for backsubstitution: 5.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 627

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2320

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6978282, upper bound: 0.7267724
time: 3.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6978692, upper bound: 0.7267205
time: 4.10 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7871451, 1.7985091
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2112286, 1.2116935
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1584973, 1.1639745
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6032228, 1.6063313
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5009303, 1.4974170
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8028841, 1.7992215
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5702205, 1.5768759
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1752369, 1.1683545
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2009482, 1.2102177
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5518041, 1.5497432

Time for backsubstitution: 5.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 172

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1683

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6653479, upper bound: 0.6866570
time: 4.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6611652, upper bound: 0.6886363
time: 4.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7878733, 1.7973738
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2095304, 1.2144864
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1583862, 1.1641486
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6171203, 1.5995708
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.4969096, 1.5101638
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8037138, 1.8021541
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5834193, 1.5748451
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1615307, 1.1714659
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2068682, 1.1965473
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5535245, 1.5543988

Time for backsubstitution: 5.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1501

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1977

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6962513, upper bound: 0.7340291
time: 4.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6987963, upper bound: 0.7271500
time: 3.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7654638, 1.7670684
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.1897757, 1.2018516
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1218998, 1.1296136
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6060266, 1.5901599
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.4713864, 1.4602878
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.7808905, 1.7980552
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5796170, 1.5840394
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1427605, 1.1411529
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.1902871, 1.1842027
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5095668, 1.4739070

Time for backsubstitution: 5.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 430

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6310958, upper bound: 0.6888875
time: 3.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6310958, upper bound: 0.6888875
time: 3.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7618690, 1.7706633
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.1933923, 1.1982348
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1375406, 1.1139729
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5910711, 1.6051159
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.4741220, 1.4575517
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.7801323, 1.7988133
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5871215, 1.5765347
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1392415, 1.1450603
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.1922770, 1.1822128
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.4936137, 1.4898603

Time for backsubstitution: 5.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1977

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6320619, upper bound: 0.6880441
time: 3.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6320619, upper bound: 0.6880441
time: 3.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.8008265, 1.8041511
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.1731884, 1.1851072
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1445453, 1.1329637
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.4839196, 1.4631100
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.4755030, 1.4652202
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8272223, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6532602, 1.6824980
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1799049, 1.1879394
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.1975133, 1.1881669
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.3511252, 1.3838472

Time for backsubstitution: 5.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 662

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6399194, upper bound: 0.7033478
time: 3.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6400364, upper bound: 0.7032123
time: 3.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.8016424, 1.8016539
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.1749997, 1.1823418
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1440203, 1.1335721
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.4630914, 1.4829535
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.4762635, 1.4644599
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8221059, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6874371, 1.6489820
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1770287, 1.1907363
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.1976306, 1.1879964
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.3986468, 1.3357956

Time for backsubstitution: 5.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 655

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2901

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6529988, upper bound: 0.7210254
time: 3.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6642895, upper bound: 0.7117715
time: 3.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.4699972, 1.4966764
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2039120, 1.1912391
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.0590560, 1.0582485
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6357007, 1.6383452
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.3137264, 1.2901411
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.6194372, 1.6442380
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5172539, 1.5272412
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1489656, 1.1326091
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.1599050, 1.1759372
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5296204, 1.5261569

Time for backsubstitution: 5.56 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 891

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 627

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7409017, upper bound: 0.7117981
time: 4.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7431397, upper bound: 0.7095673
time: 4.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.4831312, 1.4824576
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.1980495, 1.1971018
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.0564771, 1.0602317
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6381831, 1.6359835
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.2862000, 1.3148792
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.6241937, 1.6397018
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5189948, 1.5252912
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1480496, 1.1329722
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.1712351, 1.1645625
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5285065, 1.5272696

Time for backsubstitution: 5.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 662

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2382

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7261949, upper bound: 0.7216409
time: 4.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7233533, upper bound: 0.7265448
time: 4.22 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.6463566, 1.6457307
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2218363, 1.2037449
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1578355, 1.1599771
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5939746, 1.5961499
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.4719319, 1.4773059
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.6792336, 1.7000527
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5187755, 1.5260394
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1515713, 1.1330900
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.1798439, 1.1777577
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5348127, 1.5320590

Time for backsubstitution: 5.63 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 662

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 891

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7259545, upper bound: 0.6701298
time: 4.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7031185, upper bound: 0.6924402
time: 3.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.6321816, 1.6572397
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2105334, 1.2161827
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1581879, 1.1597788
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5956697, 1.5941401
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.4736099, 1.4758873
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.6799746, 1.7013116
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5210848, 1.5270720
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1491690, 1.1362557
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.1730552, 1.1824086
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5344050, 1.5324829

Time for backsubstitution: 5.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 423

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 914

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7298803, upper bound: 0.7198055
time: 4.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7296423, upper bound: 0.7206157
time: 4.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7604504, 1.7659554
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2077849, 1.2132208
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1635334, 1.1599507
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5657930, 1.5780487
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.4773068, 1.4757633
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.7794480, 1.7587862
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5074306, 1.5053844
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1227674, 1.1171610
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.1637247, 1.1520107
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5373588, 1.5362396

Time for backsubstitution: 5.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 2536

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 655

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7169094, upper bound: 0.7103173
time: 4.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7064827, upper bound: 0.7129948
time: 4.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7604914, 1.7657461
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2078280, 1.2131648
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1695051, 1.1551915
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5722957, 1.5715399
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.4734635, 1.4809093
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.7788420, 1.7601848
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5076842, 1.5149915
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1234207, 1.1192834
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.1501176, 1.1656232
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5363288, 1.5373328

Time for backsubstitution: 5.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1928

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 662

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7204505, upper bound: 0.7197747
time: 4.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7315226, upper bound: 0.7152629
time: 4.48 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7630806, 1.7661536
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2058349, 1.2161374
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1668539, 1.1574737
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6219125, 1.6063499
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.4798007, 1.4989195
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.7890868, 1.7695198
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5477324, 1.5394540
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1412172, 1.1582644
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.1960940, 1.1743674
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5426557, 1.5454829

Time for backsubstitution: 5.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 627

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1935

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6517512, upper bound: 0.6581584
time: 3.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6517512, upper bound: 0.6581584
time: 3.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7625532, 1.7661667
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2062435, 1.2161522
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1668577, 1.1581019
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6219225, 1.6070008
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.4798002, 1.4988685
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.7891064, 1.7711329
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5475173, 1.5394671
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1449351, 1.1582949
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.1961262, 1.1783957
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5426366, 1.5454836

Time for backsubstitution: 5.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1501

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2320

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6874842, upper bound: 0.6882268
time: 4.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6871680, upper bound: 0.6883017
time: 4.22 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7953935, 1.7660360
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.1683764, 1.1825535
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1474793, 1.1418965
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.4867849, 1.4669571
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.4973459, 1.4768345
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8212757, 1.8125124
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6103492, 1.6350436
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1995718, 1.1728530
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.1821415, 1.1870062
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.3591461, 1.3680420

Time for backsubstitution: 5.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1725

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2382

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6990642, upper bound: 0.6966672
time: 4.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6942366, upper bound: 0.7014528
time: 4.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7954187, 1.7661891
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.1683307, 1.1825988
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1447055, 1.1446692
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.4872160, 1.4665256
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.4971485, 1.4770300
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8208609, 1.8129268
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6045461, 1.6403732
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1995661, 1.1728585
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.1875260, 1.1816158
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.3614111, 1.3657770

Time for backsubstitution: 5.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 2901

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 165

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6899818, upper bound: 0.6897083
time: 4.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6802624, upper bound: 0.6948382
time: 4.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7964525, 1.7618794
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.1846020, 1.1942396
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1616437, 1.1585604
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.4654732, 1.4821367
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.4954548, 1.4750638
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8260007, 1.8205338
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6719232, 1.6316104
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1951520, 1.1717520
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.1825182, 1.1831472
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.4188943, 1.3333178

Time for backsubstitution: 5.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 1935

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 430

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6897298, upper bound: 0.6951817
time: 3.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6949046, upper bound: 0.6893476
time: 3.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.8006659, 1.7639315
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.1842933, 1.1962960
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1602764, 1.1634212
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.4680438, 1.4859691
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.4969006, 1.4775348
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8254395
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6742544, 1.6303749
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1948183, 1.1761169
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.1883836, 1.1817527
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.4167848, 1.3334944

Time for backsubstitution: 5.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1465
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 430

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2901

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6930593, upper bound: 0.7037408
time: 4.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7033325, upper bound: 0.6991565
time: 4.18 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 13.78 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.78
Output dim: 1, lower bound: -0.7025718, upper bound: 0.7310848
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.78
Output dim: 1, lower bound: -0.7013374, upper bound: 0.7331069
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.78
Output dim: 1, lower bound: -0.6978282, upper bound: 0.7267724
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.78
Output dim: 1, lower bound: -0.6978692, upper bound: 0.7267205
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 13.78
Output dim: 1, lower bound: -0.6653479, upper bound: 0.6866570
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 13.78
Output dim: 1, lower bound: -0.6611652, upper bound: 0.6886363
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.78
Output dim: 1, lower bound: -0.6962513, upper bound: 0.7340291
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.78
Output dim: 1, lower bound: -0.6987963, upper bound: 0.7271500
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 13.78
Output dim: 1, lower bound: -0.6310958, upper bound: 0.6888875
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 13.78
Output dim: 1, lower bound: -0.6310958, upper bound: 0.6888875
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 13.78
Output dim: 1, lower bound: -0.6320619, upper bound: 0.6880441
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 13.78
Output dim: 1, lower bound: -0.6320619, upper bound: 0.6880441
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.78
Output dim: 1, lower bound: -0.6399194, upper bound: 0.7033478
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.78
Output dim: 1, lower bound: -0.6400364, upper bound: 0.7032123
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.78
Output dim: 1, lower bound: -0.6529988, upper bound: 0.7210254
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.78
Output dim: 1, lower bound: -0.6642895, upper bound: 0.7117715
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.78
Output dim: 1, lower bound: -0.7409017, upper bound: 0.7117981
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.78
Output dim: 1, lower bound: -0.7431397, upper bound: 0.7095673
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.78
Output dim: 1, lower bound: -0.7261949, upper bound: 0.7216409
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.78
Output dim: 1, lower bound: -0.7233533, upper bound: 0.7265448
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.78
Output dim: 1, lower bound: -0.7259545, upper bound: 0.6701298
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.78
Output dim: 1, lower bound: -0.7031185, upper bound: 0.6924402
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.78
Output dim: 1, lower bound: -0.7298803, upper bound: 0.7198055
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.78
Output dim: 1, lower bound: -0.7296423, upper bound: 0.7206157
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.78
Output dim: 1, lower bound: -0.7169094, upper bound: 0.7103173
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.78
Output dim: 1, lower bound: -0.7064827, upper bound: 0.7129948
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.78
Output dim: 1, lower bound: -0.7204505, upper bound: 0.7197747
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.78
Output dim: 1, lower bound: -0.7315226, upper bound: 0.7152629
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 13.78
Output dim: 1, lower bound: -0.6517512, upper bound: 0.6581584
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 13.78
Output dim: 1, lower bound: -0.6517512, upper bound: 0.6581584
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 13.78
Output dim: 1, lower bound: -0.6874842, upper bound: 0.6882268
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 13.78
Output dim: 1, lower bound: -0.6871680, upper bound: 0.6883017
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.78
Output dim: 1, lower bound: -0.6990642, upper bound: 0.6966672
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.78
Output dim: 1, lower bound: -0.6942366, upper bound: 0.7014528
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.78
Output dim: 1, lower bound: -0.6899818, upper bound: 0.6897083
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.78
Output dim: 1, lower bound: -0.6802624, upper bound: 0.6948382
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.78
Output dim: 1, lower bound: -0.6897298, upper bound: 0.6951817
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.78
Output dim: 1, lower bound: -0.6949046, upper bound: 0.6893476
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.78
Output dim: 1, lower bound: -0.6930593, upper bound: 0.7037408
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.78
Output dim: 1, lower bound: -0.7033325, upper bound: 0.6991565

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7774262, 1.7903185
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2029238, 1.2020833
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1439877, 1.1450298
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6142855, 1.6090941
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.4943914, 1.4914827
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.7737832, 1.7754803
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5744500, 1.5840161
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1437435, 1.1414025
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.1968665, 1.2018871
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5573397, 1.5491550

Time for backsubstitution: 5.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 1928

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1683

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6632250, upper bound: 0.6877011
time: 3.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6590565, upper bound: 0.6895415
time: 4.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7787714, 1.7889733
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2021716, 1.2029791
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1455526, 1.1434650
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.6138206, 1.6095591
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.4929957, 1.4928789
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.7769904, 1.7722740
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5750480, 1.5834181
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1392059, 1.1458647
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.1975877, 1.2011657
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5551138, 1.5513058

Time for backsubstitution: 5.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 1977

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 165

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6903827, upper bound: 0.7131196
time: 3.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6875799, upper bound: 0.7228737
time: 3.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7975445, 1.8102741
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.1925838, 1.1937778
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1587203, 1.1578146
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.4829326, 1.4569225
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5095787, 1.5083277
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6499257, 1.6921182
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2046547, 1.2034056
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2047298, 1.2085242
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.3922188, 1.4341221

Time for backsubstitution: 5.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 3104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 430

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6799535, upper bound: 0.7181244
time: 3.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6862207, upper bound: 0.6980904
time: 5.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7983608, 1.8076649
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.1940949, 1.1910126
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1581950, 1.1584232
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.4620218, 1.4767656
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.5103393, 1.5075676
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8290405, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6840773, 1.6586027
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.2017784, 1.2062023
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.2048471, 1.2084069
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.4374068, 1.3860712

Time for backsubstitution: 5.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 1928
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2382
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 1977
type: RSZ, layer: 3, pos: 912

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 430

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6799879, upper bound: 0.7180701
time: 3.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6862626, upper bound: 0.6980384
time: 3.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7863436, 1.7960534
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2093463, 1.2143574
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1561131, 1.1678474
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5748959, 1.5638499
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.4903941, 1.4998040
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.7948856, 1.7927194
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5625358, 1.5542152
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1286695, 1.1392581
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.1863377, 1.1624045
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5500302, 1.5498109

Time for backsubstitution: 5.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 1928

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6329631, upper bound: 0.6706998
time: 3.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6329631, upper bound: 0.6706998
time: 3.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7865534, 1.7960129
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.2094021, 1.2143142
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1605790, 1.1618755
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.5813980, 1.5573473
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.4865503, 1.5036483
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.7942791, 1.7933254
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.5627885, 1.5539618
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1294024, 1.1386049
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.1727250, 1.1760166
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.5489373, 1.5508409

Time for backsubstitution: 5.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1250
type: RSZ, layer: 3, pos: 1159
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 1683
type: RSZ, layer: 3, pos: 1935
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 423
type: RSZ, layer: 3, pos: 2832
type: RSZ, layer: 3, pos: 172
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1509
type: RSZ, layer: 3, pos: 2460
type: RSZ, layer: 3, pos: 1269
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1725
type: RSZ, layer: 3, pos: 662
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 3104
type: RSZ, layer: 3, pos: 1501
type: RSZ, layer: 3, pos: 2320
type: RSZ, layer: 3, pos: 2139
type: RSZ, layer: 3, pos: 578
type: RSZ, layer: 3, pos: 165
type: RSZ, layer: 3, pos: 1928

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6744583, upper bound: 0.7030657
time: 3.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6745486, upper bound: 0.7028605
time: 3.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3412457, -5.1029596, -7.3412457, -5.1029596, -1.7927971, 1.8050437
1: 1.9362400, 3.5890326, 1.9362400, 3.5890326, -1.1723251, 1.1843421
2: -4.9621353, -3.2754836, -4.9621353, -3.2754836, -1.1421800, 1.1262147
3: -11.0800304, -8.8735428, -11.0800304, -8.8735428, -1.4838810, 1.4630849
4: -5.6305523, -3.8394473, -5.6305523, -3.8394473, -1.4737716, 1.4644775
5: -9.0882244, -7.2591839, -9.0882244, -7.2591839, -1.8265924, 1.8290405
6: -6.5653353, -4.2852068, -6.5653353, -4.2852068, -1.6638494, 1.6740289
7: -8.8574305, -7.3770838, -8.8574305, -7.3770838, -1.1736927, 1.2017632
8: 0.9680390, 2.5485387, 0.9680390, 2.5485387, -1.1956799, 1.1825128
9: -9.4929600, -7.3942938, -9.4929600, -7.3942938, -1.3506546, 1.3836381

Time for backsubstitution: 5.53 seconds
Binary search (step 2): status=Status.UNKNOWN, k_low=4, k_high=4, k_mid=4, eps_mid=0.0156250, abs_max=1.2436916828155518
rel_dist={1: [-0.7636396766977613, 0.7636396766977596]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01171875
execution time: 2412.15 seconds
