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
execution time: IAR + LP analysis = 15.63 + 32.31 = 47.94 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3552.06 seconds, max iter: 100)

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
Binary search time: 151.54 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.01171875


# Individual Split (IS_dual_ind) starts
Time budget: 3400.52 seconds

## Binary search (step 0) starts
Candidate k: 8, corresponding eps: 0.0312500


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.36 seconds

### Candidate
type: A, layer: 3, pos: 655

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0703535, upper bound: 1.0702229
time: 4.16 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0703554, upper bound: 1.0703538
time: 3.94 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 8.47 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 8.47
Output dim: 1, lower bound: -1.0703535, upper bound: 1.0702229
IS_A2, status: Status.UNKNOWN, split count: 1, time: 8.47
Output dim: 1, lower bound: -1.0703554, upper bound: 1.0703538

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -7.3354440, -5.1043673, -7.3412457, -5.1029596, -2.2300248, 2.2368784
1: 1.9376824, 3.5761168, 1.9362400, 3.5890326, -1.4959805, 1.4913902
2: -4.9555454, -3.2822270, -4.9621353, -3.2754836, -1.4327376, 1.4282089
3: -11.0604258, -8.8754654, -11.0800304, -8.8735428, -1.9886808, 2.0080056
4: -5.6192431, -3.8418322, -5.6305523, -3.8394473, -1.7797959, 1.7887201
5: -9.0812569, -7.2970009, -9.0882244, -7.2591839, -1.8220730, 1.7912235
6: -6.5684814, -4.3194942, -6.5653353, -4.2852068, -1.9592521, 1.9301138
7: -8.8186340, -7.3928752, -8.8574305, -7.3770838, -1.4011445, 1.4182322
8: 0.9968667, 2.5568495, 0.9680390, 2.5485387, -1.4374094, 1.4707510
9: -9.4914379, -7.3977928, -9.4929600, -7.3942938, -1.9235549, 1.9240007

Time for backsubstitution: 5.70 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 1683
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.36 seconds

### Candidate
type: B, layer: 3, pos: 655

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0664970, upper bound: 1.0664954
time: 3.82 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0664970, upper bound: 1.0664969
time: 4.18 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -7.3381610, -5.1030903, -7.3412457, -5.1029596, -2.2352014, 2.2372522
1: 1.9398844, 3.5777798, 1.9362400, 3.5890326, -1.4961293, 1.4991496
2: -4.9597774, -3.2774539, -4.9621353, -3.2754836, -1.4352298, 1.4331920
3: -11.0695000, -8.8761969, -11.0800304, -8.8735428, -1.9996109, 2.0052533
4: -5.6239738, -3.8399706, -5.6305523, -3.8394473, -1.7845266, 1.7905817
5: -9.0876074, -7.2712379, -9.0882244, -7.2591839, -1.8284235, 1.8169866
6: -6.5642338, -4.3038802, -6.5653353, -4.2852068, -1.9469631, 1.9692283
7: -8.8500109, -7.3786354, -8.8574305, -7.3770838, -1.4117603, 1.4365289
8: 0.9892220, 2.5483055, 0.9680390, 2.5485387, -1.5001907, 1.4696286
9: -9.4917717, -7.3957472, -9.4929600, -7.3942938, -1.9318457, 1.9262047

Time for backsubstitution: 5.78 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 1683
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.38 seconds

### Candidate
type: B, layer: 3, pos: 655

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0664951, upper bound: 1.0703540
time: 4.00 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0664951, upper bound: 1.0703553
time: 4.40 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 14.58 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 14.58
Output dim: 1, lower bound: -1.0664970, upper bound: 1.0664954
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 14.58
Output dim: 1, lower bound: -1.0664970, upper bound: 1.0664969
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 14.58
Output dim: 1, lower bound: -1.0664951, upper bound: 1.0703540
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 14.58
Output dim: 1, lower bound: -1.0664951, upper bound: 1.0703553

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -7.3354440, -5.1043673, -7.3354440, -5.1043673, -2.2242632, 2.2242632
1: 1.9376824, 3.5761168, 1.9376824, 3.5761168, -1.4881248, 1.4881246
2: -4.9555454, -3.2822270, -4.9555454, -3.2822270, -1.4242976, 1.4242975
3: -11.0604258, -8.8754654, -11.0604258, -8.8754654, -1.9894462, 1.9894462
4: -5.6192431, -3.8418322, -5.6192431, -3.8418322, -1.7774110, 1.7774110
5: -9.0812569, -7.2970009, -9.0812569, -7.2970009, -1.7842560, 1.7842560
6: -6.5684814, -4.3194942, -6.5684814, -4.3194942, -1.9400706, 1.9400706
7: -8.8186340, -7.3928752, -8.8186340, -7.3928752, -1.3821764, 1.3821766
8: 0.9968667, 2.5568495, 0.9968667, 2.5568495, -1.4334867, 1.4334865
9: -9.4914379, -7.3977928, -9.4914379, -7.3977928, -1.9190273, 1.9190273

Time for backsubstitution: 5.68 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.36 seconds

### Candidate
type: A, layer: 3, pos: 1683

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0244876, upper bound: 1.0173225
time: 3.69 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0055358, upper bound: 1.0111143
time: 3.52 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -7.3354440, -5.1043673, -7.3381610, -5.1030903, -2.2232108, 2.2249517
1: 1.9376824, 3.5761168, 1.9398844, 3.5777798, -1.4838529, 1.4882734
2: -4.9555454, -3.2822270, -4.9597774, -3.2774539, -1.4303045, 1.4267898
3: -11.0604258, -8.8754654, -11.0695000, -8.8761969, -1.9866939, 1.9970536
4: -5.6192431, -3.8418322, -5.6239738, -3.8399706, -1.7792726, 1.7821417
5: -9.0812569, -7.2970009, -9.0876074, -7.2712379, -1.8100190, 1.7906065
6: -6.5684814, -4.3194942, -6.5642338, -4.3038802, -1.9294055, 1.9277813
7: -8.8186340, -7.3928752, -8.8500109, -7.3786354, -1.4004736, 1.4107983
8: 0.9968667, 2.5568495, 0.9892220, 2.5483055, -1.4323640, 1.4505243
9: -9.4914379, -7.3977928, -9.4917717, -7.3957472, -1.9212310, 1.9201550

Time for backsubstitution: 5.67 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.36 seconds

### Candidate
type: A, layer: 3, pos: 1683

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0244876, upper bound: 1.0173222
time: 3.86 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0055358, upper bound: 1.0111142
time: 3.93 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -7.3381610, -5.1030903, -7.3354440, -5.1043673, -2.2249517, 2.2232111
1: 1.9398844, 3.5777798, 1.9376824, 3.5761168, -1.4882736, 1.4838529
2: -4.9597774, -3.2774539, -4.9555454, -3.2822270, -1.4267898, 1.4303045
3: -11.0695000, -8.8761969, -11.0604258, -8.8754654, -1.9970536, 1.9866939
4: -5.6239738, -3.8399706, -5.6192431, -3.8418322, -1.7821417, 1.7792726
5: -9.0876074, -7.2712379, -9.0812569, -7.2970009, -1.7906065, 1.8100190
6: -6.5642338, -4.3038802, -6.5684814, -4.3194942, -1.9277816, 1.9294052
7: -8.8500109, -7.3786354, -8.8186340, -7.3928752, -1.4107981, 1.4004736
8: 0.9892220, 2.5483055, 0.9968667, 2.5568495, -1.4505241, 1.4323640
9: -9.4917717, -7.3957472, -9.4914379, -7.3977928, -1.9201555, 1.9212313

Time for backsubstitution: 5.70 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.37 seconds

### Candidate
type: A, layer: 3, pos: 1683

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0244876, upper bound: 1.0189590
time: 3.77 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0055358, upper bound: 1.0098586
time: 3.46 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -7.3381610, -5.1030903, -7.3381610, -5.1030903, -2.2350707, 2.2350707
1: 1.9398844, 3.5777798, 1.9398844, 3.5777798, -1.4959602, 1.4959602
2: -4.9597774, -3.2774539, -4.9597774, -3.2774539, -1.4317729, 1.4317729
3: -11.0695000, -8.8761969, -11.0695000, -8.8761969, -1.9975333, 1.9975333
4: -5.6239738, -3.8399706, -5.6239738, -3.8399706, -1.7840033, 1.7840033
5: -9.0876074, -7.2712379, -9.0876074, -7.2712379, -1.8163695, 1.8163695
6: -6.5642338, -4.3038802, -6.5642338, -4.3038802, -1.9678531, 1.9678531
7: -8.8500109, -7.3786354, -8.8500109, -7.3786354, -1.4110789, 1.4110789
8: 0.9892220, 2.5483055, 0.9892220, 2.5483055, -1.4989686, 1.4989686
9: -9.4917717, -7.3957472, -9.4917717, -7.3957472, -1.9295235, 1.9295235

Time for backsubstitution: 5.80 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.39 seconds

### Candidate
type: A, layer: 3, pos: 1683

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0244876, upper bound: 1.0189589
time: 4.12 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0055358, upper bound: 1.0098584
time: 3.70 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 14.01 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 14.01
Output dim: 1, lower bound: -1.0244876, upper bound: 1.0173225
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 14.01
Output dim: 1, lower bound: -1.0055358, upper bound: 1.0111143
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 14.01
Output dim: 1, lower bound: -1.0244876, upper bound: 1.0173222
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 14.01
Output dim: 1, lower bound: -1.0055358, upper bound: 1.0111142
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 14.01
Output dim: 1, lower bound: -1.0244876, upper bound: 1.0189590
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 14.01
Output dim: 1, lower bound: -1.0055358, upper bound: 1.0098586
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 14.01
Output dim: 1, lower bound: -1.0244876, upper bound: 1.0189589
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 14.01
Output dim: 1, lower bound: -1.0055358, upper bound: 1.0098584

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -7.3346786, -5.1047831, -7.3354440, -5.1043673, -2.2235909, 2.2235539
1: 1.9385400, 3.5586948, 1.9376824, 3.5761168, -1.4873376, 1.4694548
2: -4.9532833, -3.2837548, -4.9555454, -3.2822270, -1.4148276, 1.4232247
3: -11.0594978, -8.8771133, -11.0604258, -8.8754654, -1.9873776, 1.9880276
4: -5.5946770, -3.8422813, -5.6192431, -3.8418322, -1.7528448, 1.7769618
5: -9.0767145, -7.2972412, -9.0812569, -7.2970009, -1.7797136, 1.7840157
6: -6.5674729, -4.3215342, -6.5684814, -4.3194942, -1.9391499, 1.9347718
7: -8.8161144, -7.3933764, -8.8186340, -7.3928752, -1.3712883, 1.3810790
8: 1.0123816, 2.5559411, 0.9968667, 2.5568495, -1.4225509, 1.4325597
9: -9.4507437, -7.3981647, -9.4914379, -7.3977928, -1.8691745, 1.9186635

Time for backsubstitution: 5.75 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 1683
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.38 seconds

### Candidate
type: B, layer: 3, pos: 1683

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0111142, upper bound: 1.0111144
time: 3.61 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0111142, upper bound: 1.0111144
time: 3.61 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -7.3365030, -5.1072540, -7.3351502, -5.1046953, -2.2258062, 2.2224789
1: 1.8596368, 3.5449114, 1.9380367, 3.5721278, -1.5713336, 1.4738665
2: -4.9189172, -3.2766373, -4.9506903, -3.2824183, -1.4074819, 1.4661815
3: -11.0327139, -8.8772326, -11.0572481, -8.8758650, -1.9711404, 1.9920902
4: -5.5671411, -3.7495725, -5.6118612, -3.8419209, -1.7252202, 1.8622887
5: -9.0803432, -7.2987070, -9.0802040, -7.2972078, -1.7831354, 1.7814970
6: -6.5590625, -4.4093394, -6.5680175, -4.3300614, -1.9731278, 1.8914702
7: -8.7222109, -7.3930678, -8.8078194, -7.3930082, -1.3292027, 1.4147515
8: 1.0360909, 2.6289511, 1.0016818, 2.5567007, -1.4260550, 1.4880857
9: -9.3759394, -7.2045193, -9.4768791, -7.3978295, -1.8812809, 2.1488216

Time for backsubstitution: 5.72 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 1683
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.37 seconds

### Candidate
type: B, layer: 3, pos: 1683

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0111142, upper bound: 1.0111143
time: 3.50 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0111142, upper bound: 1.0111143
time: 3.61 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -7.3346786, -5.1047831, -7.3381610, -5.1030903, -2.2225389, 2.2242427
1: 1.9385400, 3.5586948, 1.9398844, 3.5777798, -1.4830656, 1.4696035
2: -4.9532833, -3.2837548, -4.9597774, -3.2774539, -1.4208345, 1.4257169
3: -11.0594978, -8.8771133, -11.0695000, -8.8761969, -1.9846253, 1.9956355
4: -5.5946770, -3.8422813, -5.6239738, -3.8399706, -1.7547064, 1.7816925
5: -9.0767145, -7.2972412, -9.0876074, -7.2712379, -1.8054767, 1.7903662
6: -6.5674729, -4.3215342, -6.5642338, -4.3038802, -1.9284842, 1.9224825
7: -8.8161144, -7.3933764, -8.8500109, -7.3786354, -1.3895855, 1.4097006
8: 1.0123816, 2.5559411, 0.9892220, 2.5483055, -1.4214287, 1.4495978
9: -9.4507437, -7.3981647, -9.4917717, -7.3957472, -1.8713782, 1.9197912

Time for backsubstitution: 5.76 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 1683
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.37 seconds

### Candidate
type: B, layer: 3, pos: 1683

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0098588, upper bound: 1.0111144
time: 3.56 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0098588, upper bound: 1.0111144
time: 3.56 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -7.3365030, -5.1072540, -7.3378658, -5.1034174, -2.2247553, 2.2231662
1: 1.8596368, 3.5449114, 1.9402292, 3.5733802, -1.5668609, 1.4740100
2: -4.9189172, -3.2766373, -4.9549160, -3.2776437, -1.4134922, 1.4684277
3: -11.0327139, -8.8772326, -11.0662403, -8.8766050, -1.9683495, 1.9996991
4: -5.5671411, -3.7495725, -5.6165714, -3.8400559, -1.7270851, 1.8669989
5: -9.0803432, -7.2987070, -9.0865917, -7.2714453, -1.8088980, 1.7878847
6: -6.5590625, -4.4093394, -6.5637803, -4.3144469, -1.9619627, 1.8791795
7: -8.7222109, -7.3930678, -8.8388205, -7.3787727, -1.3434381, 1.4457526
8: 1.0360909, 2.6289511, 0.9944224, 2.5481591, -1.4249375, 1.5046077
9: -9.3759394, -7.2045193, -9.4771404, -7.3957820, -1.8834844, 2.1499496

Time for backsubstitution: 5.69 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 1683
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.36 seconds

### Candidate
type: B, layer: 3, pos: 1683

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0098587, upper bound: 1.0111143
time: 3.53 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0098587, upper bound: 1.0111143
time: 3.57 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -7.3373909, -5.1035042, -7.3354440, -5.1043673, -2.2242765, 2.2225025
1: 1.9407153, 3.5610929, 1.9376824, 3.5761168, -1.4874687, 1.4656041
2: -4.9575157, -3.2789927, -4.9555454, -3.2822270, -1.4176958, 1.4292104
3: -11.0687046, -8.8778534, -11.0604258, -8.8754654, -1.9949870, 1.9852471
4: -5.5993242, -3.8404155, -5.6192431, -3.8418322, -1.7574921, 1.7788277
5: -9.0829248, -7.2714796, -9.0812569, -7.2970009, -1.7859240, 1.8097773
6: -6.5632372, -4.3059192, -6.5684814, -4.3194942, -1.9268613, 1.9247372
7: -8.8481512, -7.3791618, -8.8186340, -7.3928752, -1.4006314, 1.3992805
8: 1.0040431, 2.5473943, 0.9968667, 2.5568495, -1.4401081, 1.4314883
9: -9.4507761, -7.3961210, -9.4914379, -7.3977928, -1.8704247, 1.9208570

Time for backsubstitution: 5.77 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 1683
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.38 seconds

### Candidate
type: B, layer: 3, pos: 1683

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0111142, upper bound: 1.0098587
time: 3.49 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0111142, upper bound: 1.0098587
time: 3.73 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -7.3392220, -5.1059752, -7.3351502, -5.1046953, -2.2264833, 2.2214270
1: 1.8664083, 3.5426223, 1.9380367, 3.5721278, -1.5711744, 1.4700277
2: -4.9221191, -3.2719152, -4.9506903, -3.2824183, -1.4091110, 1.4723442
3: -11.0405436, -8.8780537, -11.0572481, -8.8758650, -1.9788494, 1.9896679
4: -5.5685873, -3.7483337, -5.6118612, -3.8419209, -1.7266665, 1.8635275
5: -9.0868692, -7.2729349, -9.0802040, -7.2972078, -1.7896614, 1.8072691
6: -6.5583653, -4.3931675, -6.5680175, -4.3300614, -1.9590511, 1.8797326
7: -8.7508974, -7.3796463, -8.8078194, -7.3930082, -1.3578892, 1.4281731
8: 1.0328178, 2.6160669, 1.0016818, 2.5567007, -1.4434090, 1.4909968
9: -9.3758202, -7.2023268, -9.4768791, -7.3978295, -1.8824272, 2.1512725

Time for backsubstitution: 5.73 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 1683
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.38 seconds

### Candidate
type: B, layer: 3, pos: 1683

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0111142, upper bound: 1.0098587
time: 3.82 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0111142, upper bound: 1.0098587
time: 3.74 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -7.3373909, -5.1035042, -7.3381610, -5.1030903, -2.2343006, 2.2346568
1: 1.9407153, 3.5610929, 1.9398844, 3.5777798, -1.4951649, 1.4772909
2: -4.9575157, -3.2789927, -4.9597774, -3.2774539, -1.4223068, 1.4306705
3: -11.0687046, -8.8778534, -11.0695000, -8.8761969, -1.9954653, 1.9961181
4: -5.5993242, -3.8404155, -5.6239738, -3.8399706, -1.7593536, 1.7835584
5: -9.0829248, -7.2714796, -9.0876074, -7.2712379, -1.8116870, 1.8161278
6: -6.5632372, -4.3059192, -6.5642338, -4.3038802, -1.9669323, 1.9624522
7: -8.8481512, -7.3791618, -8.8500109, -7.3786354, -1.4001918, 1.4099684
8: 1.0040431, 2.5473943, 0.9892220, 2.5483055, -1.4877882, 1.4980345
9: -9.4507761, -7.3961210, -9.4917717, -7.3957472, -1.8800073, 1.9291494

Time for backsubstitution: 5.73 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 1683
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.40 seconds

### Candidate
type: B, layer: 3, pos: 1683

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0063874, upper bound: 1.0098589
time: 3.47 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0063874, upper bound: 1.0098589
time: 3.47 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -7.3392220, -5.1059752, -7.3378658, -5.1034174, -2.2358046, 2.2318907
1: 1.8664083, 3.5426223, 1.9402292, 3.5733802, -1.5789804, 1.4815953
2: -4.9221191, -3.2719152, -4.9549160, -3.2776437, -1.4150314, 1.4735003
3: -11.0405436, -8.8780537, -11.0662403, -8.8766050, -1.9790983, 2.0001836
4: -5.5685873, -3.7483337, -5.6165714, -3.8400559, -1.7285314, 1.8682377
5: -9.0868692, -7.2729349, -9.0865917, -7.2714453, -1.8154240, 1.8136568
6: -6.5583653, -4.3931675, -6.5637803, -4.3144469, -2.0015545, 1.9215364
7: -8.7508974, -7.3796463, -8.8388205, -7.3787727, -1.3716843, 1.4591742
8: 1.0328178, 2.6160669, 0.9944224, 2.5481591, -1.4916906, 1.5540106
9: -9.3758202, -7.2023268, -9.4771404, -7.3957820, -1.8881407, 2.1595907

Time for backsubstitution: 5.72 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 1683
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.39 seconds

### Candidate
type: B, layer: 3, pos: 1683

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0063874, upper bound: 1.0098589
time: 3.60 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0063874, upper bound: 1.0098589
time: 3.53 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 13.25 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 13.25
Output dim: 1, lower bound: -1.0111142, upper bound: 1.0111144
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 13.25
Output dim: 1, lower bound: -1.0111142, upper bound: 1.0111144
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 13.25
Output dim: 1, lower bound: -1.0111142, upper bound: 1.0111143
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 13.25
Output dim: 1, lower bound: -1.0111142, upper bound: 1.0111143
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 13.25
Output dim: 1, lower bound: -1.0098588, upper bound: 1.0111144
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 13.25
Output dim: 1, lower bound: -1.0098588, upper bound: 1.0111144
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 13.25
Output dim: 1, lower bound: -1.0098587, upper bound: 1.0111143
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 13.25
Output dim: 1, lower bound: -1.0098587, upper bound: 1.0111143
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 13.25
Output dim: 1, lower bound: -1.0111142, upper bound: 1.0098587
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 13.25
Output dim: 1, lower bound: -1.0111142, upper bound: 1.0098587
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 13.25
Output dim: 1, lower bound: -1.0111142, upper bound: 1.0098587
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 13.25
Output dim: 1, lower bound: -1.0111142, upper bound: 1.0098587
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 13.25
Output dim: 1, lower bound: -1.0063874, upper bound: 1.0098589
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 13.25
Output dim: 1, lower bound: -1.0063874, upper bound: 1.0098589
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 13.25
Output dim: 1, lower bound: -1.0063874, upper bound: 1.0098589
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 13.25
Output dim: 1, lower bound: -1.0063874, upper bound: 1.0098589

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -7.3346786, -5.1047831, -7.3346786, -5.1047831, -2.2228818, 2.2228816
1: 1.9385400, 3.5586948, 1.9385400, 3.5586948, -1.4686680, 1.4686677
2: -4.9532833, -3.2837548, -4.9532833, -3.2837548, -1.4137547, 1.4137547
3: -11.0594978, -8.8771133, -11.0594978, -8.8771133, -1.9859595, 1.9859595
4: -5.5946770, -3.8422813, -5.5946770, -3.8422813, -1.7523956, 1.7523956
5: -9.0767145, -7.2972412, -9.0767145, -7.2972412, -1.7794733, 1.7794733
6: -6.5674729, -4.3215342, -6.5674729, -4.3215342, -1.9338508, 1.9338508
7: -8.8161144, -7.3933764, -8.8161144, -7.3933764, -1.3701906, 1.3701904
8: 1.0123816, 2.5559411, 1.0123816, 2.5559411, -1.4216247, 1.4216244
9: -9.4507437, -7.3981647, -9.4507437, -7.3981647, -1.8688104, 1.8688107

Time for backsubstitution: 5.70 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.39 seconds

### Candidate
type: A, layer: 3, pos: 402

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0004808, upper bound: 1.0063425
time: 3.76 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0199424, upper bound: 1.0063427
time: 3.76 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -7.3346786, -5.1047831, -7.3365030, -5.1072540, -2.2222209, 2.2252545
1: 1.9385400, 3.5586948, 1.8596368, 3.5449114, -1.4615650, 1.5561008
2: -4.9532833, -3.2837548, -4.9189172, -3.2766373, -1.4575195, 1.4185314
3: -11.0594978, -8.8771133, -11.0327139, -8.8772326, -1.9924827, 1.9676447
4: -5.5946770, -3.8422813, -5.5671411, -3.7495725, -1.8451045, 1.7248597
5: -9.0767145, -7.2972412, -9.0803432, -7.2987070, -1.7780075, 1.7831020
6: -6.5674729, -4.3215342, -6.5590625, -4.4093394, -1.8828168, 1.9764099
7: -8.8161144, -7.3933764, -8.7222109, -7.3930678, -1.4230466, 1.3288345
8: 1.0123816, 2.5559411, 1.0360909, 2.6289511, -1.4830644, 1.3925977
9: -9.4507437, -7.3981647, -9.3759394, -7.2045193, -2.1070061, 1.8575320

Time for backsubstitution: 5.74 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.37 seconds

### Candidate
type: A, layer: 3, pos: 402

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0004808, upper bound: 1.0063425
time: 3.76 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0199424, upper bound: 1.0063427
time: 3.80 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -7.3365030, -5.1072540, -7.3346786, -5.1047831, -2.2252545, 2.2222207
1: 1.8596368, 3.5449114, 1.9385400, 3.5586948, -1.5561008, 1.4615650
2: -4.9189172, -3.2766373, -4.9532833, -3.2837548, -1.4185317, 1.4575195
3: -11.0327139, -8.8772326, -11.0594978, -8.8771133, -1.9676447, 1.9924831
4: -5.5671411, -3.7495725, -5.5946770, -3.8422813, -1.7248597, 1.8451045
5: -9.0803432, -7.2987070, -9.0767145, -7.2972412, -1.7831020, 1.7780075
6: -6.5590625, -4.4093394, -6.5674729, -4.3215342, -1.9764094, 1.8828168
7: -8.7222109, -7.3930678, -8.8161144, -7.3933764, -1.3288345, 1.4230466
8: 1.0360909, 2.6289511, 1.0123816, 2.5559411, -1.3925977, 1.4830644
9: -9.3759394, -7.2045193, -9.4507437, -7.3981647, -1.8575318, 2.1070061

Time for backsubstitution: 5.77 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.38 seconds

### Candidate
type: A, layer: 3, pos: 402

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9793051, upper bound: 1.0004486
time: 3.76 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0004484, upper bound: 1.0004487
time: 3.34 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -7.3365030, -5.1072540, -7.3365030, -5.1072540, -2.2236433, 2.2236433
1: 1.8596368, 3.5449114, 1.8596368, 3.5449114, -1.5021257, 1.5021257
2: -4.9189172, -3.2766373, -4.9189172, -3.2766373, -1.4223635, 1.4223635
3: -11.0327139, -8.8772326, -11.0327139, -8.8772326, -1.9735541, 1.9735541
4: -5.5671411, -3.7495725, -5.5671411, -3.7495725, -1.8175685, 1.8175685
5: -9.0803432, -7.2987070, -9.0803432, -7.2987070, -1.7816362, 1.7816362
6: -6.5590625, -4.4093394, -6.5590625, -4.4093394, -1.8988466, 1.8988466
7: -8.7222109, -7.3930678, -8.7222109, -7.3930678, -1.3291430, 1.3291430
8: 1.0360909, 2.6289511, 1.0360909, 2.6289511, -1.4336586, 1.4336586
9: -9.3759394, -7.2045193, -9.3759394, -7.2045193, -1.9020741, 1.9020739

Time for backsubstitution: 5.70 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.38 seconds

### Candidate
type: A, layer: 3, pos: 402

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9793051, upper bound: 1.0004486
time: 3.61 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0004484, upper bound: 1.0004487
time: 3.33 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -7.3346786, -5.1047831, -7.3373909, -5.1035042, -2.2218304, 2.2235672
1: 1.9385400, 3.5586948, 1.9407153, 3.5610929, -1.4648170, 1.4687986
2: -4.9532833, -3.2837548, -4.9575157, -3.2789927, -1.4197404, 1.4166229
3: -11.0594978, -8.8771133, -11.0687046, -8.8778534, -1.9831791, 1.9935684
4: -5.5946770, -3.8422813, -5.5993242, -3.8404155, -1.7542615, 1.7570429
5: -9.0767145, -7.2972412, -9.0829248, -7.2714796, -1.8052349, 1.7856836
6: -6.5674729, -4.3215342, -6.5632372, -4.3059192, -1.9238164, 1.9215624
7: -8.8161144, -7.3933764, -8.8481512, -7.3791618, -1.3883922, 1.3995337
8: 1.0123816, 2.5559411, 1.0040431, 2.5473943, -1.4205532, 1.4391818
9: -9.4507437, -7.3981647, -9.4507761, -7.3961210, -1.8710039, 1.8700609

Time for backsubstitution: 5.70 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.37 seconds

### Candidate
type: A, layer: 3, pos: 402

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9998347, upper bound: 1.0063427
time: 3.72 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0171222, upper bound: 1.0063440
time: 3.87 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -7.3346786, -5.1047831, -7.3392220, -5.1059752, -2.2211704, 2.2259314
1: 1.9385400, 3.5586948, 1.8664083, 3.5426223, -1.4557853, 1.5559418
2: -4.9532833, -3.2837548, -4.9221191, -3.2719152, -1.4636822, 1.4197557
3: -11.0594978, -8.8771133, -11.0405436, -8.8780537, -1.9900608, 1.9753976
4: -5.5946770, -3.8422813, -5.5685873, -3.7483337, -1.8463433, 1.7263060
5: -9.0767145, -7.2972412, -9.0868692, -7.2729349, -1.8037796, 1.7896280
6: -6.5674729, -4.3215342, -6.5583653, -4.3931675, -1.8676577, 1.9623327
7: -8.8161144, -7.3933764, -8.7508974, -7.3796463, -1.4364681, 1.3575211
8: 1.0123816, 2.5559411, 1.0328178, 2.6160669, -1.4859755, 1.4102216
9: -9.4507437, -7.3981647, -9.3758202, -7.2023268, -2.1094577, 1.8583274

Time for backsubstitution: 5.79 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.38 seconds

### Candidate
type: A, layer: 3, pos: 402

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9998347, upper bound: 1.0063427
time: 3.88 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0171222, upper bound: 1.0063426
time: 5.91 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -7.3365030, -5.1072540, -7.3373909, -5.1035042, -2.2242031, 2.2229064
1: 1.8596368, 3.5449114, 1.9407153, 3.5610929, -1.5522499, 1.4616959
2: -4.9189172, -3.2766373, -4.9575157, -3.2789927, -1.4245169, 1.4603878
3: -11.0327139, -8.8772326, -11.0687046, -8.8778534, -1.9648643, 2.0000920
4: -5.5671411, -3.7495725, -5.5993242, -3.8404155, -1.7267256, 1.8497517
5: -9.0803432, -7.2987070, -9.0829248, -7.2714796, -1.8088636, 1.7842178
6: -6.5590625, -4.4093394, -6.5632372, -4.3059192, -1.9663756, 1.8705287
7: -8.7222109, -7.3930678, -8.8481512, -7.3791618, -1.3430490, 1.4550834
8: 1.0360909, 2.6289511, 1.0040431, 2.5473943, -1.3915262, 1.5006218
9: -9.3759394, -7.2045193, -9.4507761, -7.3961210, -1.8597252, 2.1082563

Time for backsubstitution: 5.77 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.41 seconds

### Candidate
type: A, layer: 3, pos: 402

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9814948, upper bound: 1.0004485
time: 3.59 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9992030, upper bound: 1.0004488
time: 3.49 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -7.3365030, -5.1072540, -7.3392220, -5.1059752, -2.2225909, 2.2243199
1: 1.8596368, 3.5449114, 1.8664083, 3.5426223, -1.4982870, 1.5019000
2: -4.9189172, -3.2766373, -4.9221191, -3.2719152, -1.4283674, 1.4239925
3: -11.0327139, -8.8772326, -11.0405436, -8.8780537, -1.9705667, 1.9812627
4: -5.5671411, -3.7495725, -5.5685873, -3.7483337, -1.8188074, 1.8190148
5: -9.0803432, -7.2987070, -9.0868692, -7.2729349, -1.8074083, 1.7881622
6: -6.5590625, -4.4093394, -6.5583653, -4.3931675, -1.8871093, 1.8841722
7: -8.7222109, -7.3930678, -8.7508974, -7.3796463, -1.3425646, 1.3578296
8: 1.0360909, 2.6289511, 1.0328178, 2.6160669, -1.4330492, 1.4510131
9: -9.3759394, -7.2045193, -9.3758202, -7.2023268, -1.9044404, 1.9032202

Time for backsubstitution: 5.66 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.38 seconds

### Candidate
type: A, layer: 3, pos: 402

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9814950, upper bound: 1.0004486
time: 3.60 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9992031, upper bound: 1.0004486
time: 3.62 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -7.3373909, -5.1035042, -7.3346786, -5.1047831, -2.2235675, 2.2218301
1: 1.9407153, 3.5610929, 1.9385400, 3.5586948, -1.4687986, 1.4648170
2: -4.9575157, -3.2789927, -4.9532833, -3.2837548, -1.4166229, 1.4197404
3: -11.0687046, -8.8778534, -11.0594978, -8.8771133, -1.9935684, 1.9831791
4: -5.5993242, -3.8404155, -5.5946770, -3.8422813, -1.7570429, 1.7542615
5: -9.0829248, -7.2714796, -9.0767145, -7.2972412, -1.7856836, 1.8052349
6: -6.5632372, -4.3059192, -6.5674729, -4.3215342, -1.9215627, 1.9238162
7: -8.8481512, -7.3791618, -8.8161144, -7.3933764, -1.3995337, 1.3883922
8: 1.0040431, 2.5473943, 1.0123816, 2.5559411, -1.4391818, 1.4205530
9: -9.4507761, -7.3961210, -9.4507437, -7.3981647, -1.8700607, 1.8710041

Time for backsubstitution: 5.76 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.40 seconds

### Candidate
type: A, layer: 3, pos: 402

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0004808, upper bound: 1.0088758
time: 3.79 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0199424, upper bound: 1.0088761
time: 3.62 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -7.3373909, -5.1035042, -7.3365030, -5.1072540, -2.2229066, 2.2242031
1: 1.9407153, 3.5610929, 1.8596368, 3.5449114, -1.4616957, 1.5522501
2: -4.9575157, -3.2789927, -4.9189172, -3.2766373, -1.4603882, 1.4245172
3: -11.0687046, -8.8778534, -11.0327139, -8.8772326, -2.0000916, 1.9648643
4: -5.5993242, -3.8404155, -5.5671411, -3.7495725, -1.8497517, 1.7267256
5: -9.0829248, -7.2714796, -9.0803432, -7.2987070, -1.7842178, 1.8088636
6: -6.5632372, -4.3059192, -6.5590625, -4.4093394, -1.8705282, 1.9663754
7: -8.8481512, -7.3791618, -8.7222109, -7.3930678, -1.4550834, 1.3430490
8: 1.0040431, 2.5473943, 1.0360909, 2.6289511, -1.5006216, 1.3915257
9: -9.4507761, -7.3961210, -9.3759394, -7.2045193, -2.1082568, 1.8597255

Time for backsubstitution: 5.73 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.37 seconds

### Candidate
type: A, layer: 3, pos: 402

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0004808, upper bound: 1.0088758
time: 3.70 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0199424, upper bound: 1.0088760
time: 3.65 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -7.3392220, -5.1059752, -7.3346786, -5.1047831, -2.2259316, 2.2211702
1: 1.8664083, 3.5426223, 1.9385400, 3.5586948, -1.5559416, 1.4557850
2: -4.9221191, -3.2719152, -4.9532833, -3.2837548, -1.4197557, 1.4636823
3: -11.0405436, -8.8780537, -11.0594978, -8.8771133, -1.9753971, 1.9900608
4: -5.5685873, -3.7483337, -5.5946770, -3.8422813, -1.7263060, 1.8463433
5: -9.0868692, -7.2729349, -9.0767145, -7.2972412, -1.7896280, 1.8037796
6: -6.5583653, -4.3931675, -6.5674729, -4.3215342, -1.9623327, 1.8676577
7: -8.7508974, -7.3796463, -8.8161144, -7.3933764, -1.3575211, 1.4364681
8: 1.0328178, 2.6160669, 1.0123816, 2.5559411, -1.4102216, 1.4859755
9: -9.3758202, -7.2023268, -9.4507437, -7.3981647, -1.8583276, 2.1094575

Time for backsubstitution: 5.73 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.39 seconds

### Candidate
type: A, layer: 3, pos: 402

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9793051, upper bound: 0.9992031
time: 3.79 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0004484, upper bound: 0.9992031
time: 3.55 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -7.3392220, -5.1059752, -7.3365030, -5.1072540, -2.2243199, 2.2225912
1: 1.8664083, 3.5426223, 1.8596368, 3.5449114, -1.5018997, 1.4982870
2: -4.9221191, -3.2719152, -4.9189172, -3.2766373, -1.4239926, 1.4283674
3: -11.0405436, -8.8780537, -11.0327139, -8.8772326, -1.9812627, 1.9705667
4: -5.5685873, -3.7483337, -5.5671411, -3.7495725, -1.8190148, 1.8188074
5: -9.0868692, -7.2729349, -9.0803432, -7.2987070, -1.7881622, 1.8074083
6: -6.5583653, -4.3931675, -6.5590625, -4.4093394, -1.8841724, 1.8871090
7: -8.7508974, -7.3796463, -8.7222109, -7.3930678, -1.3578296, 1.3425646
8: 1.0328178, 2.6160669, 1.0360909, 2.6289511, -1.4510131, 1.4330494
9: -9.3758202, -7.2023268, -9.3759394, -7.2045193, -1.9032199, 1.9044404

Time for backsubstitution: 5.75 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.37 seconds

### Candidate
type: A, layer: 3, pos: 402

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9793051, upper bound: 0.9992031
time: 3.62 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0004484, upper bound: 0.9992031
time: 3.31 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -7.3373909, -5.1035042, -7.3373909, -5.1035042, -2.2338867, 2.2338867
1: 1.9407153, 3.5610929, 1.9407153, 3.5610929, -1.4764955, 1.4764955
2: -4.9575157, -3.2789927, -4.9575157, -3.2789927, -1.4212043, 1.4212042
3: -11.0687046, -8.8778534, -11.0687046, -8.8778534, -1.9940505, 1.9940505
4: -5.5993242, -3.8404155, -5.5993242, -3.8404155, -1.7589087, 1.7589087
5: -9.0829248, -7.2714796, -9.0829248, -7.2714796, -1.8114452, 1.8114452
6: -6.5632372, -4.3059192, -6.5632372, -4.3059192, -1.9615312, 1.9615312
7: -8.8481512, -7.3791618, -8.8481512, -7.3791618, -1.3990815, 1.3990812
8: 1.0040431, 2.5473943, 1.0040431, 2.5473943, -1.4868536, 1.4868538
9: -9.4507761, -7.3961210, -9.4507761, -7.3961210, -1.8796334, 1.8796334

Time for backsubstitution: 5.72 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.37 seconds

### Candidate
type: A, layer: 3, pos: 402

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9946546, upper bound: 1.0088759
time: 3.62 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0142587, upper bound: 1.0088770
time: 3.55 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -7.3373909, -5.1035042, -7.3392220, -5.1059752, -2.2314157, 2.2357178
1: 1.9407153, 3.5610929, 1.8664083, 3.5426223, -1.4694250, 1.5637481
2: -4.9575157, -3.2789927, -4.9221191, -3.2719152, -1.4648690, 1.4259543
3: -11.0687046, -8.8778534, -11.0405436, -8.8780537, -2.0005751, 1.9757037
4: -5.5993242, -3.8404155, -5.5685873, -3.7483337, -1.8509905, 1.7281718
5: -9.0829248, -7.2714796, -9.0868692, -7.2729349, -1.8099899, 1.8153896
6: -6.5632372, -4.3059192, -6.5583653, -4.3931675, -1.9113517, 2.0044065
7: -8.8481512, -7.3791618, -8.7508974, -7.3796463, -1.4685049, 1.3684633
8: 1.0040431, 2.5473943, 1.0328178, 2.6160669, -1.5485716, 1.4582760
9: -9.4507761, -7.3961210, -9.3758202, -7.2023268, -2.1181114, 1.8670571

Time for backsubstitution: 5.70 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.37 seconds

### Candidate
type: A, layer: 3, pos: 402

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9946546, upper bound: 1.0088762
time: 3.57 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0142587, upper bound: 1.0088761
time: 3.78 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -7.3392220, -5.1059752, -7.3373909, -5.1035042, -2.2357178, 2.2314157
1: 1.8664083, 3.5426223, 1.9407153, 3.5610929, -1.5637486, 1.4694252
2: -4.9221191, -3.2719152, -4.9575157, -3.2789927, -1.4259546, 1.4648689
3: -11.0405436, -8.8780537, -11.0687046, -8.8778534, -1.9757042, 2.0005751
4: -5.5685873, -3.7483337, -5.5993242, -3.8404155, -1.7281718, 1.8509905
5: -9.0868692, -7.2729349, -9.0829248, -7.2714796, -1.8153896, 1.8099899
6: -6.5583653, -4.3931675, -6.5632372, -4.3059192, -2.0044069, 1.9113519
7: -8.7508974, -7.3796463, -8.8481512, -7.3791618, -1.3684633, 1.4685049
8: 1.0328178, 2.6160669, 1.0040431, 2.5473943, -1.4582758, 1.5485718
9: -9.3758202, -7.2023268, -9.4507761, -7.3961210, -1.8670568, 2.1181114

Time for backsubstitution: 5.74 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.40 seconds

### Candidate
type: A, layer: 3, pos: 402

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9754732, upper bound: 0.9992032
time: 3.67 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9957327, upper bound: 0.9992030
time: 3.65 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -7.3392220, -5.1059752, -7.3392220, -5.1059752, -2.2332468, 2.2332468
1: 1.8664083, 3.5426223, 1.8664083, 3.5426223, -1.5098827, 1.5098827
2: -4.9221191, -3.2719152, -4.9221191, -3.2719152, -1.4294837, 1.4294837
3: -11.0405436, -8.8780537, -11.0405436, -8.8780537, -1.9815168, 1.9815173
4: -5.5685873, -3.7483337, -5.5685873, -3.7483337, -1.8202536, 1.8202536
5: -9.0868692, -7.2729349, -9.0868692, -7.2729349, -1.8139343, 1.8139343
6: -6.5583653, -4.3931675, -6.5583653, -4.3931675, -1.9291835, 1.9291835
7: -8.7508974, -7.3796463, -8.7508974, -7.3796463, -1.3712511, 1.3712511
8: 1.0328178, 2.6160669, 1.0328178, 2.6160669, -1.4993553, 1.4993553
9: -9.3758202, -7.2023268, -9.3758202, -7.2023268, -1.9090672, 1.9090672

Time for backsubstitution: 5.74 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.38 seconds

### Candidate
type: A, layer: 3, pos: 402

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9754732, upper bound: 0.9992034
time: 3.61 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9957327, upper bound: 0.9992030
time: 3.62 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 13.36 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.36
Output dim: 1, lower bound: -1.0004808, upper bound: 1.0063425
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.36
Output dim: 1, lower bound: -1.0199424, upper bound: 1.0063427
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.36
Output dim: 1, lower bound: -1.0004808, upper bound: 1.0063425
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.36
Output dim: 1, lower bound: -1.0199424, upper bound: 1.0063427
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.36
Output dim: 1, lower bound: -0.9793051, upper bound: 1.0004486
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.36
Output dim: 1, lower bound: -1.0004484, upper bound: 1.0004487
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.36
Output dim: 1, lower bound: -0.9793051, upper bound: 1.0004486
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.36
Output dim: 1, lower bound: -1.0004484, upper bound: 1.0004487
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.36
Output dim: 1, lower bound: -0.9998347, upper bound: 1.0063427
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.36
Output dim: 1, lower bound: -1.0171222, upper bound: 1.0063440
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.36
Output dim: 1, lower bound: -0.9998347, upper bound: 1.0063427
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.36
Output dim: 1, lower bound: -1.0171222, upper bound: 1.0063426
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.36
Output dim: 1, lower bound: -0.9814948, upper bound: 1.0004485
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.36
Output dim: 1, lower bound: -0.9992030, upper bound: 1.0004488
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.36
Output dim: 1, lower bound: -0.9814950, upper bound: 1.0004486
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.36
Output dim: 1, lower bound: -0.9992031, upper bound: 1.0004486
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.36
Output dim: 1, lower bound: -1.0004808, upper bound: 1.0088758
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.36
Output dim: 1, lower bound: -1.0199424, upper bound: 1.0088761
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.36
Output dim: 1, lower bound: -1.0004808, upper bound: 1.0088758
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.36
Output dim: 1, lower bound: -1.0199424, upper bound: 1.0088760
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.36
Output dim: 1, lower bound: -0.9793051, upper bound: 0.9992031
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.36
Output dim: 1, lower bound: -1.0004484, upper bound: 0.9992031
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.36
Output dim: 1, lower bound: -0.9793051, upper bound: 0.9992031
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.36
Output dim: 1, lower bound: -1.0004484, upper bound: 0.9992031
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.36
Output dim: 1, lower bound: -0.9946546, upper bound: 1.0088759
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.36
Output dim: 1, lower bound: -1.0142587, upper bound: 1.0088770
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.36
Output dim: 1, lower bound: -0.9946546, upper bound: 1.0088762
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.36
Output dim: 1, lower bound: -1.0142587, upper bound: 1.0088761
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.36
Output dim: 1, lower bound: -0.9754732, upper bound: 0.9992032
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.36
Output dim: 1, lower bound: -0.9957327, upper bound: 0.9992030
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.36
Output dim: 1, lower bound: -0.9754732, upper bound: 0.9992034
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.36
Output dim: 1, lower bound: -0.9957327, upper bound: 0.9992030

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -7.3056631, -5.0999122, -7.3322668, -5.1048155, -2.2008476, 2.2323546
1: 1.9669955, 3.5556262, 1.9410954, 3.5585165, -1.4309678, 1.4542804
2: -4.9469681, -3.2823098, -4.9527779, -3.2839670, -1.4046326, 1.4152592
3: -11.0628061, -8.8965263, -11.0594873, -8.8787050, -1.9895945, 1.9656940
4: -5.5925274, -3.8488555, -5.5943375, -3.8427701, -1.7497573, 1.7454820
5: -9.0700178, -7.3113856, -9.0763092, -7.2983747, -1.7716432, 1.7649236
6: -6.5675230, -4.3262696, -6.5672469, -4.3219395, -1.9360495, 1.9290993
7: -8.8376179, -7.4093542, -8.8160896, -7.3946590, -1.3730149, 1.3478725
8: 1.0145617, 2.5453753, 1.0126081, 2.5551071, -1.4216743, 1.4124389
9: -9.4554577, -7.4482117, -9.4504967, -7.4021916, -1.8643785, 1.8161151

Time for backsubstitution: 5.71 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.37 seconds

### Candidate
type: B, layer: 3, pos: 402

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0298901, upper bound: 1.0298921
time: 4.01 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0298901, upper bound: 1.0490596
time: 4.04 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -7.3252287, -5.1051140, -7.3346786, -5.1047831, -2.2204456, 2.2226639
1: 1.9471767, 3.5582409, 1.9385400, 3.5586948, -1.4451764, 1.4685500
2: -4.9501371, -3.2865560, -4.9532833, -3.2837548, -1.4218509, 1.4090512
3: -11.0593834, -8.8867168, -11.0594978, -8.8771133, -1.9858975, 1.9732208
4: -5.5903339, -3.8453085, -5.5946770, -3.8422813, -1.7480526, 1.7493684
5: -9.0735254, -7.3005614, -9.0767145, -7.2972412, -1.7762842, 1.7761531
6: -6.5660152, -4.3244324, -6.5674729, -4.3215342, -1.9324293, 1.9335349
7: -8.8158188, -7.4089589, -8.8161144, -7.3933764, -1.3690143, 1.3433752
8: 1.0131907, 2.5515699, 1.0123816, 2.5559411, -1.4207244, 1.4179418
9: -9.4498243, -7.4126263, -9.4507437, -7.3981647, -1.8681004, 1.8320370

Time for backsubstitution: 5.68 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.37 seconds

### Candidate
type: B, layer: 3, pos: 402

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0490578, upper bound: 1.0298908
time: 4.39 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0490578, upper bound: 1.0490584
time: 4.39 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -7.3056631, -5.0999122, -7.3340855, -5.1072841, -2.1983790, 2.2341733
1: 1.9669955, 3.5556262, 1.8619370, 3.5447764, -1.4238639, 1.5419288
2: -4.9469681, -3.2823098, -4.9183912, -3.2768278, -1.4484024, 1.4199237
3: -11.0628061, -8.8965263, -11.0327053, -8.8788319, -1.9961166, 1.9473801
4: -5.5925274, -3.8488555, -5.5667787, -3.7501488, -1.8423786, 1.7179232
5: -9.0700178, -7.3113856, -9.0799627, -7.2998414, -1.7701764, 1.7685771
6: -6.5675230, -4.3262696, -6.5588427, -4.4097219, -1.8850837, 1.9716172
7: -8.8376179, -7.4093542, -8.7221889, -7.3943253, -1.4432926, 1.3128347
8: 1.0145617, 2.5453753, 1.0363002, 2.6281190, -1.4830575, 1.3834510
9: -9.4554577, -7.4482117, -9.3757219, -7.2092562, -2.1019671, 1.8048477

Time for backsubstitution: 5.67 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.38 seconds

### Candidate
type: B, layer: 3, pos: 423

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9874623, upper bound: 0.9952121
time: 3.87 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9876463, upper bound: 0.9928842
time: 3.55 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -7.3252287, -5.1051140, -7.3365030, -5.1072540, -2.2179747, 2.2250369
1: 1.9471767, 3.5582409, 1.8596368, 3.5449114, -1.4392054, 1.5559831
2: -4.9501371, -3.2865560, -4.9189172, -3.2766373, -1.4629438, 1.4138279
3: -11.0593834, -8.8867168, -11.0327139, -8.8772326, -1.9924207, 1.9559574
4: -5.5903339, -3.8453085, -5.5671411, -3.7495725, -1.8407614, 1.7218325
5: -9.0735254, -7.3005614, -9.0803432, -7.2987070, -1.7748184, 1.7797818
6: -6.5660152, -4.3244324, -6.5590625, -4.4093394, -1.8813953, 1.9758775
7: -8.8158188, -7.4089589, -8.7222109, -7.3930678, -1.4227509, 1.3124857
8: 1.0131907, 2.5515699, 1.0360909, 2.6289511, -1.4821644, 1.3874424
9: -9.4498243, -7.4126263, -9.3759394, -7.2045193, -2.1062961, 1.8149090

Time for backsubstitution: 5.72 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.37 seconds

### Candidate
type: B, layer: 3, pos: 402

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0199425, upper bound: 0.9858729
time: 3.75 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0199425, upper bound: 1.0063428
time: 3.79 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -7.3072877, -5.1018953, -7.3322668, -5.1048155, -2.2024722, 2.2303715
1: 1.8873105, 3.5444431, 1.9410954, 3.5585165, -1.5227208, 1.4496617
2: -4.9123201, -3.2756791, -4.9527779, -3.2839670, -1.4089191, 1.4584334
3: -11.0354338, -8.8966808, -11.0594873, -8.8787050, -1.9727049, 1.9722323
4: -5.5645566, -3.7567065, -5.5943375, -3.8427701, -1.7217865, 1.8376310
5: -9.0741472, -7.3128581, -9.0763092, -7.2983747, -1.7757726, 1.7634511
6: -6.5592098, -4.4138160, -6.5672469, -4.3219395, -1.9784174, 1.8789065
7: -8.7437325, -7.4088888, -8.8160896, -7.3946590, -1.3421388, 1.4072008
8: 1.0390034, 2.6184092, 1.0126081, 2.5551071, -1.3909955, 1.4737771
9: -9.3746729, -7.2596254, -9.4504967, -7.4021916, -1.8478074, 2.0486069

Time for backsubstitution: 5.74 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.39 seconds

### Candidate
type: B, layer: 3, pos: 402

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9858723, upper bound: 1.0004812
time: 3.80 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9858723, upper bound: 1.0199427
time: 3.80 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -7.3270020, -5.1075830, -7.3346786, -5.1047831, -2.2222190, 2.2220032
1: 1.8684199, 3.5445313, 1.9385400, 3.5586948, -1.5330641, 1.4614334
2: -4.9158492, -3.2790942, -4.9532833, -3.2837548, -1.4242761, 1.4527977
3: -11.0326090, -8.8868847, -11.0594978, -8.8771133, -1.9675870, 1.9797440
4: -5.5624924, -3.7522728, -5.5946770, -3.8422813, -1.7202110, 1.8424041
5: -9.0771637, -7.3020205, -9.0767145, -7.2972412, -1.7799225, 1.7746940
6: -6.5576620, -4.4125118, -6.5674729, -4.3215342, -1.9750981, 1.8830996
7: -8.7219315, -7.4087892, -8.8161144, -7.3933764, -1.3285551, 1.4073253
8: 1.0368590, 2.6243343, 1.0123816, 2.5559411, -1.3918028, 1.4791679
9: -9.3750725, -7.2159300, -9.4507437, -7.3981647, -1.8568656, 2.0655417

Time for backsubstitution: 5.71 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.40 seconds

### Candidate
type: B, layer: 3, pos: 402

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0063422, upper bound: 1.0004815
time: 3.80 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0063422, upper bound: 1.0199431
time: 3.57 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -7.3072877, -5.1018953, -7.3340855, -5.1072841, -2.2000036, 2.2321901
1: 1.8873105, 3.5444431, 1.8619370, 3.5447764, -1.4651003, 1.4883389
2: -4.9123201, -3.2756791, -4.9183912, -3.2768278, -1.4128547, 1.4230323
3: -11.0354338, -8.8966808, -11.0327053, -8.8788319, -1.9785366, 1.9533033
4: -5.5645566, -3.7567065, -5.5667787, -3.7501488, -1.8144078, 1.8100722
5: -9.0741472, -7.3128581, -9.0799627, -7.2998414, -1.7743058, 1.7671046
6: -6.5592098, -4.4138160, -6.5588427, -4.4097219, -1.9009018, 1.8944237
7: -8.7437325, -7.4088888, -8.7221889, -7.3943253, -1.3443224, 1.3133001
8: 1.0390034, 2.6184092, 1.0363002, 2.6281190, -1.4336565, 1.4245334
9: -9.3746729, -7.2596254, -9.3757219, -7.2092562, -1.8990829, 1.8518271

Time for backsubstitution: 5.72 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.38 seconds

### Candidate
type: B, layer: 3, pos: 423

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9653021, upper bound: 0.9893184
time: 3.60 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9653665, upper bound: 0.9865583
time: 3.94 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -7.3270020, -5.1075830, -7.3365030, -5.1072540, -2.2197480, 2.2234259
1: 1.8684199, 3.5445313, 1.8596368, 3.5449114, -1.4776273, 1.5019958
2: -4.9158492, -3.2790942, -4.9189172, -3.2766373, -1.4307857, 1.4175954
3: -11.0326090, -8.8868847, -11.0327139, -8.8772326, -1.9734945, 1.9617853
4: -5.5624924, -3.7522728, -5.5671411, -3.7495725, -1.8129199, 1.8148682
5: -9.0771637, -7.3020205, -9.0803432, -7.2987070, -1.7784567, 1.7783227
6: -6.5576620, -4.4125118, -6.5590625, -4.4093394, -1.8975463, 1.9000297
7: -8.7219315, -7.4087892, -8.7222109, -7.3930678, -1.3288636, 1.3131714
8: 1.0368590, 2.6243343, 1.0360909, 2.6289511, -1.4328203, 1.4315791
9: -9.3750725, -7.2159300, -9.3759394, -7.2045193, -1.9013093, 1.8643253

Time for backsubstitution: 5.75 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.37 seconds

### Candidate
type: B, layer: 3, pos: 402

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0004485, upper bound: 0.9793053
time: 3.32 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0004485, upper bound: 1.0004487
time: 3.28 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -7.3056631, -5.0999122, -7.3350029, -5.1035366, -2.2010922, 2.2350907
1: 1.9669955, 3.5556262, 1.9432631, 3.5609145, -1.4270897, 1.4544325
2: -4.9469681, -3.2823098, -4.9570141, -3.2792041, -1.4106221, 1.4182625
3: -11.0628061, -8.8965263, -11.0686970, -8.8794413, -1.9868293, 1.9733019
4: -5.5925274, -3.8488555, -5.5989790, -3.8409052, -1.7516222, 1.7501235
5: -9.0700178, -7.3113856, -9.0825195, -7.2726107, -1.7974072, 1.7711339
6: -6.5675230, -4.3262696, -6.5630112, -4.3063216, -1.9260137, 1.9168034
7: -8.8376179, -7.4093542, -8.8481245, -7.3804312, -1.3917422, 1.3772178
8: 1.0145617, 2.5453753, 1.0042748, 2.5465655, -1.4207020, 1.4299963
9: -9.4554577, -7.4482117, -9.4505310, -7.4001446, -1.8665762, 1.8173666

Time for backsubstitution: 5.87 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.42 seconds

### Candidate
type: B, layer: 3, pos: 402

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0364058, upper bound: 1.0298904
time: 4.12 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0364058, upper bound: 1.0490580
time: 3.98 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -7.3252287, -5.1051140, -7.3373909, -5.1035042, -2.2217245, 2.2233496
1: 1.9471767, 3.5582409, 1.9407153, 3.5610929, -1.4432783, 1.4686809
2: -4.9501371, -3.2865560, -4.9575157, -3.2789927, -1.4280457, 1.4119194
3: -11.0593834, -8.8867168, -11.0687046, -8.8778534, -1.9831171, 1.9809294
4: -5.5903339, -3.8453085, -5.5993242, -3.8404155, -1.7499185, 1.7540157
5: -9.0735254, -7.3005614, -9.0829248, -7.2714796, -1.8020458, 1.7823634
6: -6.5660152, -4.3244324, -6.5632372, -4.3059192, -1.9223945, 1.9198251
7: -8.8158188, -7.4089589, -8.8481512, -7.3791618, -1.3872159, 1.3757701
8: 1.0131907, 2.5515699, 1.0040431, 2.5473943, -1.4196529, 1.4366910
9: -9.4498243, -7.4126263, -9.4507761, -7.3961210, -1.8702939, 1.8325911

Time for backsubstitution: 5.76 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.40 seconds

### Candidate
type: B, layer: 3, pos: 402

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0510386, upper bound: 1.0298905
time: 4.37 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0510386, upper bound: 1.0490581
time: 4.39 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -7.3056631, -5.0999122, -7.3368177, -5.1060066, -2.1996565, 2.2369056
1: 1.9669955, 3.5556262, 1.8689628, 3.5424728, -1.4180436, 1.5420241
2: -4.9469681, -3.2823098, -4.9215965, -3.2721057, -1.4545679, 1.4212397
3: -11.0628061, -8.8965263, -11.0405340, -8.8796434, -1.9937077, 1.9551306
4: -5.5925274, -3.8488555, -5.5682220, -3.7488577, -1.8436697, 1.7193666
5: -9.0700178, -7.3113856, -9.0864906, -7.2740679, -1.7959499, 1.7751050
6: -6.5675230, -4.3262696, -6.5581436, -4.3935795, -1.8698587, 1.9575667
7: -8.8376179, -7.4093542, -8.7508717, -7.3808975, -1.4567204, 1.3415174
8: 1.0145617, 2.5453753, 1.0330420, 2.6152353, -1.4862993, 1.4010589
9: -9.4554577, -7.4482117, -9.3756027, -7.2071395, -2.1044207, 1.8056431

Time for backsubstitution: 5.77 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.41 seconds

### Candidate
type: B, layer: 3, pos: 423

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9865008, upper bound: 0.9952124
time: 3.97 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9866903, upper bound: 0.9928827
time: 3.67 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -7.3252287, -5.1051140, -7.3392220, -5.1059752, -2.2192535, 2.2257137
1: 1.9471767, 3.5582409, 1.8664083, 3.5426223, -1.4372311, 1.5558238
2: -4.9501371, -3.2865560, -4.9221191, -3.2719152, -1.4690754, 1.4150521
3: -11.0593834, -8.8867168, -11.0405436, -8.8780537, -1.9899983, 1.9638081
4: -5.5903339, -3.8453085, -5.5685873, -3.7483337, -1.8420002, 1.7232788
5: -9.0735254, -7.3005614, -9.0868692, -7.2729349, -1.8005905, 1.7863078
6: -6.5660152, -4.3244324, -6.5583653, -4.3931675, -1.8662362, 1.9594891
7: -8.8158188, -7.4089589, -8.7508974, -7.3796463, -1.4361725, 1.3413539
8: 1.0131907, 2.5515699, 1.0328178, 2.6160669, -1.4850755, 1.4095066
9: -9.4498243, -7.4126263, -9.3758202, -7.2023268, -2.1087472, 1.8150139

Time for backsubstitution: 5.72 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.40 seconds

### Candidate
type: B, layer: 3, pos: 402

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0171224, upper bound: 0.9858261
time: 5.62 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0171224, upper bound: 1.0063425
time: 4.67 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -7.3072877, -5.1018953, -7.3350029, -5.1035366, -2.2033935, 2.2331076
1: 1.8873105, 3.5444431, 1.9432631, 3.5609145, -1.5188432, 1.4498136
2: -4.9123201, -3.2756791, -4.9570141, -3.2792041, -1.4149089, 1.4614367
3: -11.0354338, -8.8966808, -11.0686970, -8.8794413, -1.9699392, 1.9798408
4: -5.5645566, -3.7567065, -5.5989790, -3.8409052, -1.7236514, 1.8422725
5: -9.0741472, -7.3128581, -9.0825195, -7.2726107, -1.8015366, 1.7696614
6: -6.5592098, -4.4138160, -6.5630112, -4.3063216, -1.9683816, 1.8666105
7: -8.7437325, -7.4088888, -8.8481245, -7.3804312, -1.3608670, 1.4392357
8: 1.0390034, 2.6184092, 1.0042748, 2.5465655, -1.3900232, 1.4913342
9: -9.3746729, -7.2596254, -9.4505310, -7.4001446, -1.8500047, 2.0498586

Time for backsubstitution: 5.80 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.41 seconds

### Candidate
type: B, layer: 3, pos: 402

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9918073, upper bound: 1.0004809
time: 3.73 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9918073, upper bound: 1.0199425
time: 3.88 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -7.3270020, -5.1075830, -7.3373909, -5.1035042, -2.2234979, 2.2226892
1: 1.8684199, 3.5445313, 1.9407153, 3.5610929, -1.5311661, 1.4615643
2: -4.9158492, -3.2790942, -4.9575157, -3.2789927, -1.4304709, 1.4556661
3: -11.0326090, -8.8868847, -11.0687046, -8.8778534, -1.9648066, 1.9874525
4: -5.5624924, -3.7522728, -5.5993242, -3.8404155, -1.7220769, 1.8470514
5: -9.0771637, -7.3020205, -9.0829248, -7.2714796, -1.8056841, 1.7809043
6: -6.5576620, -4.4125118, -6.5632372, -4.3059192, -1.9650633, 1.8693898
7: -8.7219315, -7.4087892, -8.8481512, -7.3791618, -1.3427696, 1.4393620
8: 1.0368590, 2.6243343, 1.0040431, 2.5473943, -1.3907309, 1.4979169
9: -9.3750725, -7.2159300, -9.4507761, -7.3961210, -1.8590591, 2.0660958

Time for backsubstitution: 5.86 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.42 seconds

### Candidate
type: B, layer: 3, pos: 402

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0088757, upper bound: 1.0004811
time: 3.68 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0088757, upper bound: 1.0199427
time: 3.57 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -7.3072877, -5.1018953, -7.3368177, -5.1060066, -2.2012811, 2.2349224
1: 1.8873105, 3.5444431, 1.8689628, 3.5424728, -1.4612317, 1.4884691
2: -4.9123201, -3.2756791, -4.9215965, -3.2721057, -1.4188609, 1.4249136
3: -11.0354338, -8.8966808, -11.0405340, -8.8796434, -1.9755635, 1.9610119
4: -5.5645566, -3.7567065, -5.5682220, -3.7488577, -1.8156989, 1.8115156
5: -9.0741472, -7.3128581, -9.0864906, -7.2740679, -1.8000793, 1.7736325
6: -6.5592098, -4.4138160, -6.5581436, -4.3935795, -1.8891609, 1.8797638
7: -8.7437325, -7.4088888, -8.7508717, -7.3808975, -1.3627832, 1.3419828
8: 1.0390034, 2.6184092, 1.0330420, 2.6152353, -1.4331460, 1.4418776
9: -9.3746729, -7.2596254, -9.3756027, -7.2071395, -1.9014516, 1.8529725

Time for backsubstitution: 5.81 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.41 seconds

### Candidate
type: B, layer: 3, pos: 423

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9674953, upper bound: 0.9893182
time: 3.60 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9675927, upper bound: 0.9865581
time: 3.61 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -7.3270020, -5.1075830, -7.3392220, -5.1059752, -2.2210269, 2.2241027
1: 1.8684199, 3.5445313, 1.8664083, 3.5426223, -1.4756887, 1.5017700
2: -4.9158492, -3.2790942, -4.9221191, -3.2719152, -1.4370251, 1.4192244
3: -11.0326090, -8.8868847, -11.0405436, -8.8780537, -1.9705067, 1.9696736
4: -5.5624924, -3.7522728, -5.5685873, -3.7483337, -1.8141587, 1.8163145
5: -9.0771637, -7.3020205, -9.0868692, -7.2729349, -1.8042288, 1.7848487
6: -6.5576620, -4.4125118, -6.5583653, -4.3931675, -1.8858089, 1.8836622
7: -8.7219315, -7.4087892, -8.7508974, -7.3796463, -1.3422852, 1.3421082
8: 1.0368590, 2.6243343, 1.0328178, 2.6160669, -1.4322114, 1.4503424
9: -9.3750725, -7.2159300, -9.3758202, -7.2023268, -1.9036760, 1.8647749

Time for backsubstitution: 5.83 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.38 seconds

### Candidate
type: B, layer: 3, pos: 402

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9992031, upper bound: 0.9793052
time: 3.62 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9992031, upper bound: 1.0004486
time: 3.67 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -7.3086243, -5.0988011, -7.3322668, -5.1048155, -2.2031193, 2.2334657
1: 1.9694457, 3.5586653, 1.9410954, 3.5585165, -1.4386706, 1.4554825
2: -4.9515362, -3.2774415, -4.9527779, -3.2839670, -1.4077759, 1.4216143
3: -11.0759735, -8.8965273, -11.0594873, -8.8787050, -1.9975615, 1.9632535
4: -5.5970941, -3.8469534, -5.5943375, -3.8427701, -1.7543240, 1.7473841
5: -9.0761414, -7.2854629, -9.0763092, -7.2983747, -1.7777667, 1.7908463
6: -6.5633383, -4.3104649, -6.5672469, -4.3219395, -1.9224510, 1.9191604
7: -8.8699198, -7.3949952, -8.8160896, -7.3946590, -1.4053988, 1.3687322
8: 1.0048432, 2.5368629, 1.0126081, 2.5551071, -1.4411960, 1.4132073
9: -9.4552622, -7.4461021, -9.4504967, -7.4021916, -1.8649898, 1.8183014

Time for backsubstitution: 5.85 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.41 seconds

### Candidate
type: B, layer: 3, pos: 402

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0298901, upper bound: 1.0364076
time: 4.22 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0298901, upper bound: 1.0510402
time: 4.07 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -7.3279648, -5.1038380, -7.3346786, -5.1047831, -2.2231817, 2.2216122
1: 1.9493752, 3.5606382, 1.9385400, 3.5586948, -1.4452603, 1.4646327
2: -4.9543581, -3.2817464, -4.9532833, -3.2837548, -1.4248326, 1.4150560
3: -11.0685892, -8.8873997, -11.0594978, -8.8771133, -1.9935036, 1.9705644
4: -5.5949545, -3.8434410, -5.5946770, -3.8422813, -1.7526731, 1.7512360
5: -9.0797663, -7.2746916, -9.0767145, -7.2972412, -1.7825251, 1.8020229
6: -6.5617738, -4.3088121, -6.5674729, -4.3215342, -1.9200120, 1.9234910
7: -8.8478403, -7.3947415, -8.8161144, -7.3933764, -1.3983855, 1.3641481
8: 1.0049062, 2.5425978, 1.0123816, 2.5559411, -1.4382329, 1.4175019
9: -9.4498444, -7.4104156, -9.4507437, -7.3981647, -1.8693302, 1.8340695

Time for backsubstitution: 5.72 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.37 seconds

### Candidate
type: B, layer: 3, pos: 402

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0490578, upper bound: 1.0364076
time: 4.16 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0490597, upper bound: 1.0510401
time: 5.64 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -7.3086243, -5.0988011, -7.3340855, -5.1072841, -2.2013402, 2.2352843
1: 1.9694457, 3.5586653, 1.8619370, 3.5447764, -1.4315662, 1.5431309
2: -4.9515362, -3.2774415, -4.9183912, -3.2768278, -1.4515457, 1.4262787
3: -11.0759735, -8.8965273, -11.0327053, -8.8788319, -2.0040836, 1.9449396
4: -5.5970941, -3.8469534, -5.5667787, -3.7501488, -1.8469453, 1.7198253
5: -9.0761414, -7.2854629, -9.0799627, -7.2998414, -1.7763000, 1.7944999
6: -6.5633383, -4.3104649, -6.5588427, -4.4097219, -1.8714852, 1.9616783
7: -8.8699198, -7.3949952, -8.7221889, -7.3943253, -1.4755945, 1.3271937
8: 1.0048432, 2.5368629, 1.0363002, 2.6281190, -1.5025792, 1.3842196
9: -9.4552622, -7.4461021, -9.3757219, -7.2092562, -2.1025784, 1.8070335

Time for backsubstitution: 7.33 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.39 seconds

### Candidate
type: B, layer: 3, pos: 423

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9874623, upper bound: 0.9977455
time: 3.89 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9876463, upper bound: 0.9955627
time: 3.50 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -7.3279648, -5.1038380, -7.3365030, -5.1072540, -2.2207108, 2.2239852
1: 1.9493752, 3.5606382, 1.8596368, 3.5449114, -1.4392898, 1.5520661
2: -4.9543581, -3.2817464, -4.9189172, -3.2766373, -1.4659255, 1.4198327
3: -11.0685892, -8.8873997, -11.0327139, -8.8772326, -2.0000267, 1.9533010
4: -5.5949545, -3.8434410, -5.5671411, -3.7495725, -1.8453820, 1.7237000
5: -9.0797663, -7.2746916, -9.0803432, -7.2987070, -1.7810593, 1.8056517
6: -6.5617738, -4.3088121, -6.5590625, -4.4093394, -1.8689780, 1.9658337
7: -8.8478403, -7.3947415, -8.7222109, -7.3930678, -1.4547725, 1.3274693
8: 1.0049062, 2.5425978, 1.0360909, 2.6289511, -1.4996731, 1.3870022
9: -9.4498444, -7.4104156, -9.3759394, -7.2045193, -2.1075263, 1.8169417

Time for backsubstitution: 5.78 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.41 seconds

### Candidate
type: B, layer: 3, pos: 402

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0199425, upper bound: 0.9918077
time: 3.83 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0199425, upper bound: 1.0088761
time: 3.79 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -7.3102431, -5.1007833, -7.3322668, -5.1048155, -2.2054143, 2.2314835
1: 1.8918023, 3.5445967, 1.9410954, 3.5585165, -1.5269742, 1.4519508
2: -4.9162955, -3.2708511, -4.9527779, -3.2839670, -1.4106162, 1.4647361
3: -11.0491657, -8.8964367, -11.0594873, -8.8787050, -1.9806366, 1.9701328
4: -5.5661416, -3.7553396, -5.5943375, -3.8427701, -1.7233715, 1.8389978
5: -9.0802746, -7.2869272, -9.0763092, -7.2983747, -1.7818999, 1.7893820
6: -6.5576377, -4.3976560, -6.5672469, -4.3219395, -1.9622684, 1.8630269
7: -8.7770252, -7.3922777, -8.8160896, -7.3946590, -1.3710237, 1.4238119
8: 1.0335259, 2.6055570, 1.0126081, 2.5551071, -1.4151931, 1.4804685
9: -9.3740215, -7.2575064, -9.4504967, -7.4021916, -1.8476853, 2.0507135

Time for backsubstitution: 5.82 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.39 seconds

### Candidate
type: B, layer: 3, pos: 402

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9858257, upper bound: 0.9998352
time: 3.80 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9858257, upper bound: 1.0171229
time: 3.81 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -7.3297424, -5.1063108, -7.3346786, -5.1047831, -2.2249594, 2.2209513
1: 1.8750618, 3.5422070, 1.9385400, 3.5586948, -1.5333335, 1.4555411
2: -4.9190187, -3.2743287, -4.9532833, -3.2837548, -1.4265597, 1.4589804
3: -11.0404329, -8.8876667, -11.0594978, -8.8771133, -1.9753356, 1.9774418
4: -5.5640659, -3.7513127, -5.5946770, -3.8422813, -1.7217846, 1.8433642
5: -9.0837221, -7.2761359, -9.0767145, -7.2972412, -1.7864809, 1.8005786
6: -6.5569243, -4.3960748, -6.5674729, -4.3215342, -1.9606681, 1.8672011
7: -8.7505980, -7.3953652, -8.8161144, -7.3933764, -1.3572216, 1.4207492
8: 1.0336723, 2.6113100, 1.0123816, 2.5559411, -1.4093604, 1.4839649
9: -9.3749437, -7.2137265, -9.4507437, -7.3981647, -1.8576467, 2.0678277

Time for backsubstitution: 5.75 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.40 seconds

### Candidate
type: B, layer: 3, pos: 402

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0063422, upper bound: 0.9998351
time: 3.93 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0063422, upper bound: 1.0171227
time: 3.99 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -7.3102431, -5.1007833, -7.3340855, -5.1072841, -2.2029591, 2.2333021
1: 1.8918023, 3.5445967, 1.8619370, 3.5447764, -1.4717426, 1.4906263
2: -4.9162955, -3.2708511, -4.9183912, -3.2768278, -1.4153204, 1.4291550
3: -11.0491657, -8.8964367, -11.0327053, -8.8788319, -1.9865074, 1.9506373
4: -5.5661416, -3.7553396, -5.5667787, -3.7501488, -1.8159928, 1.8114390
5: -9.0802746, -7.2869272, -9.0799627, -7.2998414, -1.7804332, 1.7930355
6: -6.5576377, -4.3976560, -6.5588427, -4.4097219, -1.8846169, 1.8823702
7: -8.7770252, -7.3922777, -8.7221889, -7.3943253, -1.3773682, 1.3299112
8: 1.0335259, 2.6055570, 1.0363002, 2.6281190, -1.4533174, 1.4256239
9: -9.3740215, -7.2575064, -9.3757219, -7.2092562, -1.8995798, 1.8541753

Time for backsubstitution: 5.77 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.39 seconds

### Candidate
type: B, layer: 3, pos: 423

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9653021, upper bound: 0.9880981
time: 3.79 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9653665, upper bound: 0.9853031
time: 3.70 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -7.3297424, -5.1063108, -7.3365030, -5.1072540, -2.2224884, 2.2223737
1: 1.8750618, 3.5422070, 1.8596368, 3.5449114, -1.4780202, 1.4980755
2: -4.9190187, -3.2743287, -4.9189172, -3.2766373, -1.4337134, 1.4236194
3: -11.0404329, -8.8876667, -11.0327139, -8.8772326, -1.9811993, 1.9589157
4: -5.5640659, -3.7513127, -5.5671411, -3.7495725, -1.8144934, 1.8158283
5: -9.0837221, -7.2761359, -9.0803432, -7.2987070, -1.7850151, 1.8042073
6: -6.5569243, -4.3960748, -6.5590625, -4.4093394, -1.8825374, 1.8879795
7: -8.7505980, -7.3953652, -8.7222109, -7.3930678, -1.3575301, 1.3268456
8: 1.0336723, 2.6113100, 1.0360909, 2.6289511, -1.4500990, 1.4315815
9: -9.3749437, -7.2137265, -9.3759394, -7.2045193, -1.9024355, 1.8665311

Time for backsubstitution: 5.75 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.39 seconds

### Candidate
type: B, layer: 3, pos: 402

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0004485, upper bound: 0.9814950
time: 3.36 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0004485, upper bound: 0.9992031
time: 3.35 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 12.85 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 12.85
Output dim: 1, lower bound: -1.0298901, upper bound: 1.0298921
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 12.85
Output dim: 1, lower bound: -1.0298901, upper bound: 1.0490596
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 12.85
Output dim: 1, lower bound: -1.0490578, upper bound: 1.0298908
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 12.85
Output dim: 1, lower bound: -1.0490578, upper bound: 1.0490584
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 12.85
Output dim: 1, lower bound: -0.9874623, upper bound: 0.9952121
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 12.85
Output dim: 1, lower bound: -0.9876463, upper bound: 0.9928842
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 12.85
Output dim: 1, lower bound: -1.0199425, upper bound: 0.9858729
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 12.85
Output dim: 1, lower bound: -1.0199425, upper bound: 1.0063428
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 12.85
Output dim: 1, lower bound: -0.9858723, upper bound: 1.0004812
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 12.85
Output dim: 1, lower bound: -0.9858723, upper bound: 1.0199427
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 12.85
Output dim: 1, lower bound: -1.0063422, upper bound: 1.0004815
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 12.85
Output dim: 1, lower bound: -1.0063422, upper bound: 1.0199431
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 12.85
Output dim: 1, lower bound: -0.9653021, upper bound: 0.9893184
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 12.85
Output dim: 1, lower bound: -0.9653665, upper bound: 0.9865583
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 12.85
Output dim: 1, lower bound: -1.0004485, upper bound: 0.9793053
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 12.85
Output dim: 1, lower bound: -1.0004485, upper bound: 1.0004487
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 12.85
Output dim: 1, lower bound: -1.0364058, upper bound: 1.0298904
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 12.85
Output dim: 1, lower bound: -1.0364058, upper bound: 1.0490580
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 12.85
Output dim: 1, lower bound: -1.0510386, upper bound: 1.0298905
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 12.85
Output dim: 1, lower bound: -1.0510386, upper bound: 1.0490581
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 12.85
Output dim: 1, lower bound: -0.9865008, upper bound: 0.9952124
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 12.85
Output dim: 1, lower bound: -0.9866903, upper bound: 0.9928827
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 12.85
Output dim: 1, lower bound: -1.0171224, upper bound: 0.9858261
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 12.85
Output dim: 1, lower bound: -1.0171224, upper bound: 1.0063425
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 12.85
Output dim: 1, lower bound: -0.9918073, upper bound: 1.0004809
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 12.85
Output dim: 1, lower bound: -0.9918073, upper bound: 1.0199425
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 12.85
Output dim: 1, lower bound: -1.0088757, upper bound: 1.0004811
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 12.85
Output dim: 1, lower bound: -1.0088757, upper bound: 1.0199427
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 12.85
Output dim: 1, lower bound: -0.9674953, upper bound: 0.9893182
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 12.85
Output dim: 1, lower bound: -0.9675927, upper bound: 0.9865581
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 12.85
Output dim: 1, lower bound: -0.9992031, upper bound: 0.9793052
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 12.85
Output dim: 1, lower bound: -0.9992031, upper bound: 1.0004486
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 12.85
Output dim: 1, lower bound: -1.0298901, upper bound: 1.0364076
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 12.85
Output dim: 1, lower bound: -1.0298901, upper bound: 1.0510402
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 12.85
Output dim: 1, lower bound: -1.0490578, upper bound: 1.0364076
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 12.85
Output dim: 1, lower bound: -1.0490597, upper bound: 1.0510401
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 12.85
Output dim: 1, lower bound: -0.9874623, upper bound: 0.9977455
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 12.85
Output dim: 1, lower bound: -0.9876463, upper bound: 0.9955627
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 12.85
Output dim: 1, lower bound: -1.0199425, upper bound: 0.9918077
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 12.85
Output dim: 1, lower bound: -1.0199425, upper bound: 1.0088761
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 12.85
Output dim: 1, lower bound: -0.9858257, upper bound: 0.9998352
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 12.85
Output dim: 1, lower bound: -0.9858257, upper bound: 1.0171229
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 12.85
Output dim: 1, lower bound: -1.0063422, upper bound: 0.9998351
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 12.85
Output dim: 1, lower bound: -1.0063422, upper bound: 1.0171227
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 12.85
Output dim: 1, lower bound: -0.9653021, upper bound: 0.9880981
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 12.85
Output dim: 1, lower bound: -0.9653665, upper bound: 0.9853031
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 12.85
Output dim: 1, lower bound: -1.0004485, upper bound: 0.9814950
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 12.85
Output dim: 1, lower bound: -1.0004485, upper bound: 0.9992031
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.85
Output dim: 1, lower bound: -0.9946546, upper bound: 1.0088759
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 12.85
Output dim: 1, lower bound: -1.0142587, upper bound: 1.0088770
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.85
Output dim: 1, lower bound: -0.9946546, upper bound: 1.0088762
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 12.85
Output dim: 1, lower bound: -1.0142587, upper bound: 1.0088761
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.85
Output dim: 1, lower bound: -0.9754732, upper bound: 0.9992032
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 12.85
Output dim: 1, lower bound: -0.9957327, upper bound: 0.9992030
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.85
Output dim: 1, lower bound: -0.9754732, upper bound: 0.9992034
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 12.85
Output dim: 1, lower bound: -0.9957327, upper bound: 0.9992030
Binary search (step 0): status=Status.UNKNOWN, k_low=4, k_high=12, k_mid=8, eps_mid=0.0312500, abs_max=1.49924635887146
rel_dist={1: [-1.0866450021547305, 1.0866453172123824]}

## Binary search (step 1) starts
Candidate k: 5, corresponding eps: 0.0195312


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.38 seconds

### Candidate
type: A, layer: 3, pos: 655

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8420163, upper bound: 0.8436313
time: 4.75 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8420163, upper bound: 0.8398575
time: 8.48 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 13.63 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 13.63
Output dim: 1, lower bound: -0.8420163, upper bound: 0.8436313
IS_A2, status: Status.UNKNOWN, split count: 1, time: 13.63
Output dim: 1, lower bound: -0.8420163, upper bound: 0.8398575

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -7.3354440, -5.1043673, -7.3403769, -5.1030235, -1.8919735, 1.8998728
1: 1.9376824, 3.5761168, 1.9372885, 3.5857427, -1.3030910, 1.2987347
2: -4.9555454, -3.2822270, -4.9615331, -3.2762694, -1.2228515, 1.2190282
3: -11.0604258, -8.8754654, -11.0772495, -8.8740883, -1.7025366, 1.7194271
4: -5.6192431, -3.8418322, -5.6288843, -3.8395448, -1.5839176, 1.5924227
5: -9.0812569, -7.2970009, -9.0880909, -7.2646313, -1.8166256, 1.7910900
6: -6.5684814, -4.3194942, -6.5650177, -4.2903905, -1.6871624, 1.6600306
7: -8.8186340, -7.3928752, -8.8510590, -7.3773441, -1.2139270, 1.2247140
8: 0.9968667, 2.5568495, 0.9745913, 2.5484843, -1.2386153, 1.2638237
9: -9.4914379, -7.3977928, -9.4927759, -7.3947368, -1.6710739, 1.6722445

Time for backsubstitution: 5.67 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 1683
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.36 seconds

### Candidate
type: B, layer: 3, pos: 1683

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8033213, upper bound: 0.8103644
time: 4.06 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8052034, upper bound: 0.8057954
time: 4.35 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -7.3381610, -5.1030903, -7.3412457, -5.1029596, -1.9159737, 1.9001546
1: 1.9398844, 3.5777798, 1.9362400, 3.5890326, -1.3044631, 1.3027370
2: -4.9597774, -3.2774539, -4.9621353, -3.2754836, -1.2263949, 1.2225351
3: -11.0695000, -8.8761969, -11.0800304, -8.8735428, -1.7109084, 1.7195420
4: -5.6239738, -3.8399706, -5.6305523, -3.8394473, -1.5823231, 1.6014194
5: -9.0876074, -7.2712379, -9.0882244, -7.2591839, -1.8284235, 1.8169866
6: -6.5642338, -4.3038802, -6.5653353, -4.2852068, -1.6773717, 1.6913834
7: -8.8500109, -7.3786354, -8.8574305, -7.3770838, -1.2155910, 1.2494316
8: 0.9892220, 2.5483055, 0.9680390, 2.5485387, -1.2828555, 1.2712362
9: -9.4917717, -7.3957472, -9.4929600, -7.3942938, -1.6793656, 1.6746783

Time for backsubstitution: 5.66 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 1683
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.39 seconds

### Candidate
type: B, layer: 3, pos: 655

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8398572, upper bound: 0.8420163
time: 4.53 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8398572, upper bound: 0.8420144
time: 5.60 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 16.19 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 16.19
Output dim: 1, lower bound: -0.8033213, upper bound: 0.8103644
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 16.19
Output dim: 1, lower bound: -0.8052034, upper bound: 0.8057954
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 16.19
Output dim: 1, lower bound: -0.8398572, upper bound: 0.8420163
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 16.19
Output dim: 1, lower bound: -0.8398572, upper bound: 0.8420144

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -7.3354440, -5.1043673, -7.3396072, -5.1034346, -1.8912349, 1.8992000
1: 1.9376824, 3.5761168, 1.9381204, 3.5690007, -1.2844214, 1.2979312
2: -4.9555454, -3.2822270, -4.9592695, -3.2777996, -1.2217429, 1.2099349
3: -11.0604258, -8.8754654, -11.0763197, -8.8757401, -1.7010975, 1.7173581
4: -5.6192431, -3.8418322, -5.6043134, -3.8399904, -1.5818071, 1.5646186
5: -9.0812569, -7.2970009, -9.0834141, -7.2648740, -1.8163829, 1.7864132
6: -6.5684814, -4.3194942, -6.5640244, -4.2924299, -1.6818643, 1.6591101
7: -8.8186340, -7.3928752, -8.8491993, -7.3778734, -1.2127342, 1.2145488
8: 0.9968667, 2.5568495, 0.9894061, 2.5475802, -1.2377367, 1.2541747
9: -9.4914379, -7.3977928, -9.4517584, -7.3951087, -1.6707020, 1.6226165

Time for backsubstitution: 5.67 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.39 seconds

### Candidate
type: A, layer: 3, pos: 1683

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8025458, upper bound: 0.8023512
time: 4.08 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8025458, upper bound: 0.8057955
time: 3.95 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -7.3348346, -5.1050401, -7.3414392, -5.1059074, -1.8901768, 1.9012508
1: 1.9384165, 3.5679140, 1.8638129, 3.5542350, -1.2859669, 1.3779438
2: -4.9457450, -3.2826231, -4.9238777, -3.2706332, -1.2640131, 1.2046604
3: -11.0538721, -8.8762894, -11.0489960, -8.8758707, -1.7029085, 1.7032876
4: -5.6040249, -3.8420117, -5.5737705, -3.7479429, -1.6798158, 1.5984371
5: -9.0790882, -7.2974286, -9.0873680, -7.2663302, -1.8127580, 1.7899394
6: -6.5675273, -4.3412762, -6.5591340, -4.3812361, -1.6454473, 1.6830411
7: -8.7963371, -7.3931465, -8.7512827, -7.3784132, -1.2679532, 1.1972814
8: 1.0067916, 2.5565448, 1.0181971, 2.6162462, -1.2924368, 1.2530549
9: -9.4619932, -7.3978658, -9.3768768, -7.2014050, -1.8926587, 1.6289232

Time for backsubstitution: 5.68 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.37 seconds

### Candidate
type: A, layer: 3, pos: 1683

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8052034, upper bound: 0.8023515
time: 3.99 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8052034, upper bound: 0.8057955
time: 3.80 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -7.3381610, -5.1030903, -7.3354440, -5.1043673, -1.8878541, 1.8861134
1: 1.9398844, 3.5777798, 1.9376824, 3.5761168, -1.2966075, 1.2921867
2: -4.9597774, -3.2774539, -4.9555454, -3.2822270, -1.2179549, 1.2214699
3: -11.0695000, -8.8761969, -11.0604258, -8.8754654, -1.7113423, 1.7009830
4: -5.6239738, -3.8399706, -5.6192431, -3.8418322, -1.5907497, 1.5835648
5: -9.0876074, -7.2712379, -9.0812569, -7.2970009, -1.7906065, 1.8100190
6: -6.5642338, -4.3038802, -6.5684814, -4.3194942, -1.6581903, 1.6598144
7: -8.8500109, -7.3786354, -8.8186340, -7.3928752, -1.2237005, 1.2133763
8: 0.9892220, 2.5483055, 0.9968667, 2.5568495, -1.2521317, 1.2339714
9: -9.4917717, -7.3957472, -9.4914379, -7.3977928, -1.6686292, 1.6697049

Time for backsubstitution: 5.78 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.39 seconds

### Candidate
type: A, layer: 3, pos: 1683

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8067730, upper bound: 0.8033220
time: 4.04 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8026066, upper bound: 0.8052044
time: 3.88 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -7.3381610, -5.1030903, -7.3381610, -5.1030903, -1.9104438, 1.9104435
1: 1.9398844, 3.5777798, 1.9398844, 3.5777798, -1.2995477, 1.2995477
2: -4.9597774, -3.2774539, -4.9597774, -3.2774539, -1.2211161, 1.2211161
3: -11.0695000, -8.8761969, -11.0695000, -8.8761969, -1.7088304, 1.7088304
4: -5.6239738, -3.8399706, -5.6239738, -3.8399706, -1.5818892, 1.5818892
5: -9.0876074, -7.2712379, -9.0876074, -7.2712379, -1.8163695, 1.8163695
6: -6.5642338, -4.3038802, -6.5642338, -4.3038802, -1.6900082, 1.6900082
7: -8.8500109, -7.3786354, -8.8500109, -7.3786354, -1.2149096, 1.2149096
8: 0.9892220, 2.5483055, 0.9892220, 2.5483055, -1.2816334, 1.2816336
9: -9.4917717, -7.3957472, -9.4917717, -7.3957472, -1.6770434, 1.6770432

Time for backsubstitution: 5.62 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.36 seconds

### Candidate
type: A, layer: 3, pos: 1683

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8067730, upper bound: 0.8033220
time: 4.23 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8026066, upper bound: 0.8052030
time: 4.26 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 14.49 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 14.49
Output dim: 1, lower bound: -0.8025458, upper bound: 0.8023512
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 14.49
Output dim: 1, lower bound: -0.8025458, upper bound: 0.8057955
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 14.49
Output dim: 1, lower bound: -0.8052034, upper bound: 0.8023515
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 14.49
Output dim: 1, lower bound: -0.8052034, upper bound: 0.8057955
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 14.49
Output dim: 1, lower bound: -0.8067730, upper bound: 0.8033220
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 14.49
Output dim: 1, lower bound: -0.8026066, upper bound: 0.8052044
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 14.49
Output dim: 1, lower bound: -0.8067730, upper bound: 0.8033220
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 14.49
Output dim: 1, lower bound: -0.8026066, upper bound: 0.8052030

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -7.3346786, -5.1047831, -7.3396072, -5.1034346, -1.8905625, 1.8984909
1: 1.9385400, 3.5586948, 1.9381204, 3.5690007, -1.2836342, 1.2792616
2: -4.9532833, -3.2837548, -4.9592695, -3.2777996, -1.2122729, 1.2088621
3: -11.0594978, -8.8771133, -11.0763197, -8.8757401, -1.6990290, 1.7159405
4: -5.5946770, -3.8422813, -5.6043134, -3.8399904, -1.5544910, 1.5625091
5: -9.0767145, -7.2972412, -9.0834141, -7.2648740, -1.8118405, 1.7861729
6: -6.5674729, -4.3215342, -6.5640244, -4.2924299, -1.6809430, 1.6538115
7: -8.8161144, -7.3933764, -8.8491993, -7.3778734, -1.2018452, 1.2134511
8: 1.0123816, 2.5559411, 0.9894061, 2.5475802, -1.2268012, 1.2532480
9: -9.4507437, -7.3981647, -9.4517584, -7.3951087, -1.6208487, 1.6222525

Time for backsubstitution: 5.64 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.37 seconds

### Candidate
type: B, layer: 3, pos: 655

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8005481, upper bound: 0.8103635
time: 4.04 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8005481, upper bound: 0.8103654
time: 4.39 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -7.3365030, -5.1072540, -7.3396072, -5.1034346, -1.8929353, 1.8978300
1: 1.8596368, 3.5449114, 1.9381204, 3.5690007, -1.3710680, 1.2721589
2: -4.9189172, -3.2766373, -4.9592695, -3.2777996, -1.2170496, 1.2526271
3: -11.0327139, -8.8772326, -11.0763197, -8.8757401, -1.6807141, 1.7224636
4: -5.5671411, -3.7495725, -5.6043134, -3.8399904, -1.5392790, 1.6751490
5: -9.0803432, -7.2987070, -9.0834141, -7.2648740, -1.8154693, 1.7847071
6: -6.5590625, -4.4093394, -6.5640244, -4.2924299, -1.7235022, 1.6027775
7: -8.7222109, -7.3930678, -8.8491993, -7.3778734, -1.1711106, 1.2894621
8: 1.0360909, 2.6289511, 0.9894061, 2.5475802, -1.1977742, 1.3146877
9: -9.3759394, -7.2045193, -9.4517584, -7.3951087, -1.6095700, 1.8604479

Time for backsubstitution: 5.65 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.36 seconds

### Candidate
type: B, layer: 3, pos: 655

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8005481, upper bound: 0.8103644
time: 6.03 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8005481, upper bound: 0.8103633
time: 4.38 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -7.3346786, -5.1047831, -7.3414392, -5.1059074, -1.8900681, 1.9008646
1: 1.9385400, 3.5586948, 1.8638129, 3.5542350, -1.2771845, 1.3663509
2: -4.9532833, -3.2837548, -4.9238777, -3.2706332, -1.2562048, 1.2120235
3: -11.0594978, -8.8771133, -11.0489960, -8.8758707, -1.7059131, 1.6976662
4: -5.5946770, -3.8422813, -5.5737705, -3.7479429, -1.6683421, 1.5473700
5: -9.0767145, -7.2972412, -9.0873680, -7.2663302, -1.8103843, 1.7901268
6: -6.5674729, -4.3215342, -6.5591340, -4.3812361, -1.6282372, 1.6949754
7: -8.8161144, -7.3933764, -8.7512827, -7.3784132, -1.2704041, 1.1793652
8: 1.0123816, 2.5559411, 1.0181971, 2.6162462, -1.2929437, 1.2189846
9: -9.4507437, -7.3981647, -9.3768768, -7.2014050, -1.8593640, 1.6102245

Time for backsubstitution: 5.63 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.40 seconds

### Candidate
type: B, layer: 3, pos: 655

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7999489, upper bound: 0.8023532
time: 4.16 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7999489, upper bound: 0.8023532
time: 4.21 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -7.3365030, -5.1072540, -7.3414392, -5.1059074, -1.8915806, 1.8994193
1: 1.8596368, 3.5449114, 1.8638129, 3.5542350, -1.3145752, 1.3094506
2: -4.9189172, -3.2766373, -4.9238777, -3.2706332, -1.2243705, 1.2197843
3: -11.0327139, -8.8772326, -11.0489960, -8.8758707, -1.6888556, 1.7059770
4: -5.5671411, -3.7495725, -5.5737705, -3.7479429, -1.5830188, 1.5907576
5: -9.0803432, -7.2987070, -9.0873680, -7.2663302, -1.8140130, 1.7886610
6: -6.5590625, -4.4093394, -6.5591340, -4.3812361, -1.6532583, 1.6240876
7: -8.7222109, -7.3930678, -8.7512827, -7.3784132, -1.1854510, 1.1976883
8: 1.0360909, 2.6289511, 1.0181971, 2.6162462, -1.2356534, 1.2608249
9: -9.3759394, -7.2045193, -9.3768768, -7.2014050, -1.6488056, 1.6497540

Time for backsubstitution: 5.65 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.37 seconds

### Candidate
type: B, layer: 3, pos: 655

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7999489, upper bound: 0.8057954
time: 4.02 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7999489, upper bound: 0.8057967
time: 3.93 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -7.3373909, -5.1035042, -7.3354440, -5.1043673, -1.8871789, 1.8854048
1: 1.9407153, 3.5610929, 1.9376824, 3.5761168, -1.2958026, 1.2739382
2: -4.9575157, -3.2789927, -4.9555454, -3.2822270, -1.2088611, 1.2203758
3: -11.0687046, -8.8778534, -11.0604258, -8.8754654, -1.7092757, 1.6995363
4: -5.5993242, -3.8404155, -5.6192431, -3.8418322, -1.5629492, 1.5814576
5: -9.0829248, -7.2714796, -9.0812569, -7.2970009, -1.7859240, 1.8097773
6: -6.5632372, -4.3059192, -6.5684814, -4.3194942, -1.6572704, 1.6551464
7: -8.8481512, -7.3791618, -8.8186340, -7.3928752, -1.2135339, 1.2121832
8: 1.0040431, 2.5473943, 0.9968667, 2.5568495, -1.2417159, 1.2330956
9: -9.4507761, -7.3961210, -9.4914379, -7.3977928, -1.6188984, 1.6693308

Time for backsubstitution: 5.65 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 1683
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.39 seconds

### Candidate
type: B, layer: 3, pos: 1683

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8023532, upper bound: 0.8025463
time: 4.05 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8023512, upper bound: 0.8025465
time: 4.39 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -7.3392220, -5.1059752, -7.3348346, -5.1050401, -1.8892198, 1.8842578
1: 1.8664083, 3.5426223, 1.9384165, 3.5679140, -1.3758683, 1.2751541
2: -4.9221191, -3.2719152, -4.9457450, -3.2826231, -1.2035875, 1.2626557
3: -11.0405436, -8.8780537, -11.0538721, -8.8762894, -1.6952996, 1.7013450
4: -5.5685873, -3.7483337, -5.6040249, -3.8420117, -1.5967407, 1.6794333
5: -9.0868692, -7.2729349, -9.0790882, -7.2974286, -1.7894406, 1.8061533
6: -6.5583653, -4.3931675, -6.5675273, -4.3412762, -1.6808071, 1.6171870
7: -8.7508974, -7.3796463, -8.7963371, -7.3931465, -1.1962678, 1.2674356
8: 1.0328178, 2.6160669, 1.0067916, 2.5565448, -1.2412095, 1.2870760
9: -9.3758202, -7.2023268, -9.4619932, -7.3978658, -1.6252949, 1.8912256

Time for backsubstitution: 5.70 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 1683
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.37 seconds

### Candidate
type: B, layer: 3, pos: 1683

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8023512, upper bound: 0.8052049
time: 4.76 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8023512, upper bound: 0.8052040
time: 4.56 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -7.3373909, -5.1035042, -7.3381610, -5.1030903, -1.9097710, 1.9097345
1: 1.9407153, 3.5610929, 1.9398844, 3.5777798, -1.2987523, 1.2808783
2: -4.9575157, -3.2789927, -4.9597774, -3.2774539, -1.2116499, 1.2200136
3: -11.0687046, -8.8778534, -11.0695000, -8.8761969, -1.7067628, 1.7074151
4: -5.5993242, -3.8404155, -5.6239738, -3.8399706, -1.5544143, 1.5797811
5: -9.0829248, -7.2714796, -9.0876074, -7.2712379, -1.8116870, 1.8161278
6: -6.5632372, -4.3059192, -6.5642338, -4.3038802, -1.6890874, 1.6846073
7: -8.8481512, -7.3791618, -8.8500109, -7.3786354, -1.2040224, 1.2137990
8: 1.0040431, 2.5473943, 0.9892220, 2.5483055, -1.2704525, 1.2806995
9: -9.4507761, -7.3961210, -9.4917717, -7.3957472, -1.6275272, 1.6766691

Time for backsubstitution: 5.64 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 1683
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.37 seconds

### Candidate
type: B, layer: 3, pos: 1683

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7999494, upper bound: 0.8025462
time: 3.97 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7999494, upper bound: 0.8025470
time: 4.27 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -7.3392220, -5.1059752, -7.3375516, -5.1037636, -1.9118185, 1.9086442
1: 1.8664083, 3.5426223, 1.9405994, 3.5687561, -1.3789277, 1.2820168
2: -4.9221191, -3.2719152, -4.9497614, -3.2778482, -1.2076504, 1.2619646
3: -11.0405436, -8.8780537, -11.0627842, -8.8770409, -1.6926074, 1.7088685
4: -5.5685873, -3.7483337, -5.6087151, -3.8401475, -1.5855141, 1.6772990
5: -9.0868692, -7.2729349, -9.0855198, -7.2716627, -1.8152065, 1.8125849
6: -6.5583653, -4.3931675, -6.5632963, -4.3256330, -1.7152538, 1.6500611
7: -8.7508974, -7.3796463, -8.8269386, -7.3789148, -1.1871066, 1.2762041
8: 1.0328178, 2.6160669, 0.9999394, 2.5480042, -1.2704203, 1.3310473
9: -9.3758202, -7.2023268, -9.4621553, -7.3958206, -1.6314359, 1.8985896

Time for backsubstitution: 5.77 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 1683
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.40 seconds

### Candidate
type: B, layer: 3, pos: 1683

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7999494, upper bound: 0.8052050
time: 4.25 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7999494, upper bound: 0.8052054
time: 4.60 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 15.04 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 15.04
Output dim: 1, lower bound: -0.8005481, upper bound: 0.8103635
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 15.04
Output dim: 1, lower bound: -0.8005481, upper bound: 0.8103654
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 15.04
Output dim: 1, lower bound: -0.8005481, upper bound: 0.8103644
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 15.04
Output dim: 1, lower bound: -0.8005481, upper bound: 0.8103633
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 15.04
Output dim: 1, lower bound: -0.7999489, upper bound: 0.8023532
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 15.04
Output dim: 1, lower bound: -0.7999489, upper bound: 0.8023532
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 15.04
Output dim: 1, lower bound: -0.7999489, upper bound: 0.8057954
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 15.04
Output dim: 1, lower bound: -0.7999489, upper bound: 0.8057967
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 15.04
Output dim: 1, lower bound: -0.8023532, upper bound: 0.8025463
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 15.04
Output dim: 1, lower bound: -0.8023512, upper bound: 0.8025465
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 15.04
Output dim: 1, lower bound: -0.8023512, upper bound: 0.8052049
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 15.04
Output dim: 1, lower bound: -0.8023512, upper bound: 0.8052040
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 15.04
Output dim: 1, lower bound: -0.7999494, upper bound: 0.8025462
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 15.04
Output dim: 1, lower bound: -0.7999494, upper bound: 0.8025470
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 15.04
Output dim: 1, lower bound: -0.7999494, upper bound: 0.8052050
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 15.04
Output dim: 1, lower bound: -0.7999494, upper bound: 0.8052054

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -7.3346786, -5.1047831, -7.3346786, -5.1047831, -1.8857837, 1.8857839
1: 1.9385400, 3.5586948, 1.9385400, 3.5586948, -1.2770019, 1.2770019
2: -4.9532833, -3.2837548, -4.9532833, -3.2837548, -1.2049198, 1.2049199
3: -11.0594978, -8.8771133, -11.0594978, -8.8771133, -1.7002482, 1.7002482
4: -5.5946770, -3.8422813, -5.5946770, -3.8422813, -1.5478592, 1.5478594
5: -9.0767145, -7.2972412, -9.0767145, -7.2972412, -1.7794733, 1.7794733
6: -6.5674729, -4.3215342, -6.5674729, -4.3215342, -1.6642599, 1.6642599
7: -8.8161144, -7.3933764, -8.8161144, -7.3933764, -1.1830931, 1.1830931
8: 1.0123816, 2.5559411, 1.0123816, 2.5559411, -1.2232318, 1.2232318
9: -9.4507437, -7.3981647, -9.4507437, -7.3981647, -1.6172845, 1.6172843

Time for backsubstitution: 5.72 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.39 seconds

### Candidate
type: A, layer: 3, pos: 402

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8120473, upper bound: 0.8264358
time: 4.46 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8234692, upper bound: 0.8264363
time: 4.39 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -7.3346786, -5.1047831, -7.3373909, -5.1035042, -1.8847322, 1.8864698
1: 1.9385400, 3.5586948, 1.9407153, 3.5610929, -1.2731509, 1.2771327
2: -4.9532833, -3.2837548, -4.9575157, -3.2789927, -1.2109058, 1.2077882
3: -11.0594978, -8.8771133, -11.0687046, -8.8778534, -1.6974678, 1.7078571
4: -5.5946770, -3.8422813, -5.5993242, -3.8404155, -1.5541415, 1.5608399
5: -9.0767145, -7.2972412, -9.0829248, -7.2714796, -1.8052349, 1.7856836
6: -6.5674729, -4.3215342, -6.5632372, -4.3059192, -1.6542251, 1.6519716
7: -8.8161144, -7.3933764, -8.8481512, -7.3791618, -1.2012947, 1.2124362
8: 1.0123816, 2.5559411, 1.0040431, 2.5473943, -1.2221603, 1.2407892
9: -9.4507437, -7.3981647, -9.4507761, -7.3961210, -1.6194780, 1.6185348

Time for backsubstitution: 5.76 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.38 seconds

### Candidate
type: A, layer: 3, pos: 402

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8120473, upper bound: 0.8264353
time: 7.91 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8234692, upper bound: 0.8264370
time: 4.41 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -7.3365030, -5.1072540, -7.3346786, -5.1047831, -1.8881569, 1.8851230
1: 1.8596368, 3.5449114, 1.9385400, 3.5586948, -1.3644352, 1.2698991
2: -4.9189172, -3.2766373, -4.9532833, -3.2837548, -1.2096968, 1.2486849
3: -11.0327139, -8.8772326, -11.0594978, -8.8771133, -1.6819334, 1.7067719
4: -5.5671411, -3.7495725, -5.5946770, -3.8422813, -1.5326467, 1.6604989
5: -9.0803432, -7.2987070, -9.0767145, -7.2972412, -1.7831020, 1.7780075
6: -6.5590625, -4.4093394, -6.5674729, -4.3215342, -1.7068191, 1.6132259
7: -8.7222109, -7.3930678, -8.8161144, -7.3933764, -1.1523581, 1.2591040
8: 1.0360909, 2.6289511, 1.0123816, 2.5559411, -1.1942050, 1.2846718
9: -9.3759394, -7.2045193, -9.4507437, -7.3981647, -1.6060054, 1.8554797

Time for backsubstitution: 5.75 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.39 seconds

### Candidate
type: A, layer: 3, pos: 402

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7798646, upper bound: 0.8030674
time: 4.41 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7941533, upper bound: 0.8030675
time: 4.43 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -7.3365030, -5.1072540, -7.3373909, -5.1035042, -1.8871055, 1.8858089
1: 1.8596368, 3.5449114, 1.9407153, 3.5610929, -1.3605843, 1.2700300
2: -4.9189172, -3.2766373, -4.9575157, -3.2789927, -1.2156825, 1.2515531
3: -11.0327139, -8.8772326, -11.0687046, -8.8778534, -1.6791534, 1.7143807
4: -5.5671411, -3.7495725, -5.5993242, -3.8404155, -1.5389295, 1.6734793
5: -9.0803432, -7.2987070, -9.0829248, -7.2714796, -1.8088636, 1.7842178
6: -6.5590625, -4.4093394, -6.5632372, -4.3059192, -1.6967843, 1.6009376
7: -8.7222109, -7.3930678, -8.8481512, -7.3791618, -1.1705601, 1.2884471
8: 1.0360909, 2.6289511, 1.0040431, 2.5473943, -1.1931334, 1.3022292
9: -9.3759394, -7.2045193, -9.4507761, -7.3961210, -1.6081994, 1.8567305

Time for backsubstitution: 5.69 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.38 seconds

### Candidate
type: A, layer: 3, pos: 402

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7798646, upper bound: 0.8030690
time: 4.44 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7941533, upper bound: 0.8030670
time: 4.88 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -7.3346786, -5.1047831, -7.3365030, -5.1072540, -1.8851228, 1.8881569
1: 1.9385400, 3.5586948, 1.8596368, 3.5449114, -1.2698994, 1.3644350
2: -4.9532833, -3.2837548, -4.9189172, -3.2766373, -1.2486849, 1.2096968
3: -11.0594978, -8.8771133, -11.0327139, -8.8772326, -1.7067719, 1.6819339
4: -5.5946770, -3.8422813, -5.5671411, -3.7495725, -1.6604991, 1.5326467
5: -9.0767145, -7.2972412, -9.0803432, -7.2987070, -1.7780075, 1.7831020
6: -6.5674729, -4.3215342, -6.5590625, -4.4093394, -1.6132259, 1.7068188
7: -8.8161144, -7.3933764, -8.7222109, -7.3930678, -1.2591038, 1.1523581
8: 1.0123816, 2.5559411, 1.0360909, 2.6289511, -1.2846718, 1.1942050
9: -9.4507437, -7.3981647, -9.3759394, -7.2045193, -1.8554797, 1.6060057

Time for backsubstitution: 5.63 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.37 seconds

### Candidate
type: A, layer: 3, pos: 402

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7865145, upper bound: 0.7962389
time: 4.29 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7997889, upper bound: 0.7962386
time: 4.17 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -7.3346786, -5.1047831, -7.3392220, -5.1059752, -1.8840728, 1.8888338
1: 1.9385400, 3.5586948, 1.8664083, 3.5426223, -1.2641187, 1.3642757
2: -4.9532833, -3.2837548, -4.9221191, -3.2719152, -1.2548475, 1.2109210
3: -11.0594978, -8.8771133, -11.0405436, -8.8780537, -1.7043495, 1.6896863
4: -5.5946770, -3.8422813, -5.5685873, -3.7483337, -1.6679597, 1.5458288
5: -9.0767145, -7.2972412, -9.0868692, -7.2729349, -1.8037796, 1.7896280
6: -6.5674729, -4.3215342, -6.5583653, -4.3931675, -1.5980668, 1.6927419
7: -8.8161144, -7.3933764, -8.7508974, -7.3796463, -1.2698870, 1.1798427
8: 1.0123816, 2.5559411, 1.0328178, 2.6160669, -1.2875829, 1.2118289
9: -9.4507437, -7.3981647, -9.3758202, -7.2023268, -1.8579319, 1.6068015

Time for backsubstitution: 5.72 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.38 seconds

### Candidate
type: A, layer: 3, pos: 402

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7865145, upper bound: 0.7962401
time: 4.18 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7997889, upper bound: 0.7962379
time: 4.62 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -7.3365030, -5.1072540, -7.3365030, -5.1072540, -1.8867135, 1.8867135
1: 1.8596368, 3.5449114, 1.8596368, 3.5449114, -1.3076015, 1.3076015
2: -4.9189172, -3.2766373, -4.9189172, -3.2766373, -1.2170823, 1.2170823
3: -11.0327139, -8.8772326, -11.0327139, -8.8772326, -1.6902800, 1.6902800
4: -5.5671411, -3.7495725, -5.5671411, -3.7495725, -1.5763922, 1.5763922
5: -9.0803432, -7.2987070, -9.0803432, -7.2987070, -1.7816362, 1.7816362
6: -6.5590625, -4.4093394, -6.5590625, -4.4093394, -1.6367359, 1.6367359
7: -8.7222109, -7.3930678, -8.7222109, -7.3930678, -1.1669478, 1.1669478
8: 1.0360909, 2.6289511, 1.0360909, 2.6289511, -1.2316256, 1.2316253
9: -9.3759394, -7.2045193, -9.3759394, -7.2045193, -1.6449797, 1.6449797

Time for backsubstitution: 5.65 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.37 seconds

### Candidate
type: A, layer: 3, pos: 402

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7789239, upper bound: 0.7989638
time: 4.38 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7936031, upper bound: 0.7989641
time: 4.07 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -7.3365030, -5.1072540, -7.3392220, -5.1059752, -1.8856616, 1.8873906
1: 1.8596368, 3.5449114, 1.8664083, 3.5426223, -1.3037627, 1.3073757
2: -4.9189172, -3.2766373, -4.9221191, -3.2719152, -1.2230861, 1.2187113
3: -11.0327139, -8.8772326, -11.0405436, -8.8780537, -1.6872926, 1.6979885
4: -5.5671411, -3.7495725, -5.5685873, -3.7483337, -1.5826526, 1.5890605
5: -9.0803432, -7.2987070, -9.0868692, -7.2729349, -1.8074083, 1.7881622
6: -6.5590625, -4.4093394, -6.5583653, -4.3931675, -1.6249981, 1.6220615
7: -8.7222109, -7.3930678, -8.7508974, -7.3796463, -1.1848874, 1.1966746
8: 1.0360909, 2.6289511, 1.0328178, 2.6160669, -1.2310164, 1.2489798
9: -9.3759394, -7.2045193, -9.3758202, -7.2023268, -1.6473465, 1.6461258

Time for backsubstitution: 5.68 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.38 seconds

### Candidate
type: A, layer: 3, pos: 402

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7789239, upper bound: 0.7989630
time: 4.46 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7936031, upper bound: 0.7989634
time: 4.59 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -7.3373909, -5.1035042, -7.3346786, -5.1047831, -1.8864698, 1.8847325
1: 1.9407153, 3.5610929, 1.9385400, 3.5586948, -1.2771330, 1.2731509
2: -4.9575157, -3.2789927, -4.9532833, -3.2837548, -1.2077882, 1.2109058
3: -11.0687046, -8.8778534, -11.0594978, -8.8771133, -1.7078571, 1.6974678
4: -5.5993242, -3.8404155, -5.5946770, -3.8422813, -1.5608397, 1.5541418
5: -9.0829248, -7.2714796, -9.0767145, -7.2972412, -1.7856836, 1.8052349
6: -6.5632372, -4.3059192, -6.5674729, -4.3215342, -1.6519713, 1.6542253
7: -8.8481512, -7.3791618, -8.8161144, -7.3933764, -1.2124362, 1.2012949
8: 1.0040431, 2.5473943, 1.0123816, 2.5559411, -1.2407892, 1.2221603
9: -9.4507761, -7.3961210, -9.4507437, -7.3981647, -1.6185348, 1.6194777

Time for backsubstitution: 5.65 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.39 seconds

### Candidate
type: A, layer: 3, pos: 402

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7896606, upper bound: 0.7969805
time: 4.33 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8030669, upper bound: 0.7969813
time: 4.21 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -7.3373909, -5.1035042, -7.3365030, -5.1072540, -1.8858089, 1.8871055
1: 1.9407153, 3.5610929, 1.8596368, 3.5449114, -1.2700300, 1.3605843
2: -4.9575157, -3.2789927, -4.9189172, -3.2766373, -1.2515531, 1.2156825
3: -11.0687046, -8.8778534, -11.0327139, -8.8772326, -1.7143807, 1.6791534
4: -5.5993242, -3.8404155, -5.5671411, -3.7495725, -1.6734791, 1.5389292
5: -9.0829248, -7.2714796, -9.0803432, -7.2987070, -1.7842178, 1.8088636
6: -6.5632372, -4.3059192, -6.5590625, -4.4093394, -1.6009378, 1.6967843
7: -8.8481512, -7.3791618, -8.7222109, -7.3930678, -1.2884469, 1.1705596
8: 1.0040431, 2.5473943, 1.0360909, 2.6289511, -1.3022292, 1.1931334
9: -9.4507761, -7.3961210, -9.3759394, -7.2045193, -1.8567305, 1.6081991

Time for backsubstitution: 5.68 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.39 seconds

### Candidate
type: A, layer: 3, pos: 402

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7896606, upper bound: 0.7969806
time: 4.22 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8030669, upper bound: 0.7969812
time: 4.24 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -7.3392220, -5.1059752, -7.3346786, -5.1047831, -1.8888335, 1.8840728
1: 1.8664083, 3.5426223, 1.9385400, 3.5586948, -1.3642759, 1.2641191
2: -4.9221191, -3.2719152, -4.9532833, -3.2837548, -1.2109208, 1.2548475
3: -11.0405436, -8.8780537, -11.0594978, -8.8771133, -1.6896863, 1.7043495
4: -5.5685873, -3.7483337, -5.5946770, -3.8422813, -1.5458288, 1.6679595
5: -9.0868692, -7.2729349, -9.0767145, -7.2972412, -1.7896280, 1.8037796
6: -6.5583653, -4.3931675, -6.5674729, -4.3215342, -1.6927419, 1.5980668
7: -8.7508974, -7.3796463, -8.8161144, -7.3933764, -1.1798429, 1.2698867
8: 1.0328178, 2.6160669, 1.0123816, 2.5559411, -1.2118287, 1.2875829
9: -9.3758202, -7.2023268, -9.4507437, -7.3981647, -1.6068017, 1.8579316

Time for backsubstitution: 5.76 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.39 seconds

### Candidate
type: A, layer: 3, pos: 402

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7804329, upper bound: 0.7988239
time: 4.19 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7958351, upper bound: 0.7988237
time: 4.12 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -7.3392220, -5.1059752, -7.3365030, -5.1072540, -1.8873906, 1.8856616
1: 1.8664083, 3.5426223, 1.8596368, 3.5449114, -1.3073759, 1.3037627
2: -4.9221191, -3.2719152, -4.9189172, -3.2766373, -1.2187114, 1.2230861
3: -11.0405436, -8.8780537, -11.0327139, -8.8772326, -1.6979890, 1.6872926
4: -5.5685873, -3.7483337, -5.5671411, -3.7495725, -1.5890603, 1.5826526
5: -9.0868692, -7.2729349, -9.0803432, -7.2987070, -1.7881622, 1.8074083
6: -6.5583653, -4.3931675, -6.5590625, -4.4093394, -1.6220613, 1.6249981
7: -8.7508974, -7.3796463, -8.7222109, -7.3930678, -1.1966743, 1.1848874
8: 1.0328178, 2.6160669, 1.0360909, 2.6289511, -1.2489798, 1.2310164
9: -9.3758202, -7.2023268, -9.3759394, -7.2045193, -1.6461260, 1.6473463

Time for backsubstitution: 5.66 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.37 seconds

### Candidate
type: A, layer: 3, pos: 402

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7804329, upper bound: 0.7988234
time: 4.37 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7958351, upper bound: 0.7988228
time: 4.53 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -7.3373909, -5.1035042, -7.3373909, -5.1035042, -1.9090619, 1.9090619
1: 1.9407153, 3.5610929, 1.9407153, 3.5610929, -1.2800829, 1.2800829
2: -4.9575157, -3.2789927, -4.9575157, -3.2789927, -1.2105474, 1.2105474
3: -11.0687046, -8.8778534, -11.0687046, -8.8778534, -1.7053475, 1.7053475
4: -5.5993242, -3.8404155, -5.5993242, -3.8404155, -1.5523062, 1.5523062
5: -9.0829248, -7.2714796, -9.0829248, -7.2714796, -1.8114452, 1.8114452
6: -6.5632372, -4.3059192, -6.5632372, -4.3059192, -1.6836863, 1.6836863
7: -8.8481512, -7.3791618, -8.8481512, -7.3791618, -1.2029121, 1.2029119
8: 1.0040431, 2.5473943, 1.0040431, 2.5473943, -1.2695189, 1.2695189
9: -9.4507761, -7.3961210, -9.4507761, -7.3961210, -1.6271529, 1.6271532

Time for backsubstitution: 5.64 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.37 seconds

### Candidate
type: A, layer: 3, pos: 402

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7864687, upper bound: 0.7969815
time: 4.00 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7997889, upper bound: 0.7969810
time: 4.04 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -7.3373909, -5.1035042, -7.3392220, -5.1059752, -1.9084010, 1.9114332
1: 1.9407153, 3.5610929, 1.8664083, 3.5426223, -1.2730129, 1.3673356
2: -4.9575157, -3.2789927, -4.9221191, -3.2719152, -1.2542121, 1.2152975
3: -11.0687046, -8.8778534, -11.0405436, -8.8780537, -1.7118726, 1.6870012
4: -5.5993242, -3.8404155, -5.5685873, -3.7483337, -1.6654019, 1.5366189
5: -9.0829248, -7.2714796, -9.0868692, -7.2729349, -1.8099899, 1.8153896
6: -6.5632372, -4.3059192, -6.5583653, -4.3931675, -1.6335073, 1.7265615
7: -8.8481512, -7.3791618, -8.7508974, -7.3796463, -1.2788594, 1.1722937
8: 1.0040431, 2.5473943, 1.0328178, 2.6160669, -1.3312368, 1.2409408
9: -9.4507761, -7.3961210, -9.3758202, -7.2023268, -1.8656309, 1.6145766

Time for backsubstitution: 5.64 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.37 seconds

### Candidate
type: A, layer: 3, pos: 402

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7864687, upper bound: 0.7969811
time: 4.31 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7997889, upper bound: 0.7969812
time: 4.23 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -7.3392220, -5.1059752, -7.3373909, -5.1035042, -1.9114332, 1.9084015
1: 1.8664083, 3.5426223, 1.9407153, 3.5610929, -1.3673356, 1.2730126
2: -4.9221191, -3.2719152, -4.9575157, -3.2789927, -1.2152975, 1.2542121
3: -11.0405436, -8.8780537, -11.0687046, -8.8778534, -1.6870012, 1.7118726
4: -5.5685873, -3.7483337, -5.5993242, -3.8404155, -1.5366192, 1.6654024
5: -9.0868692, -7.2729349, -9.0829248, -7.2714796, -1.8153896, 1.8099899
6: -6.5583653, -4.3931675, -6.5632372, -4.3059192, -1.7265615, 1.6335070
7: -8.7508974, -7.3796463, -8.8481512, -7.3791618, -1.1722934, 1.2788591
8: 1.0328178, 2.6160669, 1.0040431, 2.5473943, -1.2409410, 1.3312368
9: -9.3758202, -7.2023268, -9.4507761, -7.3961210, -1.6145768, 1.8656309

Time for backsubstitution: 5.66 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.37 seconds

### Candidate
type: A, layer: 3, pos: 402

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7789260, upper bound: 0.7988241
time: 4.04 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7936035, upper bound: 0.7988238
time: 4.14 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -7.3392220, -5.1059752, -7.3392220, -5.1059752, -1.9100451, 1.9100447
1: 1.8664083, 3.5426223, 1.8664083, 3.5426223, -1.3106525, 1.3106523
2: -4.9221191, -3.2719152, -4.9221191, -3.2719152, -1.2223427, 1.2223428
3: -11.0405436, -8.8780537, -11.0405436, -8.8780537, -1.6953015, 1.6953015
4: -5.5685873, -3.7483337, -5.5685873, -3.7483337, -1.5778151, 1.5778153
5: -9.0868692, -7.2729349, -9.0868692, -7.2729349, -1.8139343, 1.8139343
6: -6.5583653, -4.3931675, -6.5583653, -4.3931675, -1.6581445, 1.6581445
7: -8.7508974, -7.3796463, -8.7508974, -7.3796463, -1.1875248, 1.1875248
8: 1.0328178, 2.6160669, 1.0328178, 2.6160669, -1.2782512, 1.2782512
9: -9.3758202, -7.2023268, -9.3758202, -7.2023268, -1.6524005, 1.6524003

Time for backsubstitution: 5.64 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.38 seconds

### Candidate
type: A, layer: 3, pos: 402

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7789239, upper bound: 0.7988235
time: 4.72 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7936031, upper bound: 0.7988238
time: 4.34 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 15.09 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.09
Output dim: 1, lower bound: -0.8120473, upper bound: 0.8264358
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.09
Output dim: 1, lower bound: -0.8234692, upper bound: 0.8264363
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.09
Output dim: 1, lower bound: -0.8120473, upper bound: 0.8264353
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.09
Output dim: 1, lower bound: -0.8234692, upper bound: 0.8264370
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.09
Output dim: 1, lower bound: -0.7798646, upper bound: 0.8030674
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.09
Output dim: 1, lower bound: -0.7941533, upper bound: 0.8030675
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.09
Output dim: 1, lower bound: -0.7798646, upper bound: 0.8030690
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.09
Output dim: 1, lower bound: -0.7941533, upper bound: 0.8030670
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.09
Output dim: 1, lower bound: -0.7865145, upper bound: 0.7962389
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.09
Output dim: 1, lower bound: -0.7997889, upper bound: 0.7962386
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.09
Output dim: 1, lower bound: -0.7865145, upper bound: 0.7962401
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.09
Output dim: 1, lower bound: -0.7997889, upper bound: 0.7962379
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.09
Output dim: 1, lower bound: -0.7789239, upper bound: 0.7989638
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.09
Output dim: 1, lower bound: -0.7936031, upper bound: 0.7989641
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.09
Output dim: 1, lower bound: -0.7789239, upper bound: 0.7989630
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.09
Output dim: 1, lower bound: -0.7936031, upper bound: 0.7989634
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.09
Output dim: 1, lower bound: -0.7896606, upper bound: 0.7969805
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.09
Output dim: 1, lower bound: -0.8030669, upper bound: 0.7969813
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.09
Output dim: 1, lower bound: -0.7896606, upper bound: 0.7969806
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.09
Output dim: 1, lower bound: -0.8030669, upper bound: 0.7969812
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.09
Output dim: 1, lower bound: -0.7804329, upper bound: 0.7988239
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.09
Output dim: 1, lower bound: -0.7958351, upper bound: 0.7988237
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.09
Output dim: 1, lower bound: -0.7804329, upper bound: 0.7988234
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.09
Output dim: 1, lower bound: -0.7958351, upper bound: 0.7988228
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.09
Output dim: 1, lower bound: -0.7864687, upper bound: 0.7969815
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.09
Output dim: 1, lower bound: -0.7997889, upper bound: 0.7969810
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.09
Output dim: 1, lower bound: -0.7864687, upper bound: 0.7969811
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.09
Output dim: 1, lower bound: -0.7997889, upper bound: 0.7969812
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.09
Output dim: 1, lower bound: -0.7789260, upper bound: 0.7988241
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.09
Output dim: 1, lower bound: -0.7936035, upper bound: 0.7988238
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.09
Output dim: 1, lower bound: -0.7789239, upper bound: 0.7988235
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.09
Output dim: 1, lower bound: -0.7936031, upper bound: 0.7988238

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -7.3056631, -5.0999122, -7.3265853, -5.1048870, -1.8650002, 1.8947487
1: 1.9669955, 3.5556262, 1.9470694, 3.5580940, -1.2391915, 1.2551186
2: -4.9469681, -3.2823098, -4.9515848, -3.2844629, -1.1941376, 1.2054296
3: -11.0628061, -8.8965263, -11.0594692, -8.8824587, -1.7000408, 1.6799712
4: -5.5925274, -3.8488555, -5.5935507, -3.8439250, -1.5416894, 1.5379267
5: -9.0700178, -7.3113856, -9.0753517, -7.3010511, -1.7689667, 1.7639661
6: -6.5675230, -4.3262696, -6.5667105, -4.3228936, -1.6655593, 1.6588326
7: -8.8376179, -7.4093542, -8.8160324, -7.3976793, -1.1816444, 1.1605079
8: 1.0145617, 2.5453753, 1.0131469, 2.5531449, -1.2216096, 1.2134714
9: -9.4554577, -7.4482117, -9.4499121, -7.4116917, -1.6028843, 1.5642436

Time for backsubstitution: 5.64 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.38 seconds

### Candidate
type: B, layer: 3, pos: 423

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8038024, upper bound: 0.8167789
time: 4.64 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8052185, upper bound: 0.8169532
time: 4.67 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -7.3252287, -5.1051140, -7.3344097, -5.1047912, -1.8876352, 1.8851371
1: 1.9471767, 3.5582409, 1.9387870, 3.5586810, -1.2524972, 1.2767618
2: -4.9501371, -3.2865560, -4.9531937, -3.2838321, -1.2133036, 1.1999872
3: -11.0593834, -8.8867168, -11.0594950, -8.8773880, -1.6999021, 1.6872554
4: -5.5903339, -3.8453085, -5.5945511, -3.8423667, -1.5407376, 1.5488262
5: -9.0735254, -7.3005614, -9.0766268, -7.2973347, -1.7761908, 1.7760653
6: -6.5660152, -4.3244324, -6.5674324, -4.3216176, -1.6627555, 1.6645553
7: -8.8158188, -7.4089589, -8.8161068, -7.3938208, -1.1815999, 1.1605511
8: 1.0131907, 2.5515699, 1.0124044, 2.5558038, -1.2221704, 1.2188337
9: -9.4498243, -7.4126263, -9.4507179, -7.3986902, -1.6160889, 1.5767291

Time for backsubstitution: 5.65 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.38 seconds

### Candidate
type: B, layer: 3, pos: 402

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8264376, upper bound: 0.8146742
time: 4.47 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8264376, upper bound: 0.8264376
time: 4.32 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -7.3056631, -5.0999122, -7.3293614, -5.1036081, -1.8639483, 1.8954802
1: 1.9669955, 3.5556262, 1.9492719, 3.5604885, -1.2352533, 1.2555671
2: -4.9469681, -3.2823098, -4.9558296, -3.2797019, -1.2000048, 1.2084333
3: -11.0628061, -8.8965263, -11.0686750, -8.8831768, -1.6973152, 1.6875777
4: -5.5925274, -3.8488555, -5.5981708, -3.8420577, -1.5479693, 1.5508626
5: -9.0700178, -7.3113856, -9.0815630, -7.2752810, -1.7947369, 1.7701774
6: -6.5675230, -4.3262696, -6.5624762, -4.3072643, -1.6555247, 1.6465201
7: -8.8376179, -7.4093542, -8.8480625, -7.3834209, -1.2010691, 1.1898556
8: 1.0145617, 2.5453753, 1.0048251, 2.5446110, -1.2208719, 1.2310305
9: -9.4554577, -7.4482117, -9.4499531, -7.4096375, -1.6050880, 1.5655017

Time for backsubstitution: 5.70 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.38 seconds

### Candidate
type: B, layer: 3, pos: 423

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8072753, upper bound: 0.8167777
time: 4.33 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8084418, upper bound: 0.8169509
time: 4.45 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -7.3252287, -5.1051140, -7.3371229, -5.1035137, -1.8865838, 1.8858228
1: 1.9471767, 3.5582409, 1.9409628, 3.5610805, -1.2505968, 1.2768874
2: -4.9501371, -3.2865560, -4.9574251, -3.2790716, -1.2194984, 1.2028244
3: -11.0593834, -8.8867168, -11.0687017, -8.8781300, -1.6971235, 1.6949635
4: -5.5903339, -3.8453085, -5.5991955, -3.8405020, -1.5470195, 1.5625694
5: -9.0735254, -7.3005614, -9.0828342, -7.2715726, -1.8019528, 1.7822728
6: -6.5660152, -4.3244324, -6.5631948, -4.3060031, -1.6527209, 1.6508381
7: -8.8158188, -7.4089589, -8.8481417, -7.3796062, -1.1997025, 1.1929493
8: 1.0131907, 2.5515699, 1.0040684, 2.5472579, -1.2210827, 1.2375813
9: -9.4498243, -7.4126263, -9.4507504, -7.3966489, -1.6182818, 1.5772829

Time for backsubstitution: 5.67 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.37 seconds

### Candidate
type: B, layer: 3, pos: 402

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8262810, upper bound: 0.8146718
time: 4.22 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8262810, upper bound: 0.8264351
time: 4.12 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -7.3072877, -5.1018953, -7.3265853, -5.1048870, -1.8673019, 1.8940854
1: 1.8873105, 3.5444431, 1.9470694, 3.5580940, -1.3309450, 1.2504997
2: -4.9123201, -3.2756791, -4.9515848, -3.2844629, -1.1984243, 1.2486038
3: -11.0354338, -8.8966808, -11.0594692, -8.8824587, -1.6831508, 1.6865101
4: -5.5645566, -3.7567065, -5.5935507, -3.8439250, -1.5250850, 1.6507134
5: -9.0741472, -7.3128581, -9.0753517, -7.3010511, -1.7730961, 1.7624936
6: -6.5592098, -4.4138160, -6.5667105, -4.3228936, -1.7079272, 1.6086397
7: -8.7437325, -7.4088888, -8.8160324, -7.3976793, -1.1507683, 1.2332368
8: 1.0390034, 2.6184092, 1.0131469, 2.5531449, -1.1909306, 1.2748096
9: -9.3746729, -7.2596254, -9.4499121, -7.4116917, -1.5863128, 1.7967358

Time for backsubstitution: 5.71 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.37 seconds

### Candidate
type: B, layer: 3, pos: 423

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7710668, upper bound: 0.7934539
time: 4.41 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7725162, upper bound: 0.7938135
time: 4.70 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -7.3270020, -5.1075830, -7.3344097, -5.1047912, -1.8899155, 1.8844767
1: 1.8684199, 3.5445313, 1.9387870, 3.5586810, -1.3403850, 1.2696452
2: -4.9158492, -3.2790942, -4.9531937, -3.2838321, -1.2157288, 1.2437339
3: -11.0326090, -8.8868847, -11.0594950, -8.8773880, -1.6815920, 1.6937785
4: -5.5624924, -3.7522728, -5.5945511, -3.8423667, -1.5244117, 1.6618400
5: -9.0771637, -7.3020205, -9.0766268, -7.2973347, -1.7798290, 1.7746062
6: -6.5576620, -4.4125118, -6.5674324, -4.3216176, -1.7054248, 1.6141202
7: -8.7219315, -7.4087892, -8.8161068, -7.3938208, -1.1508448, 1.2332389
8: 1.0368590, 2.6243343, 1.0124044, 2.5558038, -1.1932485, 1.2800598
9: -9.3750725, -7.2159300, -9.4507179, -7.3986902, -1.6048541, 1.8102341

Time for backsubstitution: 5.63 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.37 seconds

### Candidate
type: B, layer: 3, pos: 402

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7962401, upper bound: 0.7896604
time: 3.99 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7962401, upper bound: 0.8030671
time: 4.03 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -7.3072877, -5.1018953, -7.3293614, -5.1036081, -1.8662500, 1.8948169
1: 1.8873105, 3.5444431, 1.9492719, 3.5604885, -1.3270068, 1.2509482
2: -4.9123201, -3.2756791, -4.9558296, -3.2797019, -1.2042913, 1.2516077
3: -11.0354338, -8.8966808, -11.0686750, -8.8831768, -1.6804252, 1.6941166
4: -5.5645566, -3.7567065, -5.5981708, -3.8420577, -1.5313654, 1.6636498
5: -9.0741472, -7.3128581, -9.0815630, -7.2752810, -1.7988663, 1.7687049
6: -6.5592098, -4.4138160, -6.5624762, -4.3072643, -1.6978927, 1.5963273
7: -8.7437325, -7.4088888, -8.8480625, -7.3834209, -1.1701930, 1.2625844
8: 1.0390034, 2.6184092, 1.0048251, 2.5446110, -1.1901929, 1.2923687
9: -9.3746729, -7.2596254, -9.4499531, -7.4096375, -1.5885170, 1.7979937

Time for backsubstitution: 5.68 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.37 seconds

### Candidate
type: B, layer: 3, pos: 423

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7743793, upper bound: 0.7934539
time: 4.20 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7759231, upper bound: 0.7938122
time: 4.01 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -7.3270020, -5.1075830, -7.3371229, -5.1035137, -1.8888640, 1.8851624
1: 1.8684199, 3.5445313, 1.9409628, 3.5610805, -1.3384845, 1.2697709
2: -4.9158492, -3.2790942, -4.9574251, -3.2790716, -1.2219234, 1.2465708
3: -11.0326090, -8.8868847, -11.0687017, -8.8781300, -1.6788135, 1.7014866
4: -5.5624924, -3.7522728, -5.5991955, -3.8405020, -1.5306935, 1.6755831
5: -9.0771637, -7.3020205, -9.0828342, -7.2715726, -1.8055911, 1.7808137
6: -6.5576620, -4.4125118, -6.5631948, -4.3060031, -1.6953897, 1.6004031
7: -8.7219315, -7.4087892, -8.8481417, -7.3796062, -1.1689470, 1.2656367
8: 1.0368590, 2.6243343, 1.0040684, 2.5472579, -1.1921606, 1.2988071
9: -9.3750725, -7.2159300, -9.4507504, -7.3966489, -1.6070471, 1.8107877

Time for backsubstitution: 5.71 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.38 seconds

### Candidate
type: B, layer: 3, pos: 402

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7969806, upper bound: 0.7896606
time: 4.29 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7969806, upper bound: 0.8030670
time: 4.20 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -7.3056631, -5.0999122, -7.3283825, -5.1073565, -1.8643379, 1.8971062
1: 1.9669955, 3.5556262, 1.8673604, 3.5444517, -1.2320862, 1.3432760
2: -4.9469681, -3.2823098, -4.9171495, -3.2772717, -1.2377484, 1.2095113
3: -11.0628061, -8.8965263, -11.0326824, -8.8825922, -1.7065630, 1.6616564
4: -5.5925274, -3.8488555, -5.5659370, -3.7514305, -1.6544275, 1.5226409
5: -9.0700178, -7.3113856, -9.0790691, -7.3025188, -1.7674990, 1.7676835
6: -6.5675230, -4.3262696, -6.5583239, -4.4106207, -1.6147547, 1.7012584
7: -8.8376179, -7.4093542, -8.7221336, -7.3972931, -1.2568574, 1.1297665
8: 1.0145617, 2.5453753, 1.0367970, 2.6261606, -1.2828605, 1.1845760
9: -9.4554577, -7.4482117, -9.3752069, -7.2202964, -1.8390369, 1.5530000

Time for backsubstitution: 5.65 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.37 seconds

### Candidate
type: B, layer: 3, pos: 423

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7790293, upper bound: 0.7864862
time: 4.06 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7803967, upper bound: 0.7869655
time: 4.14 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -7.3252287, -5.1051140, -7.3362327, -5.1072617, -1.8869739, 1.8875074
1: 1.9471767, 3.5582409, 1.8599374, 3.5449011, -1.2465241, 1.3641512
2: -4.9501371, -3.2865560, -4.9188271, -3.2767081, -1.2543740, 1.2047803
3: -11.0593834, -8.8867168, -11.0327110, -8.8775110, -1.7064252, 1.6699920
4: -5.5903339, -3.8453085, -5.5670042, -3.7496517, -1.6533689, 1.5328143
5: -9.0735254, -7.3005614, -9.0802517, -7.2988005, -1.7747250, 1.7796903
6: -6.5660152, -4.3244324, -6.5590248, -4.4094324, -1.6117105, 1.7068980
7: -8.8158188, -7.4089589, -8.7222042, -7.3935165, -1.2576165, 1.1296570
8: 1.0131907, 2.5515699, 1.0361109, 2.6288157, -1.2836266, 1.1883368
9: -9.4498243, -7.4126263, -9.3759127, -7.2048435, -1.8544145, 1.5596046

Time for backsubstitution: 5.52 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.38 seconds

### Candidate
type: B, layer: 3, pos: 402

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8030670, upper bound: 0.7817639
time: 4.04 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8030670, upper bound: 0.7962386
time: 4.14 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -7.3056631, -5.0999122, -7.3311596, -5.1060810, -1.8632855, 1.8978252
1: 1.9669955, 3.5556262, 1.8746238, 3.5421197, -1.2261701, 1.3440428
2: -4.9469681, -3.2823098, -4.9203653, -3.2725482, -1.2439244, 1.2110440
3: -11.0628061, -8.8965263, -11.0405083, -8.8833914, -1.7041926, 1.6694069
4: -5.5925274, -3.8488555, -5.5673885, -3.7500975, -1.6618881, 1.5358489
5: -9.0700178, -7.3113856, -9.0855989, -7.2767377, -1.7932801, 1.7742133
6: -6.5675230, -4.3262696, -6.5576258, -4.3945465, -1.5993829, 1.6872721
7: -8.8376179, -7.4093542, -8.7508106, -7.3836975, -1.2684443, 1.1572533
8: 1.0145617, 2.5453753, 1.0335650, 2.6132846, -1.2868783, 1.2021506
9: -9.4554577, -7.4482117, -9.3750954, -7.2181702, -1.8414974, 1.5538020

Time for backsubstitution: 5.52 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.38 seconds

### Candidate
type: B, layer: 3, pos: 423

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7801301, upper bound: 0.7864864
time: 4.42 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7817815, upper bound: 0.7869664
time: 4.18 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -7.3252287, -5.1051140, -7.3389502, -5.1059837, -1.8859229, 1.8881855
1: 1.9471767, 3.5582409, 1.8666568, 3.5426097, -1.2445469, 1.3639395
2: -4.9501371, -3.2865560, -4.9220276, -3.2719865, -1.2605047, 1.2059867
3: -11.0593834, -8.8867168, -11.0405397, -8.8783321, -1.7040043, 1.6778431
4: -5.5903339, -3.8453085, -5.5684485, -3.7484198, -1.6608286, 1.5470328
5: -9.0735254, -7.3005614, -9.0867767, -7.2730284, -1.8004971, 1.7862153
6: -6.5660152, -4.3244324, -6.5583224, -4.3932524, -1.5965624, 1.6904988
7: -8.8158188, -7.4089589, -8.7508879, -7.3800945, -1.2683539, 1.1585283
8: 1.0131907, 2.5515699, 1.0328445, 2.6159286, -1.2864757, 1.2103982
9: -9.4498243, -7.4126263, -9.3757935, -7.2026539, -1.8568637, 1.5597086

Time for backsubstitution: 5.49 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.38 seconds

### Candidate
type: B, layer: 3, pos: 402

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8017931, upper bound: 0.7817364
time: 4.28 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8017931, upper bound: 0.7962401
time: 4.40 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -7.3072877, -5.1018953, -7.3283825, -5.1073565, -1.8658576, 1.8956590
1: 1.8873105, 3.5444431, 1.8673604, 3.5444517, -1.2704647, 1.2865915
2: -4.9123201, -3.2756791, -4.9171495, -3.2772717, -1.2057474, 1.2162499
3: -11.0354338, -8.8966808, -11.0326824, -8.8825922, -1.6914215, 1.6700163
4: -5.5645566, -3.7567065, -5.5659370, -3.7514305, -1.5703354, 1.5667367
5: -9.0741472, -7.3128581, -9.0790691, -7.3025188, -1.7716284, 1.7662110
6: -6.5592098, -4.4138160, -6.5583239, -4.4106207, -1.6378837, 1.6315477
7: -8.7437325, -7.4088888, -8.7221336, -7.3972931, -1.1652803, 1.1439908
8: 1.0390034, 2.6184092, 1.0367970, 2.6261606, -1.2299554, 1.2219923
9: -9.3746729, -7.2596254, -9.3752069, -7.2202964, -1.6325710, 1.5943520

Time for backsubstitution: 5.51 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.38 seconds

### Candidate
type: B, layer: 3, pos: 423

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7697996, upper bound: 0.7894468
time: 4.01 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7711622, upper bound: 0.7896745
time: 4.34 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -7.3270020, -5.1075830, -7.3362327, -5.1072617, -1.8885450, 1.8860643
1: 1.8684199, 3.5445313, 1.8599374, 3.5449011, -1.2825856, 1.3073258
2: -4.9158492, -3.2790942, -4.9188271, -3.2767081, -1.2246637, 1.2121011
3: -11.0326090, -8.8868847, -11.0327110, -8.8775110, -1.6899366, 1.6782570
4: -5.5624924, -3.7522728, -5.5670042, -3.7496517, -1.5693965, 1.5768566
5: -9.0771637, -7.3020205, -9.0802517, -7.2988005, -1.7783632, 1.7782311
6: -6.5576620, -4.4125118, -6.5590248, -4.4094324, -1.6353502, 1.6379714
7: -8.7219315, -7.4087892, -8.7222042, -7.3935165, -1.1654620, 1.1433594
8: 1.0368590, 2.6243343, 1.0361109, 2.6288157, -1.2306261, 1.2281611
9: -9.3750725, -7.2159300, -9.3759127, -7.2048435, -1.6436851, 1.6052096

Time for backsubstitution: 5.51 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.38 seconds

### Candidate
type: B, layer: 3, pos: 402

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7958352, upper bound: 0.7836506
time: 4.22 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7958352, upper bound: 0.7989638
time: 4.48 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -7.3072877, -5.1018953, -7.3311596, -5.1060810, -1.8648057, 1.8963776
1: 1.8873105, 3.5444431, 1.8746238, 3.5421197, -1.2665277, 1.2875624
2: -4.9123201, -3.2756791, -4.9203653, -3.2725482, -1.2117641, 1.2186587
3: -11.0354338, -8.8966808, -11.0405083, -8.8833914, -1.6884851, 1.6777229
4: -5.5645566, -3.7567065, -5.5673885, -3.7500975, -1.5765963, 1.5793719
5: -9.0741472, -7.3128581, -9.0855989, -7.2767377, -1.7974095, 1.7727408
6: -6.5592098, -4.4138160, -6.5576258, -4.3945465, -1.6261427, 1.6169224
7: -8.7437325, -7.4088888, -8.7508106, -7.3836975, -1.1847277, 1.1737225
8: 1.0390034, 2.6184092, 1.0335650, 2.6132846, -1.2296777, 1.2393219
9: -9.3746729, -7.2596254, -9.3750954, -7.2181702, -1.6349468, 1.5955040

Time for backsubstitution: 5.56 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.38 seconds

### Candidate
type: B, layer: 3, pos: 423

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7729603, upper bound: 0.7894469
time: 3.94 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7745302, upper bound: 0.7896744
time: 3.89 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -7.3270020, -5.1075830, -7.3389502, -5.1059837, -1.8874931, 1.8867407
1: 1.8684199, 3.5445313, 1.8666568, 3.5426097, -1.2806449, 1.3070276
2: -4.9158492, -3.2790942, -4.9220276, -3.2719865, -1.2308972, 1.2136822
3: -11.0326090, -8.8868847, -11.0405397, -8.8783321, -1.6869497, 1.6861458
4: -5.5624924, -3.7522728, -5.5684485, -3.7484198, -1.5756564, 1.5901439
5: -9.0771637, -7.3020205, -9.0867767, -7.2730284, -1.8041353, 1.7847562
6: -6.5576620, -4.4125118, -6.5583224, -4.3932524, -1.6236122, 1.6215937
7: -8.7219315, -7.4087892, -8.7508879, -7.3800945, -1.1833029, 1.1764050
8: 1.0368590, 2.6243343, 1.0328445, 2.6159286, -1.2299995, 1.2469225
9: -9.3750725, -7.2159300, -9.3757935, -7.2026539, -1.6460516, 1.6056585

Time for backsubstitution: 5.60 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.38 seconds

### Candidate
type: B, layer: 3, pos: 402

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7962502, upper bound: 0.7836526
time: 4.34 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7962502, upper bound: 0.7989651
time: 4.37 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -7.3086243, -5.0988011, -7.3265853, -5.1048870, -1.8659749, 1.8937194
1: 1.9694457, 3.5586653, 1.9470694, 3.5580940, -1.2468944, 1.2563207
2: -4.9515362, -3.2774415, -4.9515848, -3.2844629, -1.1972811, 1.2117846
3: -11.0759735, -8.8965273, -11.0594692, -8.8824587, -1.7080073, 1.6775308
4: -5.5970941, -3.8469534, -5.5935507, -3.8439250, -1.5544443, 1.5442371
5: -9.0761414, -7.2854629, -9.0753517, -7.3010511, -1.7750902, 1.7898889
6: -6.5633383, -4.3104649, -6.5667105, -4.3228936, -1.6519608, 1.6488936
7: -8.8699198, -7.3949952, -8.8160324, -7.3976793, -1.2140284, 1.1813676
8: 1.0048432, 2.5368629, 1.0131469, 2.5531449, -1.2411311, 1.2142401
9: -9.4552622, -7.4461021, -9.4499121, -7.4116917, -1.6034956, 1.5664299

Time for backsubstitution: 5.64 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.38 seconds

### Candidate
type: B, layer: 3, pos: 423

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8038024, upper bound: 0.8167977
time: 4.64 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8052185, upper bound: 0.8167788
time: 4.48 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -7.3279648, -5.1038380, -7.3344097, -5.1047912, -1.8883219, 1.8840857
1: 1.9493752, 3.5606382, 1.9387870, 3.5586810, -1.2525816, 1.2728446
2: -4.9543581, -3.2817464, -4.9531937, -3.2838321, -1.2162855, 1.2059920
3: -11.0685892, -8.8873997, -11.0594950, -8.8773880, -1.7075081, 1.6845989
4: -5.5949545, -3.8434410, -5.5945511, -3.8423667, -1.5537057, 1.5551047
5: -9.0797663, -7.2746916, -9.0766268, -7.2973347, -1.7824316, 1.8019352
6: -6.5617738, -4.3088121, -6.5674324, -4.3216176, -1.6503382, 1.6545117
7: -8.8478403, -7.3947415, -8.8161068, -7.3938208, -1.2109711, 1.1813245
8: 1.0049062, 2.5425978, 1.0124044, 2.5558038, -1.2396791, 1.2183936
9: -9.4498444, -7.4104156, -9.4507179, -7.3986902, -1.6173186, 1.5787618

Time for backsubstitution: 5.58 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.38 seconds

### Candidate
type: B, layer: 3, pos: 402

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8264376, upper bound: 0.8179382
time: 5.01 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8264376, upper bound: 0.8262828
time: 4.42 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -7.3086243, -5.0988011, -7.3283825, -5.1073565, -1.8653126, 1.8960772
1: 1.9694457, 3.5586653, 1.8673604, 3.5444517, -1.2397890, 1.3444781
2: -4.9515362, -3.2774415, -4.9171495, -3.2772717, -1.2408917, 1.2158663
3: -11.0759735, -8.8965273, -11.0326824, -8.8825922, -1.7145295, 1.6592155
4: -5.5970941, -3.8469534, -5.5659370, -3.7514305, -1.6671820, 1.5289516
5: -9.0761414, -7.2854629, -9.0790691, -7.3025188, -1.7736225, 1.7936063
6: -6.5633383, -4.3104649, -6.5583239, -4.4106207, -1.6011562, 1.6913195
7: -8.8699198, -7.3949952, -8.7221336, -7.3972931, -1.2892404, 1.1506262
8: 1.0048432, 2.5368629, 1.0367970, 2.6261606, -1.3023820, 1.1853447
9: -9.4552622, -7.4461021, -9.3752069, -7.2202964, -1.8396482, 1.5551863

Time for backsubstitution: 5.58 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.39 seconds

### Candidate
type: B, layer: 3, pos: 423

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7790293, upper bound: 0.7870566
time: 4.11 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7803967, upper bound: 0.7877145
time: 4.20 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -7.3279648, -5.1038380, -7.3362327, -5.1072617, -1.8876600, 1.8864558
1: 1.9493752, 3.5606382, 1.8599374, 3.5449011, -1.2466090, 1.3602340
2: -4.9543581, -3.2817464, -4.9188271, -3.2767081, -1.2573557, 1.2107852
3: -11.0685892, -8.8873997, -11.0327110, -8.8775110, -1.7140312, 1.6673350
4: -5.5949545, -3.8434410, -5.5670042, -3.7496517, -1.6663375, 1.5390928
5: -9.0797663, -7.2746916, -9.0802517, -7.2988005, -1.7809658, 1.8055601
6: -6.5617738, -4.3088121, -6.5590248, -4.4094324, -1.5992932, 1.6968541
7: -8.8478403, -7.3947415, -8.7222042, -7.3935165, -1.2869878, 1.1504300
8: 1.0049062, 2.5425978, 1.0361109, 2.6288157, -1.3011353, 1.1878970
9: -9.4498444, -7.4104156, -9.3759127, -7.2048435, -1.8556447, 1.5616369

Time for backsubstitution: 5.59 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.38 seconds

### Candidate
type: B, layer: 3, pos: 402

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8030670, upper bound: 0.7852126
time: 4.06 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8030670, upper bound: 0.7969812
time: 4.10 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 14.14 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 14.14
Output dim: 1, lower bound: -0.8038024, upper bound: 0.8167789
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 14.14
Output dim: 1, lower bound: -0.8052185, upper bound: 0.8169532
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 14.14
Output dim: 1, lower bound: -0.8264376, upper bound: 0.8146742
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 14.14
Output dim: 1, lower bound: -0.8264376, upper bound: 0.8264376
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 14.14
Output dim: 1, lower bound: -0.8072753, upper bound: 0.8167777
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 14.14
Output dim: 1, lower bound: -0.8084418, upper bound: 0.8169509
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 14.14
Output dim: 1, lower bound: -0.8262810, upper bound: 0.8146718
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 14.14
Output dim: 1, lower bound: -0.8262810, upper bound: 0.8264351
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 14.14
Output dim: 1, lower bound: -0.7710668, upper bound: 0.7934539
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 14.14
Output dim: 1, lower bound: -0.7725162, upper bound: 0.7938135
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 14.14
Output dim: 1, lower bound: -0.7962401, upper bound: 0.7896604
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 14.14
Output dim: 1, lower bound: -0.7962401, upper bound: 0.8030671
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 14.14
Output dim: 1, lower bound: -0.7743793, upper bound: 0.7934539
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 14.14
Output dim: 1, lower bound: -0.7759231, upper bound: 0.7938122
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 14.14
Output dim: 1, lower bound: -0.7969806, upper bound: 0.7896606
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 14.14
Output dim: 1, lower bound: -0.7969806, upper bound: 0.8030670
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 14.14
Output dim: 1, lower bound: -0.7790293, upper bound: 0.7864862
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 14.14
Output dim: 1, lower bound: -0.7803967, upper bound: 0.7869655
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 14.14
Output dim: 1, lower bound: -0.8030670, upper bound: 0.7817639
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 14.14
Output dim: 1, lower bound: -0.8030670, upper bound: 0.7962386
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 14.14
Output dim: 1, lower bound: -0.7801301, upper bound: 0.7864864
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 14.14
Output dim: 1, lower bound: -0.7817815, upper bound: 0.7869664
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 14.14
Output dim: 1, lower bound: -0.8017931, upper bound: 0.7817364
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 14.14
Output dim: 1, lower bound: -0.8017931, upper bound: 0.7962401
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 14.14
Output dim: 1, lower bound: -0.7697996, upper bound: 0.7894468
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 14.14
Output dim: 1, lower bound: -0.7711622, upper bound: 0.7896745
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 14.14
Output dim: 1, lower bound: -0.7958352, upper bound: 0.7836506
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 14.14
Output dim: 1, lower bound: -0.7958352, upper bound: 0.7989638
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 14.14
Output dim: 1, lower bound: -0.7729603, upper bound: 0.7894469
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 14.14
Output dim: 1, lower bound: -0.7745302, upper bound: 0.7896744
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 14.14
Output dim: 1, lower bound: -0.7962502, upper bound: 0.7836526
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 14.14
Output dim: 1, lower bound: -0.7962502, upper bound: 0.7989651
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 14.14
Output dim: 1, lower bound: -0.8038024, upper bound: 0.8167977
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 14.14
Output dim: 1, lower bound: -0.8052185, upper bound: 0.8167788
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 14.14
Output dim: 1, lower bound: -0.8264376, upper bound: 0.8179382
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 14.14
Output dim: 1, lower bound: -0.8264376, upper bound: 0.8262828
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 14.14
Output dim: 1, lower bound: -0.7790293, upper bound: 0.7870566
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 14.14
Output dim: 1, lower bound: -0.7803967, upper bound: 0.7877145
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 14.14
Output dim: 1, lower bound: -0.8030670, upper bound: 0.7852126
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 14.14
Output dim: 1, lower bound: -0.8030670, upper bound: 0.7969812
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.14
Output dim: 1, lower bound: -0.7804329, upper bound: 0.7988239
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.14
Output dim: 1, lower bound: -0.7958351, upper bound: 0.7988237
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.14
Output dim: 1, lower bound: -0.7804329, upper bound: 0.7988234
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.14
Output dim: 1, lower bound: -0.7958351, upper bound: 0.7988228
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.14
Output dim: 1, lower bound: -0.7864687, upper bound: 0.7969815
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.14
Output dim: 1, lower bound: -0.7997889, upper bound: 0.7969810
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.14
Output dim: 1, lower bound: -0.7864687, upper bound: 0.7969811
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.14
Output dim: 1, lower bound: -0.7997889, upper bound: 0.7969812
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.14
Output dim: 1, lower bound: -0.7789260, upper bound: 0.7988241
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.14
Output dim: 1, lower bound: -0.7936035, upper bound: 0.7988238
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.14
Output dim: 1, lower bound: -0.7789239, upper bound: 0.7988235
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.14
Output dim: 1, lower bound: -0.7936031, upper bound: 0.7988238
Binary search (step 1): status=Status.UNKNOWN, k_low=4, k_high=7, k_mid=5, eps_mid=0.0195312, abs_max=1.3075802326202393
rel_dist={1: [-0.8569169395709344, 0.8569169395709273]}

## Binary search (step 2) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.38 seconds

### Candidate
type: A, layer: 3, pos: 655

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7516713, upper bound: 0.7530696
time: 4.18 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7516692, upper bound: 0.7516713
time: 4.57 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 9.15 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 9.15
Output dim: 1, lower bound: -0.7516713, upper bound: 0.7530696
IS_A2, status: Status.UNKNOWN, split count: 1, time: 9.15
Output dim: 1, lower bound: -0.7516692, upper bound: 0.7516713

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -7.3354440, -5.1043673, -7.3398447, -5.1030598, -1.7790189, 1.7866802
1: 1.9376824, 3.5761168, 1.9379427, 3.5846922, -1.2384791, 1.2342308
2: -4.9555454, -3.2822270, -4.9611592, -3.2767553, -1.1525884, 1.1492021
3: -11.0604258, -8.8754654, -11.0755272, -8.8744259, -1.6070309, 1.6225677
4: -5.6192431, -3.8418322, -5.6278539, -3.8396072, -1.5054340, 1.5123098
5: -9.0812569, -7.2970009, -9.0880070, -7.2680035, -1.7960639, 1.7878923
6: -6.5684814, -4.3194942, -6.5648212, -4.2935967, -1.5957508, 1.5698619
7: -8.8186340, -7.3928752, -8.8471136, -7.3775048, -1.1514869, 1.1583741
8: 0.9968667, 2.5568495, 0.9786448, 2.5484524, -1.1722355, 1.1927004
9: -9.4914379, -7.3977928, -9.4926643, -7.3950109, -1.5866423, 1.5882597

Time for backsubstitution: 5.66 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 1683
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.37 seconds

### Candidate
type: B, layer: 3, pos: 1683

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7162319, upper bound: 0.7232062
time: 5.33 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7212726, upper bound: 0.7227206
time: 3.83 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -7.3381610, -5.1030903, -7.3408861, -5.1029730, -1.8010921, 1.7862010
1: 1.9398844, 3.5777798, 1.9366424, 3.5877502, -1.2387543, 1.2369015
2: -4.9597774, -3.2774539, -4.9618831, -3.2757103, -1.1564965, 1.1521642
3: -11.0695000, -8.8761969, -11.0787811, -8.8738413, -1.6144414, 1.6231318
4: -5.6239738, -3.8399706, -5.6297193, -3.8395028, -1.5043502, 1.5225143
5: -9.0876074, -7.2712379, -9.0881596, -7.2604723, -1.8271351, 1.7769527
6: -6.5642338, -4.3038802, -6.5652165, -4.2872143, -1.5835872, 1.5986190
7: -8.8500109, -7.3786354, -8.8566360, -7.3772707, -1.1501198, 1.1862707
8: 0.9892220, 2.5483055, 0.9709892, 2.5485129, -1.2102799, 1.2024605
9: -9.4917717, -7.3957472, -9.4928179, -7.3944492, -1.5949562, 1.5904140

Time for backsubstitution: 5.69 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 1683
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.36 seconds

### Candidate
type: B, layer: 3, pos: 1683

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7162340, upper bound: 0.7212747
time: 4.16 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7212726, upper bound: 0.7212734
time: 3.80 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 14.02 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 14.02
Output dim: 1, lower bound: -0.7162319, upper bound: 0.7232062
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 14.02
Output dim: 1, lower bound: -0.7212726, upper bound: 0.7227206
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 14.02
Output dim: 1, lower bound: -0.7162340, upper bound: 0.7212747
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 14.02
Output dim: 1, lower bound: -0.7212726, upper bound: 0.7212734

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -7.3354440, -5.1043673, -7.3390756, -5.1034746, -1.7782793, 1.7860065
1: 1.9376824, 3.5761168, 1.9387722, 3.5672705, -1.2198100, 1.2334268
2: -4.9555454, -3.2822270, -4.9588947, -3.2782848, -1.1514800, 1.1401087
3: -11.0604258, -8.8754654, -11.0745945, -8.8760805, -1.6055894, 1.6204982
4: -5.6192431, -3.8418322, -5.6032825, -3.8400528, -1.5033245, 1.4845357
5: -9.0812569, -7.2970009, -9.0833273, -7.2682467, -1.7942486, 1.7827916
6: -6.5684814, -4.3194942, -6.5638256, -4.2956381, -1.5904531, 1.5689421
7: -8.8186340, -7.3928752, -8.8452539, -7.3780327, -1.1502938, 1.1482079
8: 0.9968667, 2.5568495, 0.9934611, 2.5475464, -1.1713564, 1.1830447
9: -9.4914379, -7.3977928, -9.4516478, -7.3953819, -1.5862696, 1.5386443

Time for backsubstitution: 5.54 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.36 seconds

### Candidate
type: A, layer: 3, pos: 1683

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7160944, upper bound: 0.7160286
time: 4.29 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7160944, upper bound: 0.7227222
time: 4.40 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -7.3346677, -5.1052227, -7.3409071, -5.1059456, -1.7771535, 1.7879686
1: 1.9386196, 3.5657091, 1.8644652, 3.5531845, -1.2201452, 1.3115532
2: -4.9431820, -3.2827320, -4.9234986, -3.2710557, -1.1932828, 1.1358895
3: -11.0520840, -8.8765192, -11.0472746, -8.8762217, -1.6060143, 1.6070938
4: -5.5998716, -3.8420587, -5.5727248, -3.7479982, -1.5974908, 1.5121999
5: -9.0784998, -7.2975502, -9.0872803, -7.2697024, -1.7898111, 1.7880769
6: -6.5672636, -4.3472228, -6.5589433, -4.3843341, -1.5563240, 1.5894418
7: -8.7902489, -7.3932190, -8.7474146, -7.3785648, -1.2029738, 1.1348166
8: 1.0094972, 2.5564594, 1.0218010, 2.6162119, -1.2234704, 1.1806314
9: -9.4541140, -7.3978863, -9.3767586, -7.2016878, -1.8037102, 1.5430648

Time for backsubstitution: 5.57 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.37 seconds

### Candidate
type: A, layer: 3, pos: 402

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7068942, upper bound: 0.7167796
time: 4.04 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7156217, upper bound: 0.7168357
time: 4.05 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -7.3381610, -5.1030903, -7.3401175, -5.1033878, -1.8003840, 1.7855289
1: 1.9398844, 3.5777798, 1.9374719, 3.5710576, -1.2200656, 1.2361064
2: -4.9597774, -3.2774539, -4.9596205, -3.2772422, -1.1553969, 1.1426980
3: -11.0695000, -8.8761969, -11.0779324, -8.8754902, -1.6130352, 1.6210642
4: -5.6239738, -3.8399706, -5.6051073, -3.8399491, -1.5022407, 1.4947054
5: -9.0876074, -7.2712379, -9.0834799, -7.2607141, -1.8268933, 1.7719021
6: -6.5642338, -4.3038802, -6.5642233, -4.2892561, -1.5782871, 1.5976923
7: -8.8500109, -7.3786354, -8.8547773, -7.3777995, -1.1490192, 1.1761043
8: 0.9892220, 2.5483055, 0.9858041, 2.5476089, -1.2093496, 1.1926746
9: -9.4917717, -7.3957472, -9.4517975, -7.3948212, -1.5945823, 1.5407577

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.36 seconds

### Candidate
type: A, layer: 3, pos: 1683

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7160944, upper bound: 0.7160943
time: 4.13 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7160944, upper bound: 0.7212748
time: 4.31 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -7.3373861, -5.1039453, -7.3419466, -5.1058588, -1.7992134, 1.7874913
1: 1.9407954, 3.5663147, 1.8631663, 3.5538683, -1.2209423, 1.3143947
2: -4.9470310, -3.2779560, -4.9242330, -3.2701712, -1.1966105, 1.1397810
3: -11.0609522, -8.8772745, -11.0505295, -8.8756151, -1.6131086, 1.6075883
4: -5.6045504, -3.8401933, -5.5745859, -3.7479036, -1.5960093, 1.5222423
5: -9.0849524, -7.2717834, -9.0874376, -7.2621732, -1.8227220, 1.7773933
6: -6.5630374, -4.3313761, -6.5593266, -4.3779087, -1.5440595, 1.6205573
7: -8.8206387, -7.3789892, -8.7568607, -7.3783426, -1.2088451, 1.1627586
8: 1.0028591, 2.5479207, 1.0145950, 2.6162744, -1.2568922, 1.1902568
9: -9.4542770, -7.3958406, -9.3769207, -7.2010975, -1.8120761, 1.5452144

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.36 seconds

### Candidate
type: A, layer: 3, pos: 402

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7068942, upper bound: 0.7156229
time: 3.83 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7156217, upper bound: 0.7156232
time: 3.92 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 13.59 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 13.59
Output dim: 1, lower bound: -0.7160944, upper bound: 0.7160286
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 13.59
Output dim: 1, lower bound: -0.7160944, upper bound: 0.7227222
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 13.59
Output dim: 1, lower bound: -0.7068942, upper bound: 0.7167796
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 13.59
Output dim: 1, lower bound: -0.7156217, upper bound: 0.7168357
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 13.59
Output dim: 1, lower bound: -0.7160944, upper bound: 0.7160943
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 13.59
Output dim: 1, lower bound: -0.7160944, upper bound: 0.7212748
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 13.59
Output dim: 1, lower bound: -0.7068942, upper bound: 0.7156229
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 13.59
Output dim: 1, lower bound: -0.7156217, upper bound: 0.7156232

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -7.3346786, -5.1047831, -7.3390756, -5.1034746, -1.7776070, 1.7852974
1: 1.9385400, 3.5586948, 1.9387722, 3.5672705, -1.2190228, 1.2147570
2: -4.9532833, -3.2837548, -4.9588947, -3.2782848, -1.1420100, 1.1390358
3: -11.0594978, -8.8771133, -11.0745945, -8.8760805, -1.6035209, 1.6190805
4: -5.5946770, -3.8422813, -5.6032825, -3.8400528, -1.4760089, 1.4824262
5: -9.0767145, -7.2972412, -9.0833273, -7.2682467, -1.7891827, 1.7809839
6: -6.5674729, -4.3215342, -6.5638256, -4.2956381, -1.5895324, 1.5636430
7: -8.8161144, -7.3933764, -8.8452539, -7.3780327, -1.1394053, 1.1471102
8: 1.0123816, 2.5559411, 0.9934611, 2.5475464, -1.1604211, 1.1821182
9: -9.4507437, -7.3981647, -9.4516478, -7.3953819, -1.5364168, 1.5382805

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.36 seconds

### Candidate
type: B, layer: 3, pos: 655

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7140281, upper bound: 0.7227130
time: 5.04 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7140281, upper bound: 0.7227151
time: 4.91 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -7.3365030, -5.1072540, -7.3390756, -5.1034746, -1.7799797, 1.7846365
1: 1.8596368, 3.5449114, 1.9387722, 3.5672705, -1.3064561, 1.2076545
2: -4.9189172, -3.2766373, -4.9588947, -3.2782848, -1.1467869, 1.1828008
3: -11.0327139, -8.8772326, -11.0745945, -8.8760805, -1.5852060, 1.6256042
4: -5.5671411, -3.7495725, -5.6032825, -3.8400528, -1.4607964, 1.5950656
5: -9.0803432, -7.2987070, -9.0833273, -7.2682467, -1.7932281, 1.7799354
6: -6.5590625, -4.4093394, -6.5638256, -4.2956381, -1.6320915, 1.5126090
7: -8.7222109, -7.3930678, -8.8452539, -7.3780327, -1.1086702, 1.2231209
8: 1.0360909, 2.6289511, 0.9934611, 2.5475464, -1.1313944, 1.2435579
9: -9.3759394, -7.2045193, -9.4516478, -7.3953819, -1.5251381, 1.7764759

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.36 seconds

### Candidate
type: B, layer: 3, pos: 655

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7140281, upper bound: 0.7232058
time: 5.85 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7140281, upper bound: 0.7232082
time: 4.06 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -7.3056536, -5.1001549, -7.3301568, -5.1060882, -1.7563243, 1.7951322
1: 1.9670792, 3.5632415, 1.8752229, 3.5525246, -1.1822367, 1.2885885
2: -4.9365478, -3.2812417, -4.9211578, -3.2718971, -1.1811984, 1.1360452
3: -11.0551205, -8.8959179, -11.0472298, -8.8833408, -1.6052065, 1.5868206
4: -5.5977564, -3.8486457, -5.5711594, -3.7503533, -1.5906801, 1.5012381
5: -9.0719280, -7.3116908, -9.0855942, -7.2747822, -1.7745209, 1.7711902
6: -6.5673203, -4.3520255, -6.5579557, -4.3860254, -1.5571961, 1.5844140
7: -8.8117666, -7.4092054, -8.7473001, -7.3837295, -1.1993005, 1.1120565
8: 1.0115943, 2.5458999, 1.0227900, 2.6125026, -1.2210991, 1.1706946
9: -9.4553738, -7.4479284, -9.3757849, -7.2226171, -1.7780726, 1.4897797

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.36 seconds

### Candidate
type: B, layer: 3, pos: 655

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7032155, upper bound: 0.7167797
time: 4.07 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7032155, upper bound: 0.7167787
time: 4.53 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -7.3252163, -5.1055536, -7.3393221, -5.1060009, -1.7766948, 1.7852225
1: 1.9472482, 3.5652895, 1.8659136, 3.5531120, -1.1962922, 1.3101518
2: -4.9405088, -3.2854629, -4.9229670, -3.2714725, -1.1974812, 1.1296685
3: -11.0519772, -8.8860931, -11.0472555, -8.8778477, -1.6042991, 1.5950766
4: -5.5955625, -3.8450794, -5.5719380, -3.7485023, -1.5897589, 1.5113525
5: -9.0753460, -7.3008614, -9.0867386, -7.2702527, -1.7817473, 1.7654614
6: -6.5658174, -4.3501306, -6.5586977, -4.3848643, -1.5544105, 1.5877848
7: -8.7899685, -7.4087868, -8.7473631, -7.3811941, -1.1996968, 1.1140201
8: 1.0103025, 2.5521297, 1.0219488, 2.6154079, -1.2214279, 1.1767831
9: -9.4531918, -7.4123387, -9.3766069, -7.2035947, -1.8009119, 1.4989624

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.38 seconds

### Candidate
type: B, layer: 3, pos: 655

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7139281, upper bound: 0.7168356
time: 4.04 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7139281, upper bound: 0.7168346
time: 4.59 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -7.3373909, -5.1035042, -7.3401175, -5.1033878, -1.7997117, 1.7848203
1: 1.9407153, 3.5610929, 1.9374719, 3.5710576, -1.2192607, 1.2174373
2: -4.9575157, -3.2789927, -4.9596205, -3.2772422, -1.1463032, 1.1415956
3: -11.0687046, -8.8778534, -11.0779324, -8.8754902, -1.6109672, 1.6196175
4: -5.5993242, -3.8404155, -5.6051073, -3.8399491, -1.4747658, 1.4925978
5: -9.0829248, -7.2714796, -9.0834799, -7.2607141, -1.8221140, 1.7700930
6: -6.5632372, -4.3059192, -6.5642233, -4.2892561, -1.5773668, 1.5922914
7: -8.8481512, -7.3791618, -8.8547773, -7.3777995, -1.1381326, 1.1749115
8: 1.0040431, 2.5473943, 0.9858041, 2.5476089, -1.1981692, 1.1917992
9: -9.4507761, -7.3961210, -9.4517975, -7.3948212, -1.5450666, 1.5403836

Time for backsubstitution: 5.44 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.36 seconds

### Candidate
type: B, layer: 3, pos: 655

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7140260, upper bound: 0.7206541
time: 5.11 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7140260, upper bound: 0.7206562
time: 6.17 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -7.3392220, -5.1059752, -7.3401175, -5.1033878, -1.8020825, 1.7841606
1: 1.8664083, 3.5426223, 1.9374719, 3.5710576, -1.3064036, 1.2103670
2: -4.9221191, -3.2719152, -4.9596205, -3.2772422, -1.1494358, 1.1852603
3: -11.0405436, -8.8780537, -11.0779324, -8.8754902, -1.5926204, 1.6264992
4: -5.5685873, -3.7483337, -5.6051073, -3.8399491, -1.4590788, 1.6064155
5: -9.0868692, -7.2729349, -9.0834799, -7.2607141, -1.8259726, 1.7690516
6: -6.5583653, -4.3931675, -6.5642233, -4.2892561, -1.6181374, 1.5421119
7: -8.7508974, -7.3796463, -8.8547773, -7.3777995, -1.1075139, 1.2435031
8: 1.0328178, 2.6160669, 0.9858041, 2.5476089, -1.1695910, 1.2572217
9: -9.3758202, -7.2023268, -9.4517975, -7.3948212, -1.5324900, 1.7788370

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.36 seconds

### Candidate
type: B, layer: 3, pos: 655

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7140260, upper bound: 0.7212732
time: 5.65 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7140260, upper bound: 0.7212748
time: 4.15 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -7.3086138, -5.0990419, -7.3312044, -5.1059999, -1.7786937, 1.7946744
1: 1.9695303, 3.5644088, 1.8739216, 3.5532093, -1.1905487, 1.2915895
2: -4.9413223, -3.2763634, -4.9218907, -3.2710140, -1.1851211, 1.1383473
3: -11.0685863, -8.8959265, -11.0504818, -8.8827362, -1.6122780, 1.5876513
4: -5.6023264, -3.8467455, -5.5730181, -3.7502606, -1.5891581, 1.5112643
5: -9.0780411, -7.2857652, -9.0857525, -7.2672520, -1.8070936, 1.7606502
6: -6.5631447, -4.3360262, -6.5583367, -4.3796144, -1.5436337, 1.6159165
7: -8.8439493, -7.3948402, -8.7567463, -7.3835154, -1.2046547, 1.1425962
8: 1.0036831, 2.5373960, 1.0156026, 2.6125622, -1.2525897, 1.1821761
9: -9.4547434, -7.4458151, -9.3759441, -7.2220478, -1.7847219, 1.4919174

Time for backsubstitution: 5.44 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.36 seconds

### Candidate
type: B, layer: 3, pos: 655

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7032155, upper bound: 0.7156228
time: 4.23 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7032155, upper bound: 0.7156225
time: 4.29 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -7.3279572, -5.1042809, -7.3403616, -5.1059155, -1.8003011, 1.7847424
1: 1.9494462, 3.5658901, 1.8646162, 3.5537970, -1.1972241, 1.3132977
2: -4.9438581, -3.2806408, -4.9236989, -3.2705886, -1.2019420, 1.1333299
3: -11.0608406, -8.8867950, -11.0505085, -8.8772449, -1.6113791, 1.5958004
4: -5.6002178, -3.8432128, -5.5737972, -3.7484069, -1.5883794, 1.5214276
5: -9.0818281, -7.2749863, -9.0868950, -7.2627254, -1.8147373, 1.7549286
6: -6.5615907, -4.3342056, -6.5590801, -4.3784261, -1.5420194, 1.6227777
7: -8.8203449, -7.3945603, -8.7568102, -7.3809714, -1.2058425, 1.1447487
8: 1.0037212, 2.5431428, 1.0147467, 2.6154695, -1.2551816, 1.1879914
9: -9.4533443, -7.4101243, -9.3767672, -7.2030029, -1.8092151, 1.5011606

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.36 seconds

### Candidate
type: B, layer: 3, pos: 655

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7139281, upper bound: 0.7156230
time: 4.20 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7139281, upper bound: 0.7156231
time: 4.04 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 14.08 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 14.08
Output dim: 1, lower bound: -0.7140281, upper bound: 0.7227130
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 14.08
Output dim: 1, lower bound: -0.7140281, upper bound: 0.7227151
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 14.08
Output dim: 1, lower bound: -0.7140281, upper bound: 0.7232058
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 14.08
Output dim: 1, lower bound: -0.7140281, upper bound: 0.7232082
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 14.08
Output dim: 1, lower bound: -0.7032155, upper bound: 0.7167797
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 14.08
Output dim: 1, lower bound: -0.7032155, upper bound: 0.7167787
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 14.08
Output dim: 1, lower bound: -0.7139281, upper bound: 0.7168356
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 14.08
Output dim: 1, lower bound: -0.7139281, upper bound: 0.7168346
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 14.08
Output dim: 1, lower bound: -0.7140260, upper bound: 0.7206541
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 14.08
Output dim: 1, lower bound: -0.7140260, upper bound: 0.7206562
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 14.08
Output dim: 1, lower bound: -0.7140260, upper bound: 0.7212732
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 14.08
Output dim: 1, lower bound: -0.7140260, upper bound: 0.7212748
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 14.08
Output dim: 1, lower bound: -0.7032155, upper bound: 0.7156228
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 14.08
Output dim: 1, lower bound: -0.7032155, upper bound: 0.7156225
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 14.08
Output dim: 1, lower bound: -0.7139281, upper bound: 0.7156230
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 14.08
Output dim: 1, lower bound: -0.7139281, upper bound: 0.7156231

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -7.3346786, -5.1047831, -7.3346786, -5.1047831, -1.7734179, 1.7734179
1: 1.9385400, 3.5586948, 1.9385400, 3.5586948, -1.2131128, 1.2131131
2: -4.9532833, -3.2837548, -4.9532833, -3.2837548, -1.1353083, 1.1353084
3: -11.0594978, -8.8771133, -11.0594978, -8.8771133, -1.6050115, 1.6050115
4: -5.5946770, -3.8422813, -5.5946770, -3.8422813, -1.4694271, 1.4694273
5: -9.0767145, -7.2972412, -9.0767145, -7.2972412, -1.7571054, 1.7571054
6: -6.5674729, -4.3215342, -6.5674729, -4.3215342, -1.5743961, 1.5743961
7: -8.8161144, -7.3933764, -8.8161144, -7.3933764, -1.1207275, 1.1207273
8: 1.0123816, 2.5559411, 1.0123816, 2.5559411, -1.1571009, 1.1571009
9: -9.4507437, -7.3981647, -9.4507437, -7.3981647, -1.5334423, 1.5334423

Time for backsubstitution: 5.43 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.37 seconds

### Candidate
type: A, layer: 3, pos: 402

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7253255, upper bound: 0.7371520
time: 4.48 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7347386, upper bound: 0.7371538
time: 4.75 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -7.3346786, -5.1047831, -7.3373909, -5.1035042, -1.7723665, 1.7741039
1: 1.9385400, 3.5586948, 1.9407153, 3.5610929, -1.2092624, 1.2132442
2: -4.9532833, -3.2837548, -4.9575157, -3.2789927, -1.1412942, 1.1381767
3: -11.0594978, -8.8771133, -11.0687046, -8.8778534, -1.6022310, 1.6126204
4: -5.5946770, -3.8422813, -5.5993242, -3.8404155, -1.4757094, 1.4824078
5: -9.0767145, -7.2972412, -9.0829248, -7.2714796, -1.7915335, 1.7802629
6: -6.5674729, -4.3215342, -6.5632372, -4.3059192, -1.5643613, 1.5621078
7: -8.8161144, -7.3933764, -8.8481512, -7.3791618, -1.1389291, 1.1500704
8: 1.0123816, 2.5559411, 1.0040431, 2.5473943, -1.1560295, 1.1746583
9: -9.4507437, -7.3981647, -9.4507761, -7.3961210, -1.5356357, 1.5346928

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.37 seconds

### Candidate
type: A, layer: 3, pos: 402

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7253255, upper bound: 0.7371538
time: 4.35 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7347386, upper bound: 0.7371538
time: 4.69 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -7.3365030, -5.1072540, -7.3346786, -5.1047831, -1.7757912, 1.7727571
1: 1.8596368, 3.5449114, 1.9385400, 3.5586948, -1.3005466, 1.2060106
2: -4.9189172, -3.2766373, -4.9532833, -3.2837548, -1.1400852, 1.1790733
3: -11.0327139, -8.8772326, -11.0594978, -8.8771133, -1.5866966, 1.6115351
4: -5.5671411, -3.7495725, -5.5946770, -3.8422813, -1.4542150, 1.5820673
5: -9.0803432, -7.2987070, -9.0767145, -7.2972412, -1.7611513, 1.7560568
6: -6.5590625, -4.4093394, -6.5674729, -4.3215342, -1.6169553, 1.5233622
7: -8.7222109, -7.3930678, -8.8161144, -7.3933764, -1.0899925, 1.1967382
8: 1.0360909, 2.6289511, 1.0123816, 2.5559411, -1.1280742, 1.2185409
9: -9.3759394, -7.2045193, -9.4507437, -7.3981647, -1.5221636, 1.7716379

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.38 seconds

### Candidate
type: A, layer: 3, pos: 402

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6970733, upper bound: 0.7172415
time: 3.99 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7087000, upper bound: 0.7172560
time: 4.04 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -7.3365030, -5.1072540, -7.3373909, -5.1035042, -1.7747397, 1.7734430
1: 1.8596368, 3.5449114, 1.9407153, 3.5610929, -1.2966957, 1.2061415
2: -4.9189172, -3.2766373, -4.9575157, -3.2789927, -1.1460710, 1.1819415
3: -11.0327139, -8.8772326, -11.0687046, -8.8778534, -1.5839162, 1.6191440
4: -5.5671411, -3.7495725, -5.5993242, -3.8404155, -1.4604974, 1.5950472
5: -9.0803432, -7.2987070, -9.0829248, -7.2714796, -1.7955790, 1.7792139
6: -6.5590625, -4.4093394, -6.5632372, -4.3059192, -1.6069205, 1.5110738
7: -8.7222109, -7.3930678, -8.8481512, -7.3791618, -1.1081936, 1.2260814
8: 1.0360909, 2.6289511, 1.0040431, 2.5473943, -1.1270027, 1.2360983
9: -9.3759394, -7.2045193, -9.4507761, -7.3961210, -1.5243571, 1.7728882

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.36 seconds

### Candidate
type: A, layer: 3, pos: 402

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6970733, upper bound: 0.7172439
time: 3.86 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7087000, upper bound: 0.7172576
time: 4.37 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -7.3056536, -5.1001549, -7.3256593, -5.1073923, -1.7520571, 1.7831931
1: 1.9670792, 3.5632415, 1.8699479, 3.5442948, -1.1760831, 1.2858014
2: -4.9365478, -3.2812417, -4.9165564, -3.2774780, -1.1743217, 1.1325994
3: -11.0551205, -8.8959179, -11.0326729, -8.8843803, -1.6062694, 1.5727472
4: -5.5977564, -3.8486457, -5.5655499, -3.7520239, -1.5828905, 1.4884491
5: -9.0719280, -7.3116908, -9.0786505, -7.3037996, -1.7424374, 1.7474871
6: -6.5673203, -4.3520255, -6.5580783, -4.4110494, -1.5421810, 1.5963914
7: -8.8117666, -7.4092054, -8.7221069, -7.3987131, -1.1870055, 1.0852838
8: 1.0115943, 2.5458999, 1.0370350, 2.6252298, -1.2115145, 1.1465297
9: -9.4553738, -7.4479284, -9.3749599, -7.2254286, -1.7747793, 1.4851415

Time for backsubstitution: 5.52 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.36 seconds

### Candidate
type: A, layer: 3, pos: 1683

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6968724, upper bound: 0.7105352
time: 4.34 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6968724, upper bound: 0.7167789
time: 3.94 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -7.3056536, -5.1001549, -7.3284597, -5.1061158, -1.7510052, 1.7839274
1: 1.9670792, 3.5632415, 1.8771482, 3.5419497, -1.1721129, 1.2870991
2: -4.9365478, -3.2812417, -4.9197788, -3.2727540, -1.1805031, 1.1351905
3: -11.0551205, -8.8959179, -11.0404987, -8.8851728, -1.6039147, 1.5804543
4: -5.5977564, -3.8486457, -5.5670023, -3.7506886, -1.5903511, 1.5010724
5: -9.0719280, -7.3116908, -9.0851812, -7.2780170, -1.7768707, 1.7704797
6: -6.5673203, -4.3520255, -6.5573807, -4.3950086, -1.5304396, 1.5824337
7: -8.8117666, -7.4092054, -8.7507820, -7.3847733, -1.1988642, 1.1150167
8: 1.0115943, 2.5458999, 1.0338163, 2.6123576, -1.2159033, 1.1638510
9: -9.4553738, -7.4479284, -9.3748531, -7.2233343, -1.7772441, 1.4862971

Time for backsubstitution: 5.53 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.37 seconds

### Candidate
type: A, layer: 3, pos: 1683

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6968724, upper bound: 0.7105352
time: 4.13 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6968724, upper bound: 0.7167789
time: 4.33 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -7.3252163, -5.1055536, -7.3349166, -5.1073093, -1.7725773, 1.7733436
1: 1.9472482, 3.5652895, 1.8612332, 3.5448451, -1.1885004, 1.3091278
2: -4.9405088, -3.2854629, -4.9183879, -3.2770500, -1.1906898, 1.1275170
3: -11.0519772, -8.8860931, -11.0326977, -8.8788509, -1.6054201, 1.5809350
4: -5.5955625, -3.8450794, -5.5663462, -3.7500286, -1.5819702, 1.4985547
5: -9.0753460, -7.3008614, -9.0798054, -7.2992620, -1.7496624, 1.7419300
6: -6.5658174, -4.3501306, -6.5588293, -4.4098725, -1.5394106, 1.6022639
7: -8.7899685, -7.4087868, -8.7221632, -7.3956971, -1.1887341, 1.0863121
8: 1.0103025, 2.5521297, 1.0362210, 2.6281543, -1.2136853, 1.1523666
9: -9.4531918, -7.4123387, -9.3757906, -7.2064238, -1.7976341, 1.4956968

Time for backsubstitution: 5.51 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.38 seconds

### Candidate
type: A, layer: 3, pos: 1683

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7085496, upper bound: 0.7105339
time: 4.34 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7085496, upper bound: 0.7168367
time: 4.37 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -7.3252163, -5.1055536, -7.3376341, -5.1060295, -1.7715268, 1.7740195
1: 1.9472482, 3.5652895, 1.8678567, 3.5425491, -1.1865528, 1.3086622
2: -4.9405088, -3.2854629, -4.9215856, -3.2723317, -1.1968997, 1.1288526
3: -11.0519772, -8.8860931, -11.0405245, -8.8796797, -1.6030059, 1.5888228
4: -5.5955625, -3.8450794, -5.5677876, -3.7488368, -1.5894294, 1.5118594
5: -9.0753460, -7.3008614, -9.0863237, -7.2734866, -1.7840967, 1.7647481
6: -6.5658174, -4.3501306, -6.5581217, -4.3936558, -1.5276694, 1.5858111
7: -8.7899685, -7.4087868, -8.7508459, -7.3822765, -1.1992545, 1.1193745
8: 1.0103025, 2.5521297, 1.0329657, 2.6152620, -1.2162297, 1.1711190
9: -9.4531918, -7.4123387, -9.3756704, -7.2042341, -1.8000822, 1.4961436

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.37 seconds

### Candidate
type: A, layer: 3, pos: 1683

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7085496, upper bound: 0.7105352
time: 4.49 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7085496, upper bound: 0.7168367
time: 4.55 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -7.3373909, -5.1035042, -7.3346786, -5.1047831, -1.7741041, 1.7723665
1: 1.9407153, 3.5610929, 1.9385400, 3.5586948, -1.2132444, 1.2092624
2: -4.9575157, -3.2789927, -4.9532833, -3.2837548, -1.1381767, 1.1412942
3: -11.0687046, -8.8778534, -11.0594978, -8.8771133, -1.6126204, 1.6022310
4: -5.5993242, -3.8404155, -5.5946770, -3.8422813, -1.4824076, 1.4757097
5: -9.0829248, -7.2714796, -9.0767145, -7.2972412, -1.7802629, 1.7915330
6: -6.5632372, -4.3059192, -6.5674729, -4.3215342, -1.5621076, 1.5643616
7: -8.8481512, -7.3791618, -8.8161144, -7.3933764, -1.1500707, 1.1389291
8: 1.0040431, 2.5473943, 1.0123816, 2.5559411, -1.1746585, 1.1560295
9: -9.4507761, -7.3961210, -9.4507437, -7.3981647, -1.5346925, 1.5356357

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.37 seconds

### Candidate
type: A, layer: 3, pos: 402

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7253276, upper bound: 0.7367540
time: 4.27 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7347386, upper bound: 0.7367563
time: 4.71 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -7.3373909, -5.1035042, -7.3373909, -5.1035042, -1.7947726, 1.7947726
1: 1.9407153, 3.5610929, 1.9407153, 3.5610929, -1.2146122, 1.2146120
2: -4.9575157, -3.2789927, -4.9575157, -3.2789927, -1.1403286, 1.1403284
3: -11.0687046, -8.8778534, -11.0687046, -8.8778534, -1.6091132, 1.6091132
4: -5.5993242, -3.8404155, -5.5993242, -3.8404155, -1.4743795, 1.4743795
5: -9.0829248, -7.2714796, -9.0829248, -7.2714796, -1.7692089, 1.7692089
6: -6.5632372, -4.3059192, -6.5632372, -4.3059192, -1.5910711, 1.5910714
7: -8.8481512, -7.3791618, -8.8481512, -7.3791618, -1.1375220, 1.1375220
8: 1.0040431, 2.5473943, 1.0040431, 2.5473943, -1.1970739, 1.1970737
9: -9.4507761, -7.3961210, -9.4507761, -7.3961210, -1.5429931, 1.5429931

Time for backsubstitution: 6.36 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.38 seconds

### Candidate
type: A, layer: 3, pos: 402

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7253276, upper bound: 0.7367564
time: 4.39 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7347386, upper bound: 0.7367549
time: 4.58 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -7.3392220, -5.1059752, -7.3346786, -5.1047831, -1.7764678, 1.7717068
1: 1.8664083, 3.5426223, 1.9385400, 3.5586948, -1.3003874, 1.2002306
2: -4.9221191, -3.2719152, -4.9532833, -3.2837548, -1.1413093, 1.1852360
3: -11.0405436, -8.8780537, -11.0594978, -8.8771133, -1.5944495, 1.6091127
4: -5.5685873, -3.7483337, -5.5946770, -3.8422813, -1.4673972, 1.5895274
5: -9.0868692, -7.2729349, -9.0767145, -7.2972412, -1.7841215, 1.7904868
6: -6.5583653, -4.3931675, -6.5674729, -4.3215342, -1.6028781, 1.5082030
7: -8.7508974, -7.3796463, -8.8161144, -7.3933764, -1.1174769, 1.2075207
8: 1.0328178, 2.6160669, 1.0123816, 2.5559411, -1.1456981, 1.2214520
9: -9.3758202, -7.2023268, -9.4507437, -7.3981647, -1.5229595, 1.7740893

Time for backsubstitution: 5.56 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.38 seconds

### Candidate
type: A, layer: 3, pos: 402

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6970733, upper bound: 0.7156211
time: 4.11 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7087000, upper bound: 0.7156218
time: 5.93 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -7.3392220, -5.1059752, -7.3373909, -5.1035042, -1.7971439, 1.7941122
1: 1.8664083, 3.5426223, 1.9407153, 3.5610929, -1.3018649, 1.2075417
2: -4.9221191, -3.2719152, -4.9575157, -3.2789927, -1.1450787, 1.1839931
3: -11.0405436, -8.8780537, -11.0687046, -8.8778534, -1.5907669, 1.6156383
4: -5.5685873, -3.7483337, -5.5993242, -3.8404155, -1.4586926, 1.5874758
5: -9.0868692, -7.2729349, -9.0829248, -7.2714796, -1.7732577, 1.7681675
6: -6.5583653, -4.3931675, -6.5632372, -4.3059192, -1.6339464, 1.5408921
7: -8.7508974, -7.3796463, -8.8481512, -7.3791618, -1.1069038, 1.2134695
8: 1.0328178, 2.6160669, 1.0040431, 2.5473943, -1.1684961, 1.2587919
9: -9.3758202, -7.2023268, -9.4507761, -7.3961210, -1.5304165, 1.7814710

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.37 seconds

### Candidate
type: A, layer: 3, pos: 402

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6970733, upper bound: 0.7156239
time: 4.07 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7087000, upper bound: 0.7156238
time: 4.34 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -7.3086138, -5.0990419, -7.3256593, -5.1073923, -1.7530313, 1.7821617
1: 1.9695303, 3.5644088, 1.8699479, 3.5442948, -1.1837845, 1.2869682
2: -4.9413223, -3.2763634, -4.9165564, -3.2774780, -1.1767664, 1.1389726
3: -11.0685863, -8.8959265, -11.0326729, -8.8843803, -1.6142364, 1.5702324
4: -5.6023264, -3.8467455, -5.5655499, -3.7520239, -1.5967789, 1.4947529
5: -9.0780411, -7.2857652, -9.0786505, -7.3037996, -1.7652373, 1.7820950
6: -6.5631447, -4.3360262, -6.5580783, -4.4110494, -1.5285897, 1.5837066
7: -8.8439493, -7.3948402, -8.7221069, -7.3987131, -1.2166610, 1.1062231
8: 1.0036831, 2.5373960, 1.0370350, 2.6252298, -1.2332206, 1.1472733
9: -9.4547434, -7.4458151, -9.3749599, -7.2254286, -1.7751846, 1.4873347

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.37 seconds

### Candidate
type: A, layer: 3, pos: 1683

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6968702, upper bound: 0.7107676
time: 4.76 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6968702, upper bound: 0.7156219
time: 4.15 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -7.3086138, -5.0990419, -7.3284597, -5.1061158, -1.7737532, 1.8046246
1: 1.9695303, 3.5644088, 1.8771482, 3.5419497, -1.1786919, 1.2887752
2: -4.9413223, -3.2763634, -4.9197788, -3.2727540, -1.1794310, 1.1370454
3: -11.0685863, -8.8959265, -11.0404987, -8.8851728, -1.6104202, 1.5770378
4: -5.6023264, -3.8467455, -5.5670023, -3.7506886, -1.5887394, 1.4909935
5: -9.0780411, -7.2857652, -9.0851812, -7.2780170, -1.7545953, 1.7597728
6: -6.5631447, -4.3360262, -6.5573807, -4.3950086, -1.5608320, 1.6146669
7: -8.8439493, -7.3948402, -8.7507820, -7.3847733, -1.2041054, 1.1038096
8: 1.0036831, 2.5373960, 1.0338163, 2.6123576, -1.2516260, 1.1867454
9: -9.4547434, -7.4458151, -9.3748531, -7.2233343, -1.7825723, 1.4920828

Time for backsubstitution: 5.60 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.37 seconds

### Candidate
type: A, layer: 3, pos: 1683

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6968724, upper bound: 0.7107694
time: 4.43 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6968724, upper bound: 0.7156220
time: 4.28 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -7.3279572, -5.1042809, -7.3349166, -5.1073093, -1.7732649, 1.7722902
1: 1.9494462, 3.5658901, 1.8612332, 3.5448451, -1.1885886, 1.3042433
2: -4.9438581, -3.2806408, -4.9183879, -3.2770500, -1.1936722, 1.1335406
3: -11.0608406, -8.8867950, -11.0326977, -8.8788509, -1.6130285, 1.5782003
4: -5.6002178, -3.8432128, -5.5663462, -3.7500286, -1.5960617, 1.5048342
5: -9.0818281, -7.2749863, -9.0798054, -7.2992620, -1.7728796, 1.7763643
6: -6.5615907, -4.3342056, -6.5588293, -4.4098725, -1.5269899, 1.5895648
7: -8.8203449, -7.3945603, -8.7221632, -7.3956971, -1.2166529, 1.1071153
8: 1.0037212, 2.5431428, 1.0362210, 2.6281543, -1.2298832, 1.1518810
9: -9.4533443, -7.4101243, -9.3757906, -7.2064238, -1.7987008, 1.4977384

Time for backsubstitution: 5.55 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.37 seconds

### Candidate
type: A, layer: 3, pos: 1683

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7085496, upper bound: 0.7107841
time: 4.25 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7085496, upper bound: 0.7156221
time: 4.71 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -7.3279572, -5.1042809, -7.3376341, -5.1060295, -1.7953615, 1.7946925
1: 1.9494462, 3.5658901, 1.8678567, 3.5425491, -1.1876340, 1.3104429
2: -4.9438581, -3.2806408, -4.9215856, -3.2723317, -1.1950977, 1.1320608
3: -11.0608406, -8.8867950, -11.0405245, -8.8796797, -1.6095200, 1.5847549
4: -5.6002178, -3.8432128, -5.5677876, -3.7488368, -1.5879607, 1.5010793
5: -9.0818281, -7.2749863, -9.0863237, -7.2734866, -1.7618594, 1.7540493
6: -6.5615907, -4.3342056, -6.5581217, -4.3936558, -1.5576344, 1.6215286
7: -8.8203449, -7.3945603, -8.7508459, -7.3822765, -1.2052763, 1.1051860
8: 1.0037212, 2.5431428, 1.0329657, 2.6152620, -1.2542150, 1.1884255
9: -9.4533443, -7.4101243, -9.3756704, -7.2042341, -1.8070664, 1.5010231

Time for backsubstitution: 5.50 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1269
type: A, layer: 3, pos: 2832

Time for candidate selection: 0.37 seconds

### Candidate
type: A, layer: 3, pos: 1683

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7085496, upper bound: 0.7107843
time: 4.33 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7085496, upper bound: 0.7156223
time: 4.25 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 14.47 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.47
Output dim: 1, lower bound: -0.7253255, upper bound: 0.7371520
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.47
Output dim: 1, lower bound: -0.7347386, upper bound: 0.7371538
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.47
Output dim: 1, lower bound: -0.7253255, upper bound: 0.7371538
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.47
Output dim: 1, lower bound: -0.7347386, upper bound: 0.7371538
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.47
Output dim: 1, lower bound: -0.6970733, upper bound: 0.7172415
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.47
Output dim: 1, lower bound: -0.7087000, upper bound: 0.7172560
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.47
Output dim: 1, lower bound: -0.6970733, upper bound: 0.7172439
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.47
Output dim: 1, lower bound: -0.7087000, upper bound: 0.7172576
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.47
Output dim: 1, lower bound: -0.6968724, upper bound: 0.7105352
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.47
Output dim: 1, lower bound: -0.6968724, upper bound: 0.7167789
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.47
Output dim: 1, lower bound: -0.6968724, upper bound: 0.7105352
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.47
Output dim: 1, lower bound: -0.6968724, upper bound: 0.7167789
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.47
Output dim: 1, lower bound: -0.7085496, upper bound: 0.7105339
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.47
Output dim: 1, lower bound: -0.7085496, upper bound: 0.7168367
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.47
Output dim: 1, lower bound: -0.7085496, upper bound: 0.7105352
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.47
Output dim: 1, lower bound: -0.7085496, upper bound: 0.7168367
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.47
Output dim: 1, lower bound: -0.7253276, upper bound: 0.7367540
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.47
Output dim: 1, lower bound: -0.7347386, upper bound: 0.7367563
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.47
Output dim: 1, lower bound: -0.7253276, upper bound: 0.7367564
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.47
Output dim: 1, lower bound: -0.7347386, upper bound: 0.7367549
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.47
Output dim: 1, lower bound: -0.6970733, upper bound: 0.7156211
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.47
Output dim: 1, lower bound: -0.7087000, upper bound: 0.7156218
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.47
Output dim: 1, lower bound: -0.6970733, upper bound: 0.7156239
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.47
Output dim: 1, lower bound: -0.7087000, upper bound: 0.7156238
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.47
Output dim: 1, lower bound: -0.6968702, upper bound: 0.7107676
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.47
Output dim: 1, lower bound: -0.6968702, upper bound: 0.7156219
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.47
Output dim: 1, lower bound: -0.6968724, upper bound: 0.7107694
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.47
Output dim: 1, lower bound: -0.6968724, upper bound: 0.7156220
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.47
Output dim: 1, lower bound: -0.7085496, upper bound: 0.7107841
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.47
Output dim: 1, lower bound: -0.7085496, upper bound: 0.7156221
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.47
Output dim: 1, lower bound: -0.7085496, upper bound: 0.7107843
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.47
Output dim: 1, lower bound: -0.7085496, upper bound: 0.7156223

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -7.3056631, -5.0999122, -7.3238716, -5.1049204, -1.7526107, 1.7805424
1: 1.9669955, 3.5556262, 1.9496531, 3.5578897, -1.1752498, 1.1876521
2: -4.9469681, -3.2823098, -4.9510164, -3.2846971, -1.1237652, 1.1353431
3: -11.0628061, -8.8965263, -11.0594578, -8.8842440, -1.6029744, 1.5847287
4: -5.5925274, -3.8488555, -5.5931864, -3.8444762, -1.4626284, 1.4585278
5: -9.0700178, -7.3113856, -9.0748949, -7.3023300, -1.7409015, 1.7401681
6: -6.5675230, -4.3262696, -6.5664520, -4.3233490, -1.5752668, 1.5686479
7: -8.8376179, -7.4093542, -8.8160057, -7.3991213, -1.1173382, 1.0980163
8: 1.0145617, 2.5453753, 1.0134072, 2.5522137, -1.1546841, 1.1470647
9: -9.4554577, -7.4482117, -9.4496288, -7.4162288, -1.5142808, 1.4802353

Time for backsubstitution: 5.53 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.37 seconds

### Candidate
type: B, layer: 3, pos: 423

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7182810, upper bound: 0.7286229
time: 4.41 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7195781, upper bound: 0.7295276
time: 4.64 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -7.3252287, -5.1051140, -7.3331013, -5.1048384, -1.7730608, 1.7706890
1: 1.9471767, 3.5582409, 1.9399829, 3.5586171, -1.1882555, 1.2122784
2: -4.9501371, -3.2865560, -4.9527569, -3.2842202, -1.1433902, 1.1293317
3: -11.0593834, -8.8867168, -11.0594778, -8.8787193, -1.6032906, 1.5919261
4: -5.5903339, -3.8453085, -5.5939417, -3.8427868, -1.4617481, 1.4690113
5: -9.0735254, -7.3005614, -9.0761862, -7.2977958, -1.7490191, 1.7352400
6: -6.5660152, -4.3244324, -6.5672331, -4.3220191, -1.5724912, 1.5747299
7: -8.8158188, -7.4089589, -8.8160648, -7.3959808, -1.1176949, 1.0994685
8: 1.0131907, 2.5515699, 1.0125160, 2.5551615, -1.1552558, 1.1523490
9: -9.4498243, -7.4126263, -9.4505930, -7.4009986, -1.5298867, 1.4915187

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.37 seconds

### Candidate
type: B, layer: 3, pos: 423

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7280634, upper bound: 0.7286232
time: 4.98 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7295255, upper bound: 0.7295276
time: 4.55 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -7.3056631, -5.0999122, -7.3266726, -5.1036439, -1.7515583, 1.7812877
1: 1.9669955, 3.5556262, 1.9521410, 3.5602832, -1.1712828, 1.1887300
2: -4.9469681, -3.2823098, -4.9552641, -3.2799335, -1.1296370, 1.1383457
3: -11.0628061, -8.8965263, -11.0686626, -8.8849545, -1.6002665, 1.5923347
4: -5.5925274, -3.8488555, -5.5977998, -3.8426080, -1.4689083, 1.4714489
5: -9.0700178, -7.3113856, -9.0811110, -7.2765565, -1.7753305, 1.7633491
6: -6.5675230, -4.3262696, -6.5622206, -4.3077168, -1.5652323, 1.5563269
7: -8.8376179, -7.4093542, -8.8480320, -7.3848505, -1.1369932, 1.1273646
8: 1.0145617, 2.5453753, 1.0050888, 2.5436807, -1.1540577, 1.1646240
9: -9.4554577, -7.4482117, -9.4496727, -7.4141750, -1.5164878, 1.4814963

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.37 seconds

### Candidate
type: B, layer: 3, pos: 423

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7213599, upper bound: 0.7286247
time: 4.67 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7225311, upper bound: 0.7295276
time: 5.15 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -7.3252287, -5.1051140, -7.3358173, -5.1035595, -1.7720094, 1.7713733
1: 1.9471767, 3.5582409, 1.9421654, 3.5610182, -1.1863446, 1.2123785
2: -4.9501371, -3.2865560, -4.9569855, -3.2794609, -1.1495817, 1.1319450
3: -11.0593834, -8.8867168, -11.0686855, -8.8794651, -1.6005177, 1.5996332
4: -5.5903339, -3.8453085, -5.5985651, -3.8409219, -1.4680290, 1.4827671
5: -9.0735254, -7.3005614, -9.0823879, -7.2720332, -1.7834473, 1.7579584
6: -6.5660152, -4.3244324, -6.5629883, -4.3064032, -1.5624549, 1.5609736
7: -8.8158188, -7.4089589, -8.8480978, -7.3817639, -1.1353116, 1.1318824
8: 1.0131907, 2.5515699, 1.0041881, 2.5465856, -1.1540849, 1.1710873
9: -9.4498243, -7.4126263, -9.4506207, -7.3989234, -1.5320771, 1.4920704

Time for backsubstitution: 5.49 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.39 seconds

### Candidate
type: B, layer: 3, pos: 423

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7279946, upper bound: 0.7286237
time: 4.64 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7290980, upper bound: 0.7295265
time: 4.58 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -7.3072877, -5.1018953, -7.3238716, -5.1049204, -1.7549124, 1.7798791
1: 1.8873105, 3.5444431, 1.9496531, 3.5578897, -1.2670033, 1.1830335
2: -4.9123201, -3.2756791, -4.9510164, -3.2846971, -1.1280520, 1.1785173
3: -11.0354338, -8.8966808, -11.0594578, -8.8842440, -1.5860844, 1.5912671
4: -5.5645566, -3.7567065, -5.5931864, -3.8444762, -1.4460244, 1.5713151
5: -9.0741472, -7.3128581, -9.0748949, -7.3023300, -1.7458763, 1.7391181
6: -6.5592098, -4.4138160, -6.5664520, -4.3233490, -1.6176348, 1.5184550
7: -8.7437325, -7.4088888, -8.8160057, -7.3991213, -1.0864625, 1.1707451
8: 1.0390034, 2.6184092, 1.0134072, 2.5522137, -1.1240053, 1.2084026
9: -9.3746729, -7.2596254, -9.4496288, -7.4162288, -1.4977098, 1.7127271

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.39 seconds

### Candidate
type: B, layer: 3, pos: 423

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6898078, upper bound: 0.7086875
time: 4.38 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6918135, upper bound: 0.7097632
time: 4.55 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -7.3270020, -5.1075830, -7.3331013, -5.1048384, -1.7753410, 1.7700286
1: 1.8684199, 3.5445313, 1.9399829, 3.5586171, -1.2761433, 1.2051618
2: -4.9158492, -3.2790942, -4.9527569, -3.2842202, -1.1458151, 1.1730783
3: -11.0326090, -8.8868847, -11.0594778, -8.8787193, -1.5849800, 1.5984492
4: -5.5624924, -3.7522728, -5.5939417, -3.8427868, -1.4454226, 1.5820246
5: -9.0771637, -7.3020205, -9.0761862, -7.2977958, -1.7529430, 1.7341924
6: -6.5576620, -4.4125118, -6.5672331, -4.3220191, -1.6151605, 1.5242949
7: -8.7219315, -7.4087892, -8.8160648, -7.3959808, -1.0869403, 1.1721559
8: 1.0368590, 2.6243343, 1.0125160, 2.5551615, -1.1263340, 1.2135749
9: -9.3750725, -7.2159300, -9.4505930, -7.4009986, -1.5186524, 1.7250235

Time for backsubstitution: 5.58 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.37 seconds

### Candidate
type: B, layer: 3, pos: 423

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7011202, upper bound: 0.7087028
time: 4.85 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7030954, upper bound: 0.7097786
time: 5.08 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -7.3072877, -5.1018953, -7.3266726, -5.1036439, -1.7538600, 1.7806244
1: 1.8873105, 3.5444431, 1.9521410, 3.5602832, -1.2630358, 1.1841111
2: -4.9123201, -3.2756791, -4.9552641, -3.2799335, -1.1339235, 1.1815197
3: -11.0354338, -8.8966808, -11.0686626, -8.8849545, -1.5833769, 1.5988736
4: -5.5645566, -3.7567065, -5.5977998, -3.8426080, -1.4523044, 1.5842357
5: -9.0741472, -7.3128581, -9.0811110, -7.2765565, -1.7803049, 1.7622991
6: -6.5592098, -4.4138160, -6.5622206, -4.3077168, -1.6076002, 1.5061340
7: -8.7437325, -7.4088888, -8.8480320, -7.3848505, -1.1061170, 1.2000935
8: 1.0390034, 2.6184092, 1.0050888, 2.5436807, -1.1233790, 1.2259622
9: -9.3746729, -7.2596254, -9.4496727, -7.4141750, -1.4999168, 1.7139883

Time for backsubstitution: 5.50 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.37 seconds

### Candidate
type: B, layer: 3, pos: 423

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6919354, upper bound: 0.7086875
time: 4.36 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6939092, upper bound: 0.7097614
time: 4.10 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -7.3270020, -5.1075830, -7.3358173, -5.1035595, -1.7742896, 1.7707129
1: 1.8684199, 3.5445313, 1.9421654, 3.5610182, -1.2742324, 1.2052619
2: -4.9158492, -3.2790942, -4.9569855, -3.2794609, -1.1520066, 1.1756916
3: -11.0326090, -8.8868847, -11.0686855, -8.8794651, -1.5822072, 1.6061563
4: -5.5624924, -3.7522728, -5.5985651, -3.8409219, -1.4517031, 1.5957804
5: -9.0771637, -7.3020205, -9.0823879, -7.2720332, -1.7873707, 1.7569113
6: -6.5576620, -4.4125118, -6.5629883, -4.3064032, -1.6051242, 1.5105386
7: -8.7219315, -7.4087892, -8.8480978, -7.3817639, -1.1045566, 1.2045701
8: 1.0368590, 2.6243343, 1.0041881, 2.5465856, -1.1251628, 1.2323132
9: -9.3750725, -7.2159300, -9.4506207, -7.3989234, -1.5208428, 1.7255750

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.39 seconds

### Candidate
type: B, layer: 3, pos: 423

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7014914, upper bound: 0.7087030
time: 7.63 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7034746, upper bound: 0.7097770
time: 4.22 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -7.3056631, -5.0999122, -7.3256593, -5.1073923, -1.7519484, 1.7828946
1: 1.9669955, 3.5556262, 1.8699479, 3.5442948, -1.1681433, 1.2760525
2: -4.9469681, -3.2823098, -4.9165564, -3.2774780, -1.1672738, 1.1391485
3: -11.0628061, -8.8965263, -11.0326729, -8.8843803, -1.6094999, 1.5664120
4: -5.5925274, -3.8488555, -5.5655499, -3.7520239, -1.5753999, 1.4432216
5: -9.0700178, -7.3113856, -9.0786505, -7.3037996, -1.7398510, 1.7442455
6: -6.5675230, -4.3262696, -6.5580783, -4.4110494, -1.5245404, 1.6110337
7: -8.8376179, -7.4093542, -8.7221069, -7.3987131, -1.1921020, 1.0672719
8: 1.0145617, 2.5453753, 1.0370350, 2.6252298, -1.2158720, 1.1182144
9: -9.4554577, -7.4482117, -9.3749599, -7.2254286, -1.7497468, 1.4690027

Time for backsubstitution: 5.62 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.37 seconds

### Candidate
type: B, layer: 3, pos: 423

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6896793, upper bound: 0.7018868
time: 4.56 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6916731, upper bound: 0.7030503
time: 4.76 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -7.3072877, -5.1018953, -7.3256593, -5.1073923, -1.7535243, 1.7815027
1: 1.8873105, 3.5444431, 1.8699479, 3.5442948, -1.2055693, 1.2183030
2: -4.9123201, -3.2756791, -4.9165564, -3.2774780, -1.1364534, 1.1471748
3: -11.0354338, -8.8966808, -11.0326729, -8.8843803, -1.5951705, 1.5755844
4: -5.5645566, -3.7567065, -5.5655499, -3.7520239, -1.4861813, 1.4822202
5: -9.0741472, -7.3128581, -9.0786505, -7.3037996, -1.7372117, 1.7365389
6: -6.5592098, -4.4138160, -6.5580783, -4.4110494, -1.5500808, 1.5438163
7: -8.7437325, -7.4088888, -8.7221069, -7.3987131, -1.1050682, 1.0857012
8: 1.0390034, 2.6184092, 1.0370350, 2.6252298, -1.1618185, 1.1544051
9: -9.3746729, -7.2596254, -9.3749599, -7.2254286, -1.5423753, 1.5084705

Time for backsubstitution: 5.50 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.37 seconds

### Candidate
type: B, layer: 3, pos: 423

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6896793, upper bound: 0.7082115
time: 4.57 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6916731, upper bound: 0.7092989
time: 4.11 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -7.3056631, -5.0999122, -7.3284597, -5.1061158, -1.7508979, 1.7836289
1: 1.9669955, 3.5556262, 1.8771482, 3.5419497, -1.1621814, 1.2773502
2: -4.9469681, -3.2823098, -4.9197788, -3.2727540, -1.1734550, 1.1407816
3: -11.0628061, -8.8965263, -11.0404987, -8.8851728, -1.6071453, 1.5741630
4: -5.5925274, -3.8488555, -5.5670023, -3.7506886, -1.5828605, 1.4564359
5: -9.0700178, -7.3113856, -9.0851812, -7.2780170, -1.7742853, 1.7672381
6: -6.5675230, -4.3262696, -6.5573807, -4.3950086, -1.5090985, 1.5970755
7: -8.8376179, -7.4093542, -8.7507820, -7.3847733, -1.2039607, 1.0947599
8: 1.0145617, 2.5453753, 1.0338163, 2.6123576, -1.2202606, 1.1357727
9: -9.4554577, -7.4482117, -9.3748531, -7.2233343, -1.7522111, 1.4698086

Time for backsubstitution: 5.59 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.38 seconds

### Candidate
type: B, layer: 3, pos: 423

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6916845, upper bound: 0.7018863
time: 4.51 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6936411, upper bound: 0.7030522
time: 4.32 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -7.3072877, -5.1018953, -7.3284597, -5.1061158, -1.7524729, 1.7822380
1: 1.8873105, 3.5444431, 1.8771482, 3.5419497, -1.2015991, 1.2196751
2: -4.9123201, -3.2756791, -4.9197788, -3.2727540, -1.1424754, 1.1497660
3: -11.0354338, -8.8966808, -11.0404987, -8.8851728, -1.5922494, 1.5832925
4: -5.5645566, -3.7567065, -5.5670023, -3.7506886, -1.4924421, 1.4948435
5: -9.0741472, -7.3128581, -9.0851812, -7.2780170, -1.7716489, 1.7595315
6: -6.5592098, -4.4138160, -6.5573807, -4.3950086, -1.5383389, 1.5292065
7: -8.7437325, -7.4088888, -8.7507820, -7.3847733, -1.1248593, 1.1154342
8: 1.0390034, 2.6184092, 1.0338163, 2.6123576, -1.1616521, 1.1717267
9: -9.3746729, -7.2596254, -9.3748531, -7.2233343, -1.5447540, 1.5096264

Time for backsubstitution: 5.52 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.40 seconds

### Candidate
type: B, layer: 3, pos: 423

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6916845, upper bound: 0.7082102
time: 4.21 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.6936411, upper bound: 0.7092979
time: 4.28 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -7.3252287, -5.1051140, -7.3349166, -5.1073093, -1.7724004, 1.7730446
1: 1.9471767, 3.5582409, 1.8612332, 3.5448451, -1.1822758, 1.2994547
2: -4.9501371, -3.2865560, -4.9183879, -3.2770500, -1.1843491, 1.1342897
3: -11.0593834, -8.8867168, -11.0326977, -8.8788509, -1.6098080, 1.5746632
4: -5.5903339, -3.8453085, -5.5663462, -3.7500286, -1.5743389, 1.4528341
5: -9.0735254, -7.3005614, -9.0798054, -7.2992620, -1.7479696, 1.7400784
6: -6.5660152, -4.3244324, -6.5588293, -4.4098725, -1.5213900, 1.6170683
7: -8.8158188, -7.4089589, -8.7221632, -7.3956971, -1.1937342, 1.0685582
8: 1.0131907, 2.5515699, 1.0362210, 2.6281543, -1.2167916, 1.1218646
9: -9.4498243, -7.4126263, -9.3757906, -7.2064238, -1.7688370, 1.4744058

Time for backsubstitution: 5.51 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.38 seconds

### Candidate
type: B, layer: 3, pos: 423

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7010766, upper bound: 0.7018864
time: 4.85 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7030502, upper bound: 0.7030507
time: 6.22 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -7.3270020, -5.1075830, -7.3349166, -5.1073093, -1.7740121, 1.7716570
1: 1.8684199, 3.5445313, 1.8612332, 3.5448451, -1.2175493, 1.2417765
2: -4.9158492, -3.2790942, -4.9183879, -3.2770500, -1.1555622, 1.1427983
3: -11.0326090, -8.8868847, -11.0326977, -8.8788509, -1.5941305, 1.5837402
4: -5.5624924, -3.7522728, -5.5663462, -3.7500286, -1.4852872, 1.4920349
5: -9.0771637, -7.3020205, -9.0798054, -7.2992620, -1.7452374, 1.7309842
6: -6.5576620, -4.4125118, -6.5588293, -4.4098725, -1.5475664, 1.5504501
7: -8.7219315, -7.4087892, -8.7221632, -7.3956971, -1.1059966, 1.0866141
8: 1.0368590, 2.6243343, 1.0362210, 2.6281543, -1.1625001, 1.1602418
9: -9.3750725, -7.2159300, -9.3757906, -7.2064238, -1.5554132, 1.5187204

Time for backsubstitution: 5.55 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.38 seconds

### Candidate
type: B, layer: 3, pos: 423

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7010766, upper bound: 0.7082772
time: 4.47 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7030502, upper bound: 0.7093580
time: 4.09 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -7.3252287, -5.1051140, -7.3376341, -5.1060295, -1.7713490, 1.7737207
1: 1.9471767, 3.5582409, 1.8678567, 3.5425491, -1.1802788, 1.2989893
2: -4.9501371, -3.2865560, -4.9215856, -3.2723317, -1.1905589, 1.1353482
3: -11.0593834, -8.8867168, -11.0405245, -8.8796797, -1.6073937, 1.5825133
4: -5.5903339, -3.8453085, -5.5677876, -3.7488368, -1.5817981, 1.4670997
5: -9.0735254, -7.3005614, -9.0863237, -7.2734866, -1.7824049, 1.7626081
6: -6.5660152, -4.3244324, -6.5581217, -4.3936558, -1.5062985, 1.6006155
7: -8.8158188, -7.4089589, -8.7508459, -7.3822765, -1.2042551, 1.0974431
8: 1.0131907, 2.5515699, 1.0329657, 2.6152620, -1.2193360, 1.1439114
9: -9.4498243, -7.4126263, -9.3756704, -7.2042341, -1.7712851, 1.4745092

Time for backsubstitution: 5.49 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.37 seconds

### Candidate
type: B, layer: 3, pos: 423

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7012960, upper bound: 0.7018879
time: 4.96 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7033174, upper bound: 0.7030513
time: 4.57 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -7.3270020, -5.1075830, -7.3376341, -5.1060295, -1.7729611, 1.7723327
1: 1.8684199, 3.5445313, 1.8678567, 3.5425491, -1.2156012, 1.2411244
2: -4.9158492, -3.2790942, -4.9215856, -3.2723317, -1.1617751, 1.1441338
3: -11.0326090, -8.8868847, -11.0405245, -8.8796797, -1.5911512, 1.5916276
4: -5.5624924, -3.7522728, -5.5677876, -3.7488368, -1.4915462, 1.5053401
5: -9.0771637, -7.3020205, -9.0863237, -7.2734866, -1.7796736, 1.7538028
6: -6.5576620, -4.4125118, -6.5581217, -4.3936558, -1.5358257, 1.5340216
7: -8.7219315, -7.4087892, -8.7508459, -7.3822765, -1.1231108, 1.1196766
8: 1.0368590, 2.6243343, 1.0329657, 2.6152620, -1.1617880, 1.1789942
9: -9.3750725, -7.2159300, -9.3756704, -7.2042341, -1.5577769, 1.5191674

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.38 seconds

### Candidate
type: B, layer: 3, pos: 423

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7012960, upper bound: 0.7082778
time: 4.71 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7033174, upper bound: 0.7093566
time: 4.70 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -7.3086243, -5.0988011, -7.3238716, -5.1049204, -1.7535853, 1.7795129
1: 1.9694457, 3.5586653, 1.9496531, 3.5578897, -1.1829526, 1.1888542
2: -4.9515362, -3.2774415, -4.9510164, -3.2846971, -1.1269088, 1.1416981
3: -11.0759735, -8.8965273, -11.0594578, -8.8842440, -1.6109409, 1.5822883
4: -5.5970941, -3.8469534, -5.5931864, -3.8444762, -1.4753828, 1.4648387
5: -9.0761414, -7.2854629, -9.0748949, -7.3023300, -1.7638688, 1.7747712
6: -6.5633383, -4.3104649, -6.5664520, -4.3233490, -1.5616689, 1.5587091
7: -8.8699198, -7.3949952, -8.8160057, -7.3991213, -1.1497221, 1.1188760
8: 1.0048432, 2.5368629, 1.0134072, 2.5522137, -1.1742055, 1.1478329
9: -9.4552622, -7.4461021, -9.4496288, -7.4162288, -1.5148921, 1.4824214

Time for backsubstitution: 5.53 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.37 seconds

### Candidate
type: B, layer: 3, pos: 423

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7182810, upper bound: 0.7287119
time: 4.63 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7195781, upper bound: 0.7291003
time: 4.94 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -7.3279648, -5.1038380, -7.3331013, -5.1048384, -1.7737474, 1.7696376
1: 1.9493752, 3.5606382, 1.9399829, 3.5586171, -1.1883399, 1.2083611
2: -4.9543581, -3.2817464, -4.9527569, -3.2842202, -1.1463718, 1.1353366
3: -11.0685892, -8.8873997, -11.0594778, -8.8787193, -1.6108966, 1.5892696
4: -5.5949545, -3.8434410, -5.5939417, -3.8427868, -1.4747167, 1.4752893
5: -9.0797663, -7.2746916, -9.0761862, -7.2977958, -1.7722569, 1.7696652
6: -6.5617738, -4.3088121, -6.5672331, -4.3220191, -1.5600743, 1.5646863
7: -8.8478403, -7.3947415, -8.8160648, -7.3959808, -1.1470661, 1.1202412
8: 1.0049062, 2.5425978, 1.0125160, 2.5551615, -1.1727645, 1.1519089
9: -9.4498444, -7.4104156, -9.4505930, -7.4009986, -1.5311170, 1.4935515

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.37 seconds

### Candidate
type: B, layer: 3, pos: 423

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7280634, upper bound: 0.7287114
time: 5.20 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7295255, upper bound: 0.7291003
time: 4.65 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -7.3086243, -5.0988011, -7.3266726, -5.1036439, -1.7742620, 1.8019810
1: 1.9694457, 3.5586653, 1.9521410, 3.5602832, -1.1778965, 1.1896882
2: -4.9515362, -3.2774415, -4.9552641, -3.2799335, -1.1292288, 1.1398636
3: -11.0759735, -8.8965273, -11.0686626, -8.8849545, -1.6071239, 1.5890851
4: -5.5970941, -3.8469534, -5.5977998, -3.8426080, -1.4675426, 1.4634948
5: -9.0761414, -7.2854629, -9.0811110, -7.2765565, -1.7530713, 1.7524385
6: -6.5633383, -4.3104649, -6.5622206, -4.3077168, -1.5923333, 1.5862863
7: -8.8699198, -7.3949952, -8.8480320, -7.3848505, -1.1350327, 1.1158986
8: 1.0048432, 2.5368629, 1.0050888, 2.5436807, -1.1935835, 1.1870673
9: -9.4552622, -7.4461021, -9.4496727, -7.4141750, -1.5220494, 1.4893019

Time for backsubstitution: 5.49 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.38 seconds

### Candidate
type: B, layer: 3, pos: 423

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7164464, upper bound: 0.7287118
time: 4.15 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7176982, upper bound: 0.7291004
time: 4.56 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -7.3279648, -5.1038380, -7.3358173, -5.1035595, -1.7958455, 1.7920413
1: 1.9493752, 3.5606382, 1.9421654, 3.5610182, -1.1873045, 1.2137659
2: -4.9543581, -3.2817464, -4.9569855, -3.2794609, -1.1468287, 1.1338941
3: -11.0685892, -8.8873997, -11.0686855, -8.8794651, -1.6073856, 1.5957532
4: -5.5949545, -3.8434410, -5.5985651, -3.8409219, -1.4667964, 1.4739540
5: -9.0797663, -7.2746916, -9.0823879, -7.2720332, -1.7612300, 1.7473593
6: -6.5617738, -4.3088121, -6.5629883, -4.3064032, -1.5892525, 1.5934732
7: -8.8478403, -7.3947415, -8.8480978, -7.3817639, -1.1345043, 1.1171930
8: 1.0049062, 2.5425978, 1.0041881, 2.5465856, -1.1952195, 1.1887755
9: -9.4498444, -7.4104156, -9.4506207, -7.3989234, -1.5394154, 1.4985700

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1269
type: B, layer: 3, pos: 2832

Time for candidate selection: 0.38 seconds

### Candidate
type: B, layer: 3, pos: 423

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7258294, upper bound: 0.7287144
time: 4.49 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7270886, upper bound: 0.7291004
time: 5.32 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 15.68 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 15.68
Output dim: 1, lower bound: -0.7182810, upper bound: 0.7286229
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 15.68
Output dim: 1, lower bound: -0.7195781, upper bound: 0.7295276
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 15.68
Output dim: 1, lower bound: -0.7280634, upper bound: 0.7286232
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 15.68
Output dim: 1, lower bound: -0.7295255, upper bound: 0.7295276
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 15.68
Output dim: 1, lower bound: -0.7213599, upper bound: 0.7286247
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 15.68
Output dim: 1, lower bound: -0.7225311, upper bound: 0.7295276
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 15.68
Output dim: 1, lower bound: -0.7279946, upper bound: 0.7286237
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 15.68
Output dim: 1, lower bound: -0.7290980, upper bound: 0.7295265
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 15.68
Output dim: 1, lower bound: -0.6898078, upper bound: 0.7086875
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 15.68
Output dim: 1, lower bound: -0.6918135, upper bound: 0.7097632
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 15.68
Output dim: 1, lower bound: -0.7011202, upper bound: 0.7087028
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 15.68
Output dim: 1, lower bound: -0.7030954, upper bound: 0.7097786
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 15.68
Output dim: 1, lower bound: -0.6919354, upper bound: 0.7086875
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 15.68
Output dim: 1, lower bound: -0.6939092, upper bound: 0.7097614
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 15.68
Output dim: 1, lower bound: -0.7014914, upper bound: 0.7087030
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 15.68
Output dim: 1, lower bound: -0.7034746, upper bound: 0.7097770
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 15.68
Output dim: 1, lower bound: -0.6896793, upper bound: 0.7018868
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 15.68
Output dim: 1, lower bound: -0.6916731, upper bound: 0.7030503
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 15.68
Output dim: 1, lower bound: -0.6896793, upper bound: 0.7082115
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 15.68
Output dim: 1, lower bound: -0.6916731, upper bound: 0.7092989
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 15.68
Output dim: 1, lower bound: -0.6916845, upper bound: 0.7018863
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 15.68
Output dim: 1, lower bound: -0.6936411, upper bound: 0.7030522
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 15.68
Output dim: 1, lower bound: -0.6916845, upper bound: 0.7082102
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 15.68
Output dim: 1, lower bound: -0.6936411, upper bound: 0.7092979
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 15.68
Output dim: 1, lower bound: -0.7010766, upper bound: 0.7018864
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 15.68
Output dim: 1, lower bound: -0.7030502, upper bound: 0.7030507
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 15.68
Output dim: 1, lower bound: -0.7010766, upper bound: 0.7082772
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 15.68
Output dim: 1, lower bound: -0.7030502, upper bound: 0.7093580
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 15.68
Output dim: 1, lower bound: -0.7012960, upper bound: 0.7018879
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 15.68
Output dim: 1, lower bound: -0.7033174, upper bound: 0.7030513
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 15.68
Output dim: 1, lower bound: -0.7012960, upper bound: 0.7082778
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 15.68
Output dim: 1, lower bound: -0.7033174, upper bound: 0.7093566
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 15.68
Output dim: 1, lower bound: -0.7182810, upper bound: 0.7287119
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 15.68
Output dim: 1, lower bound: -0.7195781, upper bound: 0.7291003
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 15.68
Output dim: 1, lower bound: -0.7280634, upper bound: 0.7287114
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 15.68
Output dim: 1, lower bound: -0.7295255, upper bound: 0.7291003
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 15.68
Output dim: 1, lower bound: -0.7164464, upper bound: 0.7287118
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 15.68
Output dim: 1, lower bound: -0.7176982, upper bound: 0.7291004
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 15.68
Output dim: 1, lower bound: -0.7258294, upper bound: 0.7287144
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 15.68
Output dim: 1, lower bound: -0.7270886, upper bound: 0.7291004
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.68
Output dim: 1, lower bound: -0.6970733, upper bound: 0.7156211
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.68
Output dim: 1, lower bound: -0.7087000, upper bound: 0.7156218
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.68
Output dim: 1, lower bound: -0.6970733, upper bound: 0.7156239
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.68
Output dim: 1, lower bound: -0.7087000, upper bound: 0.7156238
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.68
Output dim: 1, lower bound: -0.6968702, upper bound: 0.7107676
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.68
Output dim: 1, lower bound: -0.6968702, upper bound: 0.7156219
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.68
Output dim: 1, lower bound: -0.6968724, upper bound: 0.7107694
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.68
Output dim: 1, lower bound: -0.6968724, upper bound: 0.7156220
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.68
Output dim: 1, lower bound: -0.7085496, upper bound: 0.7107841
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.68
Output dim: 1, lower bound: -0.7085496, upper bound: 0.7156221
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.68
Output dim: 1, lower bound: -0.7085496, upper bound: 0.7107843
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.68
Output dim: 1, lower bound: -0.7085496, upper bound: 0.7156223
Binary search (step 2): status=Status.UNKNOWN, k_low=4, k_high=4, k_mid=4, eps_mid=0.0156250, abs_max=1.2436916828155518
rel_dist={1: [-0.7636396766977613, 0.7636396766977596]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01171875
execution time: 2412.88 seconds
