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
execution time: IAR + LP analysis = 15.60 + 32.62 = 48.22 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3551.78 seconds, max iter: 100)

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
Binary search time: 152.17 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.01171875


# Individual Split (IS_dual) starts
Time budget: 3399.61 seconds

## Binary search (step 0) starts
Candidate k: 8, corresponding eps: 0.0312500


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 1928
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 655
type: B, layer: 3, pos: 655
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.40 seconds

### Candidate
type: B, layer: 3, pos: 1928

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0677377, upper bound: 1.0828177
time: 3.91 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0828174, upper bound: 1.0828180
time: 3.63 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 7.95 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 7.95
Output dim: 1, lower bound: -1.0677377, upper bound: 1.0828177
IS_B2, status: Status.UNKNOWN, split count: 1, time: 7.95
Output dim: 1, lower bound: -1.0828174, upper bound: 1.0828180

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -7.3412457, -5.1029596, -7.2890615, -5.1841650, -2.1570807, 2.1733029
1: 1.9362400, 3.5890326, 1.9458289, 3.5798233, -1.4777291, 1.4631557
2: -4.9621353, -3.2754836, -4.9594469, -3.2859809, -1.4252143, 1.4343807
3: -11.0800304, -8.8735428, -11.0325260, -8.8911295, -1.9447856, 1.8936982
4: -5.6305523, -3.8394473, -5.6086173, -3.8460803, -1.7783628, 1.7691700
5: -9.0882244, -7.2591839, -9.0567703, -7.3194046, -1.7688198, 1.7975864
6: -6.5653353, -4.2852068, -6.5046525, -4.2907782, -1.9273176, 1.8910627
7: -8.8574305, -7.3770838, -8.8444023, -7.4142337, -1.2863379, 1.4339504
8: 0.9680390, 2.5485387, 0.9678707, 2.5229740, -1.4556060, 1.4248371
9: -9.4929600, -7.3942938, -9.4739857, -7.4211969, -1.8989854, 1.8977652

Time for backsubstitution: 5.67 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 655
type: B, layer: 3, pos: 655
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 1725

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0483218, upper bound: 0.9654236
time: 3.85 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0141605, upper bound: 1.0266870
time: 3.78 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -7.3412457, -5.1029596, -7.3349719, -5.1344824, -2.1396966, 2.2320123
1: 1.9362400, 3.5890326, 1.9418974, 3.5842206, -1.5202889, 1.4796607
2: -4.9621353, -3.2754836, -4.9584117, -3.2864370, -1.4255910, 1.4340074
3: -11.0800304, -8.8735428, -11.0543442, -8.8779135, -2.0035644, 1.9761443
4: -5.6305523, -3.8394473, -5.6055589, -3.8417616, -1.7887907, 1.7661116
5: -9.0882244, -7.2591839, -9.0845490, -7.2844133, -1.8038111, 1.8253651
6: -6.5653353, -4.2852068, -6.5309668, -4.2874689, -1.9472842, 1.9151096
7: -8.8574305, -7.3770838, -8.8546734, -7.3944950, -1.4203730, 1.4350073
8: 0.9680390, 2.5485387, 0.9718418, 2.5160675, -1.4917920, 1.4713802
9: -9.4929600, -7.3942938, -9.4891787, -7.4034848, -1.9190397, 1.9260166

Time for backsubstitution: 5.66 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1928
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 655
type: B, layer: 3, pos: 655
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 1928

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0828174, upper bound: 1.0677362
time: 3.81 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0828174, upper bound: 1.0828177
time: 3.94 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 13.62 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 13.62
Output dim: 1, lower bound: -1.0483218, upper bound: 0.9654236
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 13.62
Output dim: 1, lower bound: -1.0141605, upper bound: 1.0266870
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 13.62
Output dim: 1, lower bound: -1.0828174, upper bound: 1.0677362
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 13.62
Output dim: 1, lower bound: -1.0828174, upper bound: 1.0828177

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -7.3293495, -5.1079168, -7.2890615, -5.1841650, -2.1451845, 2.1694906
1: 1.9414697, 3.5673254, 1.9458289, 3.5798233, -1.4718606, 1.4324789
2: -4.9334102, -3.2776134, -4.9594469, -3.2859809, -1.3984058, 1.4315237
3: -11.0793180, -8.8735704, -11.0325260, -8.8911295, -1.9445491, 1.8926775
4: -5.6190424, -3.8701024, -5.6086173, -3.8460803, -1.7729621, 1.7385149
5: -9.0876713, -7.2768574, -9.0567703, -7.3194046, -1.7682667, 1.7799129
6: -6.5546036, -4.3242135, -6.5046525, -4.2907782, -1.9213123, 1.8472862
7: -8.8574152, -7.3813982, -8.8444023, -7.4142337, -1.2860131, 1.4272823
8: 0.9689741, 2.5476437, 0.9678707, 2.5229740, -1.4518018, 1.4228265
9: -9.4155626, -7.3969879, -9.4739857, -7.4211969, -1.8257618, 1.8956661

Time for backsubstitution: 5.66 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 655
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 1683

## Relational analysis of IS_B1_A1_B1

### Relational analysis result of IS_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9916278, upper bound: 0.9192362
time: 3.71 seconds

## Relational analysis of IS_B1_A1_B2

### Relational analysis result of IS_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9813278, upper bound: 0.9005962
time: 3.37 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -7.3357038, -5.0759382, -7.2875547, -5.1841850, -2.1515188, 2.2076218
1: 1.8803887, 3.5882750, 1.9458549, 3.5788798, -1.5416002, 1.4433670
2: -4.9466424, -3.1960044, -4.9555821, -3.2860007, -1.4166114, 1.5222431
3: -11.0801077, -8.8732910, -11.0324898, -8.8911428, -1.9484773, 1.8922212
4: -5.7279572, -3.8749826, -5.6085587, -3.8595338, -1.8684235, 1.7335761
5: -9.1399250, -7.2773123, -9.0567236, -7.3235540, -1.8163710, 1.7794113
6: -6.6605115, -4.3014612, -6.5046034, -4.2939868, -2.0443482, 1.8872979
7: -8.8574581, -7.3757858, -8.8443985, -7.4160633, -1.2907743, 1.4339826
8: 0.9713149, 2.5288334, 0.9679461, 2.5194798, -1.4521408, 1.4263761
9: -9.4831629, -7.1796589, -9.4704971, -7.4212136, -1.8716936, 2.1076107

Time for backsubstitution: 5.79 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 655
type: B, layer: 3, pos: 655
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 1683

## Relational analysis of IS_B1_A2_B1

### Relational analysis result of IS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9550713, upper bound: 0.9809803
time: 3.69 seconds

## Relational analysis of IS_B1_A2_B2

### Relational analysis result of IS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9409419, upper bound: 0.9597574
time: 3.60 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -7.2890615, -5.1841650, -7.3349719, -5.1344824, -2.1545792, 2.1508069
1: 1.9458289, 3.5798233, 1.9418974, 3.5842206, -1.4442601, 1.4581437
2: -4.9594469, -3.2859809, -4.9584117, -3.2864370, -1.4243202, 1.4225727
3: -11.0325260, -8.8911295, -11.0543442, -8.8779135, -1.8879714, 1.9381280
4: -5.6086173, -3.8460803, -5.6055589, -3.8417616, -1.7668557, 1.7594786
5: -9.0567703, -7.3194046, -9.0845490, -7.2844133, -1.7723570, 1.7651443
6: -6.5046525, -4.2907782, -6.5309668, -4.2874689, -1.8889771, 1.9055462
7: -8.8444023, -7.4142337, -8.8546734, -7.3944950, -1.4206340, 1.2678874
8: 0.9678707, 2.5229740, 0.9718418, 2.5160675, -1.4100838, 1.4523122
9: -9.4739857, -7.4211969, -9.4891787, -7.4034848, -1.8924990, 1.8964744

Time for backsubstitution: 5.67 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 655
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 1725

## Relational analysis of IS_B2_A1_B1

### Relational analysis result of IS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9497255, upper bound: 1.0483220
time: 4.18 seconds

## Relational analysis of IS_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0141605, upper bound: 1.0141604
time: 4.07 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -7.3349719, -5.1344824, -7.3349719, -5.1344824, -2.1348639, 2.1348639
1: 1.9418974, 3.5842206, 1.9418974, 3.5842206, -1.5060577, 1.5060577
2: -4.9584117, -3.2864370, -4.9584117, -3.2864370, -1.4229493, 1.4229493
3: -11.0543442, -8.8779135, -11.0543442, -8.8779135, -1.9724684, 1.9724684
4: -5.6055589, -3.8417616, -5.6055589, -3.8417616, -1.7637973, 1.7637973
5: -9.0845490, -7.2844133, -9.0845490, -7.2844133, -1.8001356, 1.8001356
6: -6.5309668, -4.2874689, -6.5309668, -4.2874689, -1.9130142, 1.9130142
7: -8.8546734, -7.3944950, -8.8546734, -7.3944950, -1.4181490, 1.4181488
8: 0.9718418, 2.5160675, 0.9718418, 2.5160675, -1.4880774, 1.4880772
9: -9.4891787, -7.4034848, -9.4891787, -7.4034848, -1.9165287, 1.9165287

Time for backsubstitution: 5.68 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 1683
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 655
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 1725

## Relational analysis of IS_B2_A2_B1

### Relational analysis result of IS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9497255, upper bound: 1.0613192
time: 3.89 seconds

## Relational analysis of IS_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.0141605, upper bound: 1.0262804
time: 3.87 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 13.66 seconds
IS_B1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 13.66
Output dim: 1, lower bound: -0.9916278, upper bound: 0.9192362
IS_B1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 13.66
Output dim: 1, lower bound: -0.9813278, upper bound: 0.9005962
IS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 13.66
Output dim: 1, lower bound: -0.9550713, upper bound: 0.9809803
IS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 13.66
Output dim: 1, lower bound: -0.9409419, upper bound: 0.9597574
IS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 13.66
Output dim: 1, lower bound: -0.9497255, upper bound: 1.0483220
IS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 13.66
Output dim: 1, lower bound: -1.0141605, upper bound: 1.0141604
IS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 13.66
Output dim: 1, lower bound: -0.9497255, upper bound: 1.0613192
IS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 13.66
Output dim: 1, lower bound: -1.0141605, upper bound: 1.0262804

## BFS IS instance: IS_B1_A1_B1

### Backsubstitution after applying IS history:
0: -7.3293495, -5.1079168, -7.2883701, -5.1846838, -2.1446657, 2.1689365
1: 1.9414697, 3.5673254, 1.9466350, 3.5629015, -1.4528196, 1.4316189
2: -4.9334102, -3.2776134, -4.9566669, -3.2875462, -1.3972707, 1.4223294
3: -11.0793180, -8.8735704, -11.0318794, -8.8928223, -1.9432864, 1.8903310
4: -5.6190424, -3.8701024, -5.5839472, -3.8463845, -1.7726579, 1.7138448
5: -9.0876713, -7.2768574, -9.0527601, -7.3194895, -1.7681818, 1.7759027
6: -6.5546036, -4.3242135, -6.5037718, -4.2940197, -1.9158609, 1.8465385
7: -8.8574152, -7.3813982, -8.8417425, -7.4146304, -1.2841096, 1.4155309
8: 0.9689741, 2.5476437, 0.9823718, 2.5220494, -1.4508824, 1.4111042
9: -9.4155626, -7.3969879, -9.4325314, -7.4215832, -1.8253889, 1.8446958

Time for backsubstitution: 5.66 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 655
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 1683
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 655

## Relational analysis of IS_B1_A1_B1_A1

### Relational analysis result of IS_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9747681, upper bound: 0.9054462
time: 3.89 seconds

## Relational analysis of IS_B1_A1_B1_A2

### Relational analysis result of IS_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9747681, upper bound: 0.9031352
time: 3.96 seconds

## BFS IS instance: IS_B1_A1_B2

### Backsubstitution after applying IS history:
0: -7.3290529, -5.1082439, -7.2918048, -5.1862307, -2.1428223, 2.1728721
1: 1.9418185, 3.5629320, 1.8706703, 3.5473242, -1.4586515, 1.5111556
2: -4.9285603, -3.2778094, -4.9242573, -3.2798150, -1.4415233, 1.4144576
3: -11.0761414, -8.8739796, -11.0040150, -8.8928928, -1.9484630, 1.8744819
4: -5.6116538, -3.8701768, -5.5557842, -3.7440040, -1.8676498, 1.6856074
5: -9.0866585, -7.2770567, -9.0556183, -7.3206935, -1.7659650, 1.7785616
6: -6.5541458, -4.3347721, -6.5048437, -4.3718333, -1.8726606, 1.8621595
7: -8.8462238, -7.3815389, -8.7486925, -7.4096055, -1.3418999, 1.3671536
8: 0.9741740, 2.5474973, 1.0088367, 2.5907869, -1.5075159, 1.4154758
9: -9.4015408, -7.3970251, -9.3555136, -7.2289448, -2.0642915, 1.8550653

Time for backsubstitution: 5.71 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 655
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 1928
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 1683
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 655

## Relational analysis of IS_B1_A1_B2_A1

### Relational analysis result of IS_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9612677, upper bound: 0.8842575
time: 3.60 seconds

## Relational analysis of IS_B1_A1_B2_A2

### Relational analysis result of IS_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9612677, upper bound: 0.8841185
time: 3.63 seconds

## BFS IS instance: IS_B1_A2_B1

### Backsubstitution after applying IS history:
0: -7.3357038, -5.0759382, -7.2868624, -5.1847034, -2.1510005, 2.2070682
1: 1.8803887, 3.5882750, 1.9466591, 3.5619581, -1.5225415, 1.4425068
2: -4.9466424, -3.1960044, -4.9528751, -3.2875631, -1.4154768, 1.5130496
3: -11.0801077, -8.8732910, -11.0318432, -8.8928366, -1.9472141, 1.8898735
4: -5.7279572, -3.8749826, -5.5838881, -3.8599126, -1.8680446, 1.7089055
5: -9.1399250, -7.2773123, -9.0527153, -7.3236380, -1.8162870, 1.7754030
6: -6.6605115, -4.3014612, -6.5037260, -4.2974043, -2.0390263, 1.8865514
7: -8.8574581, -7.3757858, -8.8417397, -7.4164562, -1.2888551, 1.4222312
8: 0.9713149, 2.5288334, 0.9824457, 2.5185556, -1.4512208, 1.4146609
9: -9.4831629, -7.1796589, -9.4294062, -7.4215980, -1.8713207, 2.0563426

Time for backsubstitution: 5.69 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 655
type: B, layer: 3, pos: 655
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 1683
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 655

## Relational analysis of IS_B1_A2_B1_A1

### Relational analysis result of IS_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9395375, upper bound: 0.9624910
time: 3.78 seconds

## Relational analysis of IS_B1_A2_B1_A2

### Relational analysis result of IS_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9395375, upper bound: 0.9623082
time: 3.76 seconds

## BFS IS instance: IS_B1_A2_B2

### Backsubstitution after applying IS history:
0: -7.3353906, -5.0762572, -7.2903109, -5.1862478, -2.1491427, 2.2110133
1: 1.8807344, 3.5838737, 1.8706965, 3.5463653, -1.5289347, 1.5221684
2: -4.9424448, -3.1961975, -4.9204917, -3.2798326, -1.4597383, 1.5053647
3: -11.0769329, -8.8737001, -11.0039759, -8.8929081, -1.9523926, 1.8739977
4: -5.7207127, -3.8751087, -5.5557218, -3.7552865, -1.9654262, 1.6806130
5: -9.1388817, -7.2775197, -9.0555630, -7.3248386, -1.8140430, 1.7780433
6: -6.6600642, -4.3119941, -6.5047956, -4.3752627, -1.9962907, 1.9035432
7: -8.8462648, -7.3759332, -8.7486906, -7.4113126, -1.3468430, 1.3727574
8: 0.9765105, 2.5286875, 1.0089264, 2.5872927, -1.5077395, 1.4203453
9: -9.4677305, -7.1796913, -9.3513823, -7.2289605, -2.1101894, 2.0672469

Time for backsubstitution: 5.64 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 655
type: B, layer: 3, pos: 655
type: A, layer: 3, pos: 1928
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 1683
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 655

## Relational analysis of IS_B1_A2_B2_A1

### Relational analysis result of IS_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9231270, upper bound: 0.9398825
time: 3.67 seconds

## Relational analysis of IS_B1_A2_B2_A2

### Relational analysis result of IS_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9231270, upper bound: 0.9405049
time: 3.62 seconds

## BFS IS instance: IS_B2_A1_B1

### Backsubstitution after applying IS history:
0: -7.2890615, -5.1841650, -7.3230686, -5.1392536, -2.1498079, 2.1389036
1: 1.9458289, 3.5798233, 1.9471262, 3.5624185, -1.4136086, 1.4524031
2: -4.9594469, -3.2859809, -4.9296737, -3.2884791, -1.4215231, 1.3957540
3: -11.0325260, -8.8911295, -11.0536289, -8.8779402, -1.8870783, 1.9379072
4: -5.6086173, -3.8460803, -5.5940585, -3.8728399, -1.7357774, 1.7479782
5: -9.0567703, -7.3194046, -9.0840378, -7.3011351, -1.7556353, 1.7646332
6: -6.5046525, -4.2907782, -6.5199671, -4.3264866, -1.8451803, 1.8987012
7: -8.8444023, -7.4142337, -8.8546562, -7.3986664, -1.4139173, 1.2675731
8: 0.9678707, 2.5229740, 0.9727139, 2.5151730, -1.4085836, 1.4485064
9: -9.4739857, -7.4211969, -9.4114399, -7.4061780, -1.8903525, 1.8232317

Time for backsubstitution: 5.65 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1683
type: B, layer: 3, pos: 1683
type: B, layer: 3, pos: 655
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 1683

## Relational analysis of IS_B2_A1_B1_A1

### Relational analysis result of IS_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9192359, upper bound: 0.9916279
time: 3.82 seconds

## Relational analysis of IS_B2_A1_B1_A2

### Relational analysis result of IS_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9005962, upper bound: 0.9813274
time: 3.62 seconds

## BFS IS instance: IS_B2_A1_B2

### Backsubstitution after applying IS history:
0: -7.2875547, -5.1841850, -7.3294983, -5.1072264, -2.1803284, 2.1453133
1: 1.9458549, 3.5788798, 1.8860457, 3.5832958, -1.4245796, 1.5222793
2: -4.9555821, -3.2860007, -4.9430962, -3.2060246, -1.5133910, 1.4139619
3: -11.0324898, -8.8911428, -11.0544147, -8.8776598, -1.8866258, 1.9418364
4: -5.6085587, -3.8595338, -5.7035041, -3.8766904, -1.7318683, 1.8439703
5: -9.0567236, -7.3235540, -9.1362724, -7.3025498, -1.7541738, 1.8127184
6: -6.5046034, -4.2939868, -6.6270022, -4.3036032, -1.8853512, 2.0216310
7: -8.8443985, -7.4160633, -8.8546991, -7.3931198, -1.4206381, 1.2724757
8: 0.9679461, 2.5194798, 0.9749146, 2.4963598, -1.4121320, 1.4489295
9: -9.4704971, -7.4212136, -9.4795465, -7.1889610, -2.1025820, 1.8692918

Time for backsubstitution: 5.70 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1683
type: B, layer: 3, pos: 655
type: A, layer: 3, pos: 655
type: B, layer: 3, pos: 1683
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 1683

## Relational analysis of IS_B2_A1_B2_A1

### Relational analysis result of IS_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9809800, upper bound: 0.9550711
time: 3.97 seconds

## Relational analysis of IS_B2_A1_B2_A2

### Relational analysis result of IS_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9597573, upper bound: 0.9409420
time: 3.66 seconds

## BFS IS instance: IS_B2_A2_B1

### Backsubstitution after applying IS history:
0: -7.3349719, -5.1344824, -7.3230686, -5.1392536, -2.1308446, 2.1208463
1: 1.9418974, 3.5842206, 1.9471262, 3.5624185, -1.4753296, 1.5000875
2: -4.9584117, -3.2864370, -4.9296737, -3.2884791, -1.4201469, 1.3962251
3: -11.0543442, -8.8779135, -11.0536289, -8.8779402, -1.9713883, 1.9722486
4: -5.6055589, -3.8417616, -5.5940585, -3.8728399, -1.7327189, 1.7522969
5: -9.0845490, -7.2844133, -9.0840378, -7.3011351, -1.7834139, 1.7996244
6: -6.5309668, -4.2874689, -6.5199671, -4.3264866, -1.8670740, 1.9061689
7: -8.8546734, -7.3944950, -8.8546562, -7.3986664, -1.4114490, 1.4178321
8: 0.9718418, 2.5160675, 0.9727139, 2.5151730, -1.4841309, 1.4843040
9: -9.4891787, -7.4034848, -9.4114399, -7.4061780, -1.9143457, 1.8432863

Time for backsubstitution: 5.67 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1683
type: B, layer: 3, pos: 1683
type: B, layer: 3, pos: 655
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 1683

## Relational analysis of IS_B2_A2_B1_A1

### Relational analysis result of IS_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9037067, upper bound: 1.0053141
time: 3.78 seconds

## Relational analysis of IS_B2_A2_B1_A2

### Relational analysis result of IS_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8817847, upper bound: 0.9948397
time: 3.67 seconds

## BFS IS instance: IS_B2_A2_B2

### Backsubstitution after applying IS history:
0: -7.3336630, -5.1345015, -7.3294983, -5.1072264, -2.1684065, 2.1272950
1: 1.9419210, 3.5832829, 1.8860457, 3.5832958, -1.4867878, 1.5720098
2: -4.9546742, -3.2864552, -4.9430962, -3.2060246, -1.5117269, 1.4143380
3: -11.0542994, -8.8779240, -11.0544147, -8.8776598, -1.9708200, 1.9761829
4: -5.6055007, -3.8533823, -5.7035041, -3.8766904, -1.7288103, 1.8501217
5: -9.0844917, -7.2885647, -9.1362724, -7.3025498, -1.7819419, 1.8477077
6: -6.5309186, -4.2908506, -6.6270022, -4.3036032, -1.9136581, 2.0307238
7: -8.8546705, -7.3963089, -8.8546991, -7.3931198, -1.4180429, 1.4225547
8: 0.9719172, 2.5125737, 0.9749146, 2.4963598, -1.4830642, 1.4849143
9: -9.4857025, -7.4034963, -9.4795465, -7.1889610, -2.1195669, 1.8894424

Time for backsubstitution: 5.66 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1683
type: B, layer: 3, pos: 655
type: A, layer: 3, pos: 655
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 1683

## Relational analysis of IS_B2_A2_B2_A1

### Relational analysis result of IS_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9642876, upper bound: 0.9705798
time: 3.90 seconds

## Relational analysis of IS_B2_A2_B2_A2

### Relational analysis result of IS_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9409418, upper bound: 0.9574087
time: 3.51 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 13.28 seconds
IS_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 13.28
Output dim: 1, lower bound: -0.9747681, upper bound: 0.9054462
IS_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 13.28
Output dim: 1, lower bound: -0.9747681, upper bound: 0.9031352
IS_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 13.28
Output dim: 1, lower bound: -0.9612677, upper bound: 0.8842575
IS_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 13.28
Output dim: 1, lower bound: -0.9612677, upper bound: 0.8841185
IS_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 13.28
Output dim: 1, lower bound: -0.9395375, upper bound: 0.9624910
IS_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 13.28
Output dim: 1, lower bound: -0.9395375, upper bound: 0.9623082
IS_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 13.28
Output dim: 1, lower bound: -0.9231270, upper bound: 0.9398825
IS_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 13.28
Output dim: 1, lower bound: -0.9231270, upper bound: 0.9405049
IS_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 13.28
Output dim: 1, lower bound: -0.9192359, upper bound: 0.9916279
IS_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 13.28
Output dim: 1, lower bound: -0.9005962, upper bound: 0.9813274
IS_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 13.28
Output dim: 1, lower bound: -0.9809800, upper bound: 0.9550711
IS_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 13.28
Output dim: 1, lower bound: -0.9597573, upper bound: 0.9409420
IS_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 13.28
Output dim: 1, lower bound: -0.9037067, upper bound: 1.0053141
IS_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 13.28
Output dim: 1, lower bound: -0.8817847, upper bound: 0.9948397
IS_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 13.28
Output dim: 1, lower bound: -0.9642876, upper bound: 0.9705798
IS_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 13.28
Output dim: 1, lower bound: -0.9409418, upper bound: 0.9574087

## BFS IS instance: IS_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -7.3235512, -5.1092343, -7.2883701, -5.1846838, -2.1388674, 2.1635377
1: 1.9421787, 3.5543673, 1.9466350, 3.5629015, -1.4506366, 1.4232900
2: -4.9268980, -3.2843523, -4.9566669, -3.2875462, -1.3941061, 1.4138126
3: -11.0597095, -8.8754940, -11.0318794, -8.8928223, -1.9247231, 1.8865490
4: -5.6076269, -3.8724446, -5.5839472, -3.8463845, -1.7560673, 1.7115026
5: -9.0807152, -7.3146591, -9.0527601, -7.3194895, -1.7612257, 1.7381010
6: -6.5577707, -4.3584986, -6.5037718, -4.2940197, -1.9275472, 1.8282752
7: -8.8186169, -7.3968081, -8.8417425, -7.4146304, -1.2553554, 1.3967621
8: 0.9978461, 2.5559621, 0.9823718, 2.5220494, -1.4136691, 1.4021177
9: -9.4142237, -7.4004898, -9.4325314, -7.4215832, -1.8204141, 1.8402083

Time for backsubstitution: 5.71 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 655
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 1683
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 1725

## Relational analysis of IS_B1_A1_B1_A1_B1

### Relational analysis result of IS_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9744456, upper bound: 0.9054462
time: 3.80 seconds

## Relational analysis of IS_B1_A1_B1_A1_B2

### Relational analysis result of IS_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9744456, upper bound: 0.9054475
time: 3.67 seconds

## BFS IS instance: IS_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -7.3262653, -5.1080284, -7.2883701, -5.1846838, -2.1415815, 2.1624801
1: 1.9450760, 3.5560288, 1.9466350, 3.5629015, -1.4497411, 1.4349105
2: -4.9310536, -3.2795374, -4.9566669, -3.2875462, -1.3960464, 1.4189563
3: -11.0687599, -8.8762255, -11.0318794, -8.8928223, -1.9398127, 1.8886850
4: -5.6125240, -3.8706160, -5.5839472, -3.8463845, -1.7550960, 1.7133312
5: -9.0870552, -7.2889376, -9.0527601, -7.3194895, -1.7675657, 1.7638226
6: -6.5535812, -4.3428826, -6.5037718, -4.2940197, -1.9134133, 1.8666639
7: -8.8499956, -7.3828640, -8.8417425, -7.4146304, -1.2599881, 1.4148657
8: 0.9901543, 2.5474091, 0.9823718, 2.5220494, -1.4681823, 1.4067051
9: -9.4143372, -7.3984575, -9.4325314, -7.4215832, -1.8287070, 1.8424590

Time for backsubstitution: 5.72 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 1928
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 1725

## Relational analysis of IS_B1_A1_B1_A2_B1

### Relational analysis result of IS_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9744456, upper bound: 0.9031351
time: 3.85 seconds

## Relational analysis of IS_B1_A1_B1_A2_B2

### Relational analysis result of IS_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9744456, upper bound: 0.9031367
time: 3.90 seconds

## BFS IS instance: IS_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -7.3232536, -5.1095614, -7.2918048, -5.1862307, -2.1370230, 2.1674576
1: 1.9425373, 3.5503826, 1.8706703, 3.5473242, -1.4564672, 1.5028086
2: -4.9222851, -3.2845514, -4.9242573, -3.2798150, -1.4385774, 1.4059381
3: -11.0565290, -8.8758936, -11.0040150, -8.8928928, -1.9298997, 1.8706586
4: -5.6002426, -3.8725188, -5.5557842, -3.7440040, -1.8527522, 1.6832654
5: -9.0796633, -7.3148575, -9.0556183, -7.3206935, -1.7589698, 1.7407608
6: -6.5573053, -4.3690581, -6.5048437, -4.3718333, -1.8843489, 1.8438458
7: -8.8078022, -7.3969421, -8.7486925, -7.4096055, -1.3134246, 1.3517504
8: 1.0026622, 2.5558147, 1.0088367, 2.5907869, -1.4710727, 1.4064705
9: -9.4002028, -7.4005260, -9.3555136, -7.2289448, -2.0593176, 1.8505774

Time for backsubstitution: 5.70 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 655
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 1683
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 1725

## Relational analysis of IS_B1_A1_B2_A1_B1

### Relational analysis result of IS_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9612677, upper bound: 0.8842575
time: 3.83 seconds

## Relational analysis of IS_B1_A1_B2_A1_B2

### Relational analysis result of IS_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9612677, upper bound: 0.8842575
time: 3.69 seconds

## BFS IS instance: IS_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -7.3259716, -5.1083546, -7.2918048, -5.1862307, -2.1397409, 2.1664000
1: 1.9454250, 3.5516279, 1.8706703, 3.5473242, -1.4555731, 1.5172842
2: -4.9262028, -3.2797363, -4.9242573, -3.2798150, -1.4402983, 1.4114082
3: -11.0655041, -8.8766336, -11.0040150, -8.8928928, -1.9437103, 1.8728054
4: -5.6051173, -3.8706903, -5.5557842, -3.7440040, -1.8486915, 1.6850939
5: -9.0860386, -7.2891359, -9.0556183, -7.3206935, -1.7653451, 1.7664824
6: -6.5531225, -4.3534412, -6.5048437, -4.3718333, -1.8702126, 1.8848491
7: -8.8388042, -7.3830013, -8.7486925, -7.4096055, -1.3247859, 1.3656912
8: 0.9953523, 2.5472646, 1.0088367, 2.5907869, -1.5269141, 1.4110765
9: -9.4003162, -7.3984938, -9.3555136, -7.2289448, -2.0675409, 1.8528287

Time for backsubstitution: 5.76 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 1928
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 1725

## Relational analysis of IS_B1_A1_B2_A2_B1

### Relational analysis result of IS_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9612677, upper bound: 0.8841187
time: 3.69 seconds

## Relational analysis of IS_B1_A1_B2_A2_B2

### Relational analysis result of IS_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9612677, upper bound: 0.8841185
time: 3.70 seconds

## BFS IS instance: IS_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -7.3299055, -5.0773525, -7.2868624, -5.1847034, -2.1452022, 2.2017839
1: 1.8809752, 3.5751677, 1.9466591, 3.5619581, -1.5183196, 1.4331832
2: -4.9403496, -3.2029963, -4.9528751, -3.2875631, -1.4119327, 1.5046134
3: -11.0604973, -8.8752155, -11.0318432, -8.8928366, -1.9286456, 1.8861034
4: -5.7164073, -3.8765256, -5.5838881, -3.8599126, -1.8564947, 1.7073624
5: -9.1332655, -7.3159857, -9.0527153, -7.3236380, -1.8096275, 1.7367296
6: -6.6635323, -4.3357487, -6.5037260, -4.2974043, -2.0495658, 1.8668594
7: -8.8186531, -7.3910089, -8.8417397, -7.4164562, -1.2604175, 1.4039936
8: 0.9995713, 2.5376291, 0.9824457, 2.5185556, -1.4146268, 1.4051735
9: -9.4818926, -7.1831555, -9.4294062, -7.4215980, -1.8663521, 2.0518291

Time for backsubstitution: 5.72 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 1683
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 1928

## Relational analysis of IS_B1_A2_B1_A1_A1

### Relational analysis result of IS_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9395375, upper bound: 0.9595647
time: 3.74 seconds

## Relational analysis of IS_B1_A2_B1_A1_A2

### Relational analysis result of IS_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9395375, upper bound: 0.9624910
time: 3.73 seconds

## BFS IS instance: IS_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -7.3326211, -5.0760612, -7.2868624, -5.1847034, -2.1479177, 2.2007487
1: 1.8840065, 3.5769000, 1.9466591, 3.5619581, -1.5194678, 1.4447780
2: -4.9442549, -3.1978793, -4.9528751, -3.2875631, -1.4140623, 1.5103797
3: -11.0695801, -8.8759470, -11.0318432, -8.8928366, -1.9436622, 1.8882408
4: -5.7211289, -3.8755307, -5.5838881, -3.8599126, -1.8612163, 1.7083573
5: -9.1393318, -7.2882881, -9.0527153, -7.3236380, -1.8156939, 1.7644272
6: -6.6594758, -4.3201294, -6.5037260, -4.2974043, -2.0364356, 1.9046874
7: -8.8500376, -7.3772836, -8.8417397, -7.4164562, -1.2652521, 1.4215710
8: 0.9924660, 2.5286002, 0.9824457, 2.5185556, -1.4690514, 1.4102616
9: -9.4820032, -7.1811523, -9.4294062, -7.4215980, -1.8746414, 2.0538988

Time for backsubstitution: 5.79 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1928
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 1928

## Relational analysis of IS_B1_A2_B1_A2_A1

### Relational analysis result of IS_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9395375, upper bound: 0.9576242
time: 3.72 seconds

## Relational analysis of IS_B1_A2_B1_A2_A2

### Relational analysis result of IS_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9395375, upper bound: 0.9623081
time: 3.76 seconds

## BFS IS instance: IS_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -7.3295903, -5.0776730, -7.2903109, -5.1862478, -2.1433425, 2.2057123
1: 1.8813293, 3.5711708, 1.8706965, 3.5463653, -1.5247023, 1.5126605
2: -4.9360542, -3.2031918, -4.9204917, -3.2798326, -1.4564214, 1.4969261
3: -11.0573206, -8.8756161, -11.0039759, -8.8929081, -1.9338236, 1.8702118
4: -5.7091684, -3.8766527, -5.5557218, -3.7552865, -1.9538820, 1.6790690
5: -9.1322231, -7.3161926, -9.0555630, -7.3248386, -1.8073845, 1.7393703
6: -6.6630745, -4.3462811, -6.5047956, -4.3752627, -2.0068164, 1.8836658
7: -8.8078384, -7.3911514, -8.7486906, -7.4113126, -1.3185437, 1.3575392
8: 1.0043840, 2.5374780, 1.0089264, 2.5872927, -1.4718108, 1.4108405
9: -9.4664593, -7.1831899, -9.3513823, -7.2289605, -2.1052227, 2.0627315

Time for backsubstitution: 5.78 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1928
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 1683
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 1928

## Relational analysis of IS_B1_A2_B2_A1_A1

### Relational analysis result of IS_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9231270, upper bound: 0.9330294
time: 3.78 seconds

## Relational analysis of IS_B1_A2_B2_A1_A2

### Relational analysis result of IS_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9231270, upper bound: 0.9398825
time: 3.76 seconds

## BFS IS instance: IS_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -7.3323069, -5.0763807, -7.2903109, -5.1862478, -2.1460590, 2.2046785
1: 1.8843505, 3.5724957, 1.8706965, 3.5463653, -1.5258615, 1.5270822
2: -4.9400568, -3.1980741, -4.9204917, -3.2798326, -1.4583240, 1.5030231
3: -11.0663280, -8.8763542, -11.0039759, -8.8929081, -1.9475594, 1.8723609
4: -5.7138882, -3.8756571, -5.5557218, -3.7552865, -1.9586017, 1.6800647
5: -9.1382885, -7.2884941, -9.0555630, -7.3248386, -1.8134499, 1.7670689
6: -6.6590257, -4.3306613, -6.5047956, -4.3752627, -1.9936986, 1.9240572
7: -8.8388481, -7.3774285, -8.7486906, -7.4113126, -1.3300240, 1.3712621
8: 0.9976592, 2.5284548, 1.0089264, 2.5872927, -1.5277712, 1.4159467
9: -9.4665699, -7.1811857, -9.3513823, -7.2289605, -2.1134436, 2.0648038

Time for backsubstitution: 5.72 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1928
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 1928

## Relational analysis of IS_B1_A2_B2_A2_A1

### Relational analysis result of IS_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9231270, upper bound: 0.9312228
time: 3.78 seconds

## Relational analysis of IS_B1_A2_B2_A2_A2

### Relational analysis result of IS_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9231270, upper bound: 0.9405051
time: 3.68 seconds

## BFS IS instance: IS_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -7.2883701, -5.1846838, -7.3230686, -5.1392536, -2.1491165, 2.1383848
1: 1.9466350, 3.5629015, 1.9471262, 3.5624185, -1.4127488, 1.4333620
2: -4.9566669, -3.2875462, -4.9296737, -3.2884791, -1.4123290, 1.3946190
3: -11.0318794, -8.8928223, -11.0536289, -8.8779402, -1.8847318, 1.9366446
4: -5.5839472, -3.8463845, -5.5940585, -3.8728399, -1.7111073, 1.7476740
5: -9.0527601, -7.3194895, -9.0840378, -7.3011351, -1.7516251, 1.7645483
6: -6.5037718, -4.2940197, -6.5199671, -4.3264866, -1.8444326, 1.8932498
7: -8.8417425, -7.4146304, -8.8546562, -7.3986664, -1.4021657, 1.2656698
8: 0.9823718, 2.5220494, 0.9727139, 2.5151730, -1.3968613, 1.4475865
9: -9.4325314, -7.4215832, -9.4114399, -7.4061780, -1.8393824, 1.8228590

Time for backsubstitution: 5.72 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 655
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 655

## Relational analysis of IS_B2_A1_B1_A1_B1

### Relational analysis result of IS_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9054458, upper bound: 0.9747683
time: 4.05 seconds

## Relational analysis of IS_B2_A1_B1_A1_B2

### Relational analysis result of IS_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9031350, upper bound: 0.9747684
time: 3.99 seconds

## BFS IS instance: IS_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -7.2918048, -5.1862307, -7.3227825, -5.1395764, -2.1522284, 2.1365519
1: 1.8706703, 3.5473242, 1.9474659, 3.5580149, -1.4924424, 1.4391825
2: -4.9242573, -3.2798150, -4.9248047, -3.2886801, -1.4044552, 1.4388717
3: -11.0040150, -8.8928928, -11.0505514, -8.8783331, -1.8688211, 1.9420147
4: -5.5557842, -3.7440040, -5.5866547, -3.8729103, -1.6828740, 1.8426507
5: -9.0556183, -7.3206935, -9.0830336, -7.3013391, -1.7542791, 1.7623401
6: -6.5048437, -4.3718333, -6.5195084, -4.3370590, -1.8600426, 1.8500559
7: -8.7486925, -7.4096055, -8.8434658, -7.3988056, -1.3498869, 1.3241837
8: 1.0088367, 2.5907869, 0.9779177, 2.5150256, -1.4012327, 1.5042815
9: -9.3555136, -7.2289448, -9.3974037, -7.4062142, -1.8497527, 2.0617619

Time for backsubstitution: 5.68 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 655
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 655

## Relational analysis of IS_B2_A1_B1_A2_B1

### Relational analysis result of IS_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8842574, upper bound: 0.9612677
time: 3.30 seconds

## Relational analysis of IS_B2_A1_B1_A2_B2

### Relational analysis result of IS_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8841186, upper bound: 0.9612677
time: 3.65 seconds

## BFS IS instance: IS_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -7.2868624, -5.1847034, -7.3294983, -5.1072264, -2.1796360, 2.1447949
1: 1.9466591, 3.5619581, 1.8860457, 3.5832958, -1.4237189, 1.5032210
2: -4.9528751, -3.2875631, -4.9430962, -3.2060246, -1.5041976, 1.4128270
3: -11.0318432, -8.8928366, -11.0544147, -8.8776598, -1.8842778, 1.9405732
4: -5.5838881, -3.8599126, -5.7035041, -3.8766904, -1.7071977, 1.8435915
5: -9.0527153, -7.3236380, -9.1362724, -7.3025498, -1.7501655, 1.8126345
6: -6.5037260, -4.2974043, -6.6270022, -4.3036032, -1.8846049, 2.0163090
7: -8.8417397, -7.4164562, -8.8546991, -7.3931198, -1.4088864, 1.2705560
8: 0.9824457, 2.5185556, 0.9749146, 2.4963598, -1.4004166, 1.4480093
9: -9.4294062, -7.4215980, -9.4795465, -7.1889610, -2.0513144, 1.8689189

Time for backsubstitution: 5.70 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 655
type: A, layer: 3, pos: 655
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 655

## Relational analysis of IS_B2_A1_B2_A1_B1

### Relational analysis result of IS_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9624908, upper bound: 0.9395377
time: 3.76 seconds

## Relational analysis of IS_B2_A1_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9623082, upper bound: 0.9395376
time: 3.58 seconds

## BFS IS instance: IS_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -7.2903109, -5.1862478, -7.3291988, -5.1075444, -2.1827664, 2.1429510
1: 1.8706965, 3.5463653, 1.8863809, 3.5788858, -1.5033817, 1.5096035
2: -4.9204917, -3.2798326, -4.9388852, -3.2062228, -1.4965105, 1.4570903
3: -11.0039759, -8.8929081, -11.0513411, -8.8780556, -1.8683648, 1.9459453
4: -5.5557218, -3.7552865, -5.6956000, -3.8768115, -1.6789103, 1.9403136
5: -9.0555630, -7.3248386, -9.1352406, -7.3027592, -1.7528038, 1.8104019
6: -6.5047956, -4.3752627, -6.6265526, -4.3141460, -1.9015856, 1.9735808
7: -8.7486906, -7.4113126, -8.8435087, -7.3932657, -1.3554249, 1.3291271
8: 1.0089264, 2.5872927, 0.9801126, 2.4962139, -1.4061007, 1.5045838
9: -9.3513823, -7.2289605, -9.4641027, -7.1889958, -2.0622187, 2.1077886

Time for backsubstitution: 5.67 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 655
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 655

## Relational analysis of IS_B2_A1_B2_A2_B1

### Relational analysis result of IS_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9398824, upper bound: 0.9231269
time: 3.55 seconds

## Relational analysis of IS_B2_A1_B2_A2_B2

### Relational analysis result of IS_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9405048, upper bound: 0.9231271
time: 3.59 seconds

## BFS IS instance: IS_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -7.3342299, -5.1349053, -7.3230686, -5.1392536, -2.1302671, 2.1201537
1: 1.9427099, 3.5675063, 1.9471262, 3.5624185, -1.4744880, 1.4818428
2: -4.9561667, -3.2879825, -4.9296737, -3.2884791, -1.4110451, 1.3951087
3: -11.0532627, -8.8795404, -11.0536289, -8.8779402, -1.9689903, 1.9708343
4: -5.5809526, -3.8421807, -5.5940585, -3.8728399, -1.7081127, 1.7518778
5: -9.0798616, -7.2846737, -9.0840378, -7.3011351, -1.7787266, 1.7993641
6: -6.5299716, -4.2894926, -6.5199671, -4.3264866, -1.8661871, 1.9001851
7: -8.8528118, -7.3950086, -8.8546562, -7.3986664, -1.4004989, 1.4166179
8: 0.9866581, 2.5151229, 0.9727139, 2.5151730, -1.4750166, 1.4833744
9: -9.4480572, -7.4038544, -9.4114399, -7.4061780, -1.8643293, 1.8429127

Time for backsubstitution: 5.77 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 655
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 1683
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 655

## Relational analysis of IS_B2_A2_B1_A1_B1

### Relational analysis result of IS_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8880345, upper bound: 0.9882712
time: 3.94 seconds

## Relational analysis of IS_B2_A2_B1_A1_B2

### Relational analysis result of IS_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8838170, upper bound: 0.9882711
time: 4.46 seconds

## BFS IS instance: IS_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -7.3359895, -5.1373329, -7.3227825, -5.1395764, -2.1336660, 2.1193688
1: 1.8684154, 3.5498276, 1.9474659, 3.5580149, -1.5477188, 1.4853139
2: -4.9204550, -3.2813711, -4.9248047, -3.2886801, -1.4020293, 1.4372456
3: -11.0284929, -8.8796349, -11.0505514, -8.8783331, -1.9537163, 1.9755092
4: -5.5508881, -3.7506237, -5.5866547, -3.8729103, -1.6779778, 1.8360310
5: -9.0838490, -7.2861376, -9.0830336, -7.3013391, -1.7825098, 1.7968960
6: -6.5258422, -4.3784766, -6.5195084, -4.3370590, -1.8969426, 1.8553243
7: -8.7548981, -7.3989048, -8.8434658, -7.3988056, -1.3560925, 1.4445610
8: 1.0155959, 2.5790672, 0.9779177, 2.5150256, -1.4729879, 1.5386610
9: -9.3728790, -7.2114639, -9.3974037, -7.4062142, -1.8758428, 2.0654931

Time for backsubstitution: 5.73 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 655
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 1683
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 655

## Relational analysis of IS_B2_A2_B1_A2_B1

### Relational analysis result of IS_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8640902, upper bound: 0.9771827
time: 3.63 seconds

## Relational analysis of IS_B2_A2_B1_A2_B2

### Relational analysis result of IS_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8605408, upper bound: 0.9771830
time: 3.67 seconds

## BFS IS instance: IS_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -7.3329220, -5.1349235, -7.3294983, -5.1072264, -2.1678300, 2.1266017
1: 1.9427357, 3.5665708, 1.8860457, 3.5832958, -1.4859452, 1.5537701
2: -4.9524317, -3.2880006, -4.9430962, -3.2060246, -1.5026264, 1.4132224
3: -11.0532198, -8.8795519, -11.0544147, -8.8776598, -1.9684219, 1.9747682
4: -5.5808945, -3.8538718, -5.7035041, -3.8766904, -1.7042041, 1.8496323
5: -9.0798101, -7.2888203, -9.1362724, -7.3025498, -1.7772603, 1.8474522
6: -6.5299230, -4.2928767, -6.6270022, -4.3036032, -1.9127722, 2.0247390
7: -8.8528099, -7.3968153, -8.8546991, -7.3931198, -1.4070928, 1.4213443
8: 0.9867339, 2.5116272, 0.9749146, 2.4963598, -1.4739375, 1.4839852
9: -9.4456148, -7.4038692, -9.4795465, -7.1889610, -2.0692329, 1.8890691

Time for backsubstitution: 5.77 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 655
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 1683
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 655

## Relational analysis of IS_B2_A2_B2_A1_B1

### Relational analysis result of IS_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9474879, upper bound: 0.9535115
time: 4.09 seconds

## Relational analysis of IS_B2_A2_B2_A1_B2

### Relational analysis result of IS_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9483478, upper bound: 0.9535100
time: 3.75 seconds

## BFS IS instance: IS_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -7.3346839, -5.1373510, -7.3291988, -5.1075444, -2.1711731, 2.1257911
1: 1.8684385, 3.5488775, 1.8863809, 3.5788858, -1.5590982, 1.5556929
2: -4.9176950, -3.2813902, -4.9388852, -3.2062228, -1.4936328, 1.4551501
3: -11.0284491, -8.8796492, -11.0513411, -8.8780556, -1.9531455, 1.9794436
4: -5.5508261, -3.7622437, -5.6956000, -3.8768115, -1.6740146, 1.9333563
5: -9.0837851, -7.2903066, -9.1352406, -7.3027592, -1.7810259, 1.8449340
6: -6.5257945, -4.3816762, -6.6265526, -4.3141460, -1.9435496, 1.9798911
7: -8.7548943, -7.4006310, -8.8435087, -7.3932657, -1.3616285, 1.4428778
8: 1.0156851, 2.5755730, 0.9801126, 2.4962139, -1.4740753, 1.5396259
9: -9.3687687, -7.2114773, -9.4641027, -7.1889958, -2.0810437, 2.1116154

Time for backsubstitution: 5.71 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 655
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 1683
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 655

## Relational analysis of IS_B2_A2_B2_A2_B1

### Relational analysis result of IS_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9228046, upper bound: 0.9383472
time: 3.50 seconds

## Relational analysis of IS_B2_A2_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9232987, upper bound: 0.9383474
time: 3.59 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 13.03 seconds
IS_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 13.03
Output dim: 1, lower bound: -0.9744456, upper bound: 0.9054462
IS_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 13.03
Output dim: 1, lower bound: -0.9744456, upper bound: 0.9054475
IS_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 13.03
Output dim: 1, lower bound: -0.9744456, upper bound: 0.9031351
IS_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 13.03
Output dim: 1, lower bound: -0.9744456, upper bound: 0.9031367
IS_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 13.03
Output dim: 1, lower bound: -0.9612677, upper bound: 0.8842575
IS_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 13.03
Output dim: 1, lower bound: -0.9612677, upper bound: 0.8842575
IS_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 13.03
Output dim: 1, lower bound: -0.9612677, upper bound: 0.8841187
IS_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 13.03
Output dim: 1, lower bound: -0.9612677, upper bound: 0.8841185
IS_B1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 13.03
Output dim: 1, lower bound: -0.9395375, upper bound: 0.9595647
IS_B1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 13.03
Output dim: 1, lower bound: -0.9395375, upper bound: 0.9624910
IS_B1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 13.03
Output dim: 1, lower bound: -0.9395375, upper bound: 0.9576242
IS_B1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 13.03
Output dim: 1, lower bound: -0.9395375, upper bound: 0.9623081
IS_B1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 13.03
Output dim: 1, lower bound: -0.9231270, upper bound: 0.9330294
IS_B1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 13.03
Output dim: 1, lower bound: -0.9231270, upper bound: 0.9398825
IS_B1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 13.03
Output dim: 1, lower bound: -0.9231270, upper bound: 0.9312228
IS_B1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 13.03
Output dim: 1, lower bound: -0.9231270, upper bound: 0.9405051
IS_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 13.03
Output dim: 1, lower bound: -0.9054458, upper bound: 0.9747683
IS_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 13.03
Output dim: 1, lower bound: -0.9031350, upper bound: 0.9747684
IS_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 13.03
Output dim: 1, lower bound: -0.8842574, upper bound: 0.9612677
IS_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 13.03
Output dim: 1, lower bound: -0.8841186, upper bound: 0.9612677
IS_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 13.03
Output dim: 1, lower bound: -0.9624908, upper bound: 0.9395377
IS_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 13.03
Output dim: 1, lower bound: -0.9623082, upper bound: 0.9395376
IS_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 13.03
Output dim: 1, lower bound: -0.9398824, upper bound: 0.9231269
IS_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 13.03
Output dim: 1, lower bound: -0.9405048, upper bound: 0.9231271
IS_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 13.03
Output dim: 1, lower bound: -0.8880345, upper bound: 0.9882712
IS_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 13.03
Output dim: 1, lower bound: -0.8838170, upper bound: 0.9882711
IS_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 13.03
Output dim: 1, lower bound: -0.8640902, upper bound: 0.9771827
IS_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 13.03
Output dim: 1, lower bound: -0.8605408, upper bound: 0.9771830
IS_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 13.03
Output dim: 1, lower bound: -0.9474879, upper bound: 0.9535115
IS_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 13.03
Output dim: 1, lower bound: -0.9483478, upper bound: 0.9535100
IS_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 13.03
Output dim: 1, lower bound: -0.9228046, upper bound: 0.9383472
IS_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 13.03
Output dim: 1, lower bound: -0.9232987, upper bound: 0.9383474

## BFS IS instance: IS_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -7.3235512, -5.1092343, -7.2769489, -5.1897917, -2.1337595, 2.1494002
1: 1.9421787, 3.5543673, 1.9518673, 3.5409729, -1.4199769, 1.4175718
2: -4.9268980, -3.2843523, -4.9282980, -3.2894833, -1.3911746, 1.3886759
3: -11.0597095, -8.8754940, -11.0311432, -8.8928490, -1.9236550, 1.8862481
4: -5.6076269, -3.8724446, -5.5725021, -3.8723450, -1.6864467, 1.7000575
5: -9.0807152, -7.3146591, -9.0523262, -7.3381581, -1.7425570, 1.7376671
6: -6.5577707, -4.3584986, -6.4927859, -4.3330555, -1.8859305, 1.8221374
7: -8.8186169, -7.3968081, -8.8417253, -7.4188471, -1.2486796, 1.3964443
8: 0.9978461, 2.5559621, 0.9832053, 2.5211549, -1.4097221, 1.3983591
9: -9.4142237, -7.4004898, -9.3544703, -7.4244251, -1.8180661, 1.7678285

Time for backsubstitution: 5.77 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 1683
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 1928

## Relational analysis of IS_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9744456, upper bound: 0.8994768
time: 3.84 seconds

## Relational analysis of IS_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9744456, upper bound: 0.9054462
time: 3.83 seconds

## BFS IS instance: IS_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -7.3235512, -5.1092343, -7.2822495, -5.1590724, -2.1644788, 2.1567595
1: 1.9421787, 3.5543673, 1.8907847, 3.5610566, -1.4359674, 1.4756885
2: -4.9268980, -3.2843523, -4.9394035, -3.2089179, -1.4833839, 1.4042628
3: -11.0597095, -8.8754940, -11.0319614, -8.8925762, -1.9236388, 1.8861957
4: -5.6076269, -3.8724446, -5.6818819, -3.8854723, -1.7221546, 1.8094373
5: -9.0807152, -7.3146591, -9.1051350, -7.3377194, -1.7429957, 1.7904758
6: -6.5577707, -4.3584986, -6.5974941, -4.3099504, -1.9240537, 1.9361761
7: -8.8186169, -7.3968081, -8.8417683, -7.4136162, -1.2568121, 1.4020724
8: 0.9978461, 2.5559621, 0.9852152, 2.5023451, -1.3962159, 1.4040420
9: -9.4142237, -7.4004898, -9.4236612, -7.2071385, -2.0289237, 1.8248985

Time for backsubstitution: 5.72 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 655
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 1683
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 1928

## Relational analysis of IS_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9744456, upper bound: 0.8994782
time: 3.74 seconds

## Relational analysis of IS_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9744456, upper bound: 0.9054461
time: 4.22 seconds

## BFS IS instance: IS_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -7.3262653, -5.1080284, -7.2769489, -5.1897917, -2.1364737, 2.1483426
1: 1.9450760, 3.5560288, 1.9518673, 3.5409729, -1.4190824, 1.4303534
2: -4.9310536, -3.2795374, -4.9282980, -3.2894833, -1.3931148, 1.3938372
3: -11.0687599, -8.8762255, -11.0311432, -8.8928490, -1.9387751, 1.8883841
4: -5.6125240, -3.8706160, -5.5725021, -3.8723450, -1.6854734, 1.7018862
5: -9.0870552, -7.2889376, -9.0523262, -7.3381581, -1.7488971, 1.7633886
6: -6.5535812, -4.3428826, -6.4927859, -4.3330555, -1.8717961, 1.8627229
7: -8.8499956, -7.3828640, -8.8417253, -7.4188471, -1.2528448, 1.4145477
8: 0.9901543, 2.5474091, 0.9832053, 2.5211549, -1.4639163, 1.4029465
9: -9.4143372, -7.3984575, -9.3544703, -7.4244251, -1.8263583, 1.7700791

Time for backsubstitution: 5.75 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1928
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 655
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 1928

## Relational analysis of IS_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9744456, upper bound: 0.8940714
time: 3.67 seconds

## Relational analysis of IS_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9744456, upper bound: 0.9031353
time: 3.74 seconds

## BFS IS instance: IS_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -7.3262653, -5.1080284, -7.2822495, -5.1590724, -2.1671929, 2.1557014
1: 1.9450760, 3.5560288, 1.8907847, 3.5610566, -1.4350724, 1.4884799
2: -4.9310536, -3.2795374, -4.9394035, -3.2089179, -1.4853246, 1.4106505
3: -11.0687599, -8.8762255, -11.0319614, -8.8925762, -1.9387722, 1.8883317
4: -5.6125240, -3.8706160, -5.6818819, -3.8854723, -1.7270517, 1.8112659
5: -9.0870552, -7.2889376, -9.1051350, -7.3377194, -1.7493358, 1.8161974
6: -6.5535812, -4.3428826, -6.5974941, -4.3099504, -1.9099193, 1.9763319
7: -8.8499956, -7.3828640, -8.8417683, -7.4136162, -1.2626059, 1.4201760
8: 0.9901543, 2.5474091, 0.9852152, 2.5023451, -1.4511111, 1.4086292
9: -9.4143372, -7.3984575, -9.4236612, -7.2071385, -2.0367277, 1.8271492

Time for backsubstitution: 5.80 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1928
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 1928

## Relational analysis of IS_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9744456, upper bound: 0.8940728
time: 3.89 seconds

## Relational analysis of IS_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9744456, upper bound: 0.9031367
time: 3.80 seconds

## BFS IS instance: IS_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -7.3232536, -5.1095614, -7.2804403, -5.1913214, -2.1319323, 2.1533804
1: 1.9425373, 3.5503826, 1.8761187, 3.5254929, -1.4257531, 1.4967306
2: -4.9222851, -3.2845514, -4.8956699, -3.2818928, -1.4354539, 1.3799214
3: -11.0565290, -8.8758936, -11.0032778, -8.8929253, -1.9288344, 1.8703570
4: -5.6002426, -3.8725188, -5.5437984, -3.7734871, -1.7824543, 1.6712797
5: -9.0796633, -7.3148575, -9.0550995, -7.3392749, -1.7403884, 1.7402420
6: -6.5573053, -4.3690581, -6.4931102, -4.4106774, -1.8429017, 1.8363636
7: -8.8078022, -7.3969421, -8.7486773, -7.4140849, -1.3067107, 1.3517351
8: 1.0026622, 2.5558147, 1.0098567, 2.5898929, -1.4670992, 1.4027390
9: -9.4002028, -7.4005260, -9.2789249, -7.2317133, -2.0572529, 1.7777219

Time for backsubstitution: 5.73 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 655
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 1683
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 1928

## Relational analysis of IS_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9612677, upper bound: 0.8742171
time: 3.66 seconds

## Relational analysis of IS_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9612677, upper bound: 0.8842575
time: 3.74 seconds

## BFS IS instance: IS_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -7.3232536, -5.1095614, -7.2856598, -5.1607676, -2.1624861, 2.1607671
1: 1.9425373, 3.5503826, 1.8150520, 3.5451560, -1.4419527, 1.5554607
2: -4.9222851, -3.2845514, -4.9098349, -3.2009244, -1.5257964, 1.3966057
3: -11.0565290, -8.8758936, -11.0040970, -8.8926573, -1.9288044, 1.8703060
4: -5.6002426, -3.8725188, -5.6536593, -3.7901065, -1.8101361, 1.7811406
5: -9.0796633, -7.3148575, -9.1081238, -7.3389163, -1.7407470, 1.7932663
6: -6.5573053, -4.3690581, -6.5974555, -4.3883834, -1.8831253, 1.9549942
7: -8.8078022, -7.3969421, -8.7487202, -7.4086123, -1.3160024, 1.3517780
8: 1.0026622, 2.5558147, 1.0123968, 2.5710812, -1.4536552, 1.4088545
9: -9.4002028, -7.4005260, -9.3372498, -7.0187893, -2.2664771, 1.8369594

Time for backsubstitution: 5.75 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 655
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 1683
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 1928

## Relational analysis of IS_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9612677, upper bound: 0.8742171
time: 3.62 seconds

## Relational analysis of IS_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9612677, upper bound: 0.8842575
time: 3.50 seconds

## BFS IS instance: IS_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -7.3259716, -5.1083546, -7.2804403, -5.1913214, -2.1346502, 2.1523228
1: 1.9454250, 3.5516279, 1.8761187, 3.5254929, -1.4248590, 1.5126240
2: -4.9262028, -3.2797363, -4.8956699, -3.2818928, -1.4371753, 1.3853195
3: -11.0655041, -8.8766336, -11.0032778, -8.8929253, -1.9426737, 1.8725038
4: -5.6051173, -3.8706903, -5.5437984, -3.7734871, -1.7783937, 1.6731081
5: -9.0860386, -7.2891359, -9.0550995, -7.3392749, -1.7467637, 1.7659636
6: -6.5531225, -4.3534412, -6.4931102, -4.4106774, -1.8287659, 1.8792250
7: -8.8388042, -7.3830013, -8.7486773, -7.4140849, -1.3178389, 1.3656759
8: 0.9953523, 2.5472646, 1.0098567, 2.5898929, -1.5224097, 1.4073451
9: -9.4003162, -7.3984938, -9.2789249, -7.2317133, -2.0654757, 1.7799730

Time for backsubstitution: 5.79 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1928
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 655
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 1928

## Relational analysis of IS_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9612677, upper bound: 0.8700824
time: 3.65 seconds

## Relational analysis of IS_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9612677, upper bound: 0.8841187
time: 3.55 seconds

## BFS IS instance: IS_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -7.3259716, -5.1083546, -7.2856598, -5.1607676, -2.1652040, 2.1597095
1: 1.9454250, 3.5516279, 1.8150520, 3.5451560, -1.4410591, 1.5712397
2: -4.9262028, -3.2797363, -4.9098349, -3.2009244, -1.5275173, 1.4030337
3: -11.0655041, -8.8766336, -11.0040970, -8.8926573, -1.9426560, 1.8724527
4: -5.6051173, -3.8706903, -5.6536593, -3.7901065, -1.8150108, 1.7829690
5: -9.0860386, -7.2891359, -9.1081238, -7.3389163, -1.7471223, 1.8189878
6: -6.5531225, -4.3534412, -6.5974555, -4.3883834, -1.8689895, 1.9971704
7: -8.8388042, -7.3830013, -8.7487202, -7.4086123, -1.3282454, 1.3657188
8: 0.9953523, 2.5472646, 1.0123968, 2.5710812, -1.5101275, 1.4134603
9: -9.4003162, -7.3984938, -9.3372498, -7.0187893, -2.2737460, 1.8392105

Time for backsubstitution: 5.76 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1928
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 1928

## Relational analysis of IS_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9612677, upper bound: 0.8700824
time: 3.73 seconds

## Relational analysis of IS_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9612677, upper bound: 0.8841185
time: 3.62 seconds

## BFS IS instance: IS_B1_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -7.2771063, -5.1603718, -7.2868624, -5.1847034, -2.0849557, 2.1264906
1: 1.8894880, 3.5658026, 1.9466591, 3.5619581, -1.4906034, 1.4160941
2: -4.9368787, -3.2143183, -4.9528751, -3.2875631, -1.4095664, 1.4919289
3: -11.0119371, -8.8926182, -11.0318432, -8.8928366, -1.8120022, 1.8236043
4: -5.6945982, -3.8868690, -5.5838881, -3.8599126, -1.8346856, 1.6812749
5: -9.0962486, -7.3761148, -9.0527153, -7.3236380, -1.7726107, 1.6766005
6: -6.6011786, -4.3389587, -6.5037260, -4.2974043, -2.0000062, 1.8564420
7: -8.7990932, -7.4329472, -8.8417397, -7.4164562, -1.2360950, 1.2387643
8: 1.0057302, 2.5071068, 0.9824457, 2.5185556, -1.3415215, 1.3682532
9: -9.4628487, -7.2098322, -9.4294062, -7.4215980, -1.8338037, 2.0190065

Time for backsubstitution: 5.72 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 1935

## Relational analysis of IS_B1_A2_B1_A1_A1_A1

### Relational analysis result of IS_B1_A2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8530358, upper bound: 0.9057481
time: 4.03 seconds

## Relational analysis of IS_B1_A2_B1_A1_A1_A2

### Relational analysis result of IS_B1_A2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7934066, upper bound: 0.8158663
time: 4.03 seconds

## BFS IS instance: IS_B1_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -7.3238783, -5.1086454, -7.2868624, -5.1847034, -2.1391749, 2.1782169
1: 1.8866072, 3.5694194, 1.9466591, 3.5619581, -1.5022264, 1.4143255
2: -4.9366636, -3.2128606, -4.9528751, -3.2875631, -1.4092135, 1.4959648
3: -11.0350170, -8.8797626, -11.0318432, -8.8928366, -1.9215240, 1.8806858
4: -5.6920261, -3.8782740, -5.5838881, -3.8599126, -1.8321135, 1.7056141
5: -9.1296043, -7.3412037, -9.0527153, -7.3236380, -1.8059664, 1.7115116
6: -6.6246901, -4.3378649, -6.5037260, -4.2974043, -2.0265260, 1.8648448
7: -8.8159819, -7.4049392, -8.8417397, -7.4164562, -1.2385535, 1.3962700
8: 1.0031295, 2.5102730, 0.9824457, 2.5185556, -1.4114020, 1.3945270
9: -9.4781694, -7.1924286, -9.4294062, -7.4215980, -1.8637910, 2.0469117

Time for backsubstitution: 5.74 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 655
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 1683
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 1935

## Relational analysis of IS_B1_A2_B1_A1_A2_A1

### Relational analysis result of IS_B1_A2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8530358, upper bound: 0.9065118
time: 4.08 seconds

## Relational analysis of IS_B1_A2_B1_A1_A2_A2

### Relational analysis result of IS_B1_A2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7934066, upper bound: 0.8158662
time: 3.56 seconds

## BFS IS instance: IS_B1_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -7.2795219, -5.1586504, -7.2868624, -5.1847034, -2.0948186, 2.1282120
1: 1.8938050, 3.5667272, 1.9466591, 3.5619581, -1.4858866, 1.4276879
2: -4.9398794, -3.2094123, -4.9528751, -3.2875631, -1.4117100, 1.4979587
3: -11.0225153, -8.8937149, -11.0318432, -8.8928366, -1.8213496, 1.8216891
4: -5.6992822, -3.8853343, -5.5838881, -3.8599126, -1.8393695, 1.6834898
5: -9.1085863, -7.3483725, -9.0527153, -7.3236380, -1.7849483, 1.7043428
6: -6.5976477, -4.3255382, -6.5037260, -4.2974043, -1.9943247, 1.8936405
7: -8.8383636, -7.4144926, -8.8417397, -7.4164562, -1.2429724, 1.2604411
8: 0.9913850, 2.5030365, 0.9824457, 2.5185556, -1.3880525, 1.3763056
9: -9.4624414, -7.2084641, -9.4294062, -7.4215980, -1.8408971, 2.0212927

Time for backsubstitution: 5.76 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 1935

## Relational analysis of IS_B1_A2_B1_A2_A1_B1

### Relational analysis result of IS_B1_A2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8562878, upper bound: 0.8567411
time: 3.40 seconds

## Relational analysis of IS_B1_A2_B1_A2_A1_B2

### Relational analysis result of IS_B1_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7934066, upper bound: 0.8169553
time: 3.93 seconds

## BFS IS instance: IS_B1_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -7.3270097, -5.1073532, -7.2868624, -5.1847034, -2.1423063, 2.1795092
1: 1.8894062, 3.5719640, 1.9466591, 3.5619581, -1.5005288, 1.4259212
2: -4.9406886, -3.2075667, -4.9528751, -3.2875631, -1.4114070, 1.5013182
3: -11.0430441, -8.8800011, -11.0318432, -8.8928366, -1.9364738, 1.8829513
4: -5.6967745, -3.8772445, -5.5838881, -3.8599126, -1.8368618, 1.7066436
5: -9.1356792, -7.3135424, -9.0527153, -7.3236380, -1.8120413, 1.7391729
6: -6.6259532, -4.3221083, -6.5037260, -4.2974043, -2.0136032, 1.9028635
7: -8.8472643, -7.3941998, -8.8417397, -7.4164562, -1.2432804, 1.4083652
8: 0.9958587, 2.4961128, 0.9824457, 2.5185556, -1.4660969, 1.3961568
9: -9.4787025, -7.1904650, -9.4294062, -7.4215980, -1.8724670, 2.0488925

Time for backsubstitution: 5.72 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 1935

## Relational analysis of IS_B1_A2_B1_A2_A2_B1

### Relational analysis result of IS_B1_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8562878, upper bound: 0.8567406
time: 3.50 seconds

## Relational analysis of IS_B1_A2_B1_A2_A2_B2

### Relational analysis result of IS_B1_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7934066, upper bound: 0.8169553
time: 3.51 seconds

## BFS IS instance: IS_B1_A2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -7.2768764, -5.1606188, -7.2903109, -5.1862478, -2.0838399, 2.1296921
1: 1.8897932, 3.5619054, 1.8706965, 3.5463653, -1.4988594, 1.4956565
2: -4.9326611, -3.2145147, -4.9204917, -3.2798326, -1.4539652, 1.4842505
3: -11.0088272, -8.8929558, -11.0039759, -8.8929081, -1.8186641, 1.8077765
4: -5.6876144, -3.8869233, -5.5557218, -3.7552865, -1.9323280, 1.6687984
5: -9.0951080, -7.3762627, -9.0555630, -7.3248386, -1.7702694, 1.6793003
6: -6.6008058, -4.3484688, -6.5047956, -4.3752627, -1.9542096, 1.8725502
7: -8.7883797, -7.4330335, -8.7486906, -7.4113126, -1.2952826, 1.2064042
8: 1.0099330, 2.5069652, 1.0089264, 2.5872927, -1.4136496, 1.3739388
9: -9.4475269, -7.2098680, -9.3513823, -7.2289605, -2.0754294, 2.0299103

Time for backsubstitution: 5.75 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 1935

## Relational analysis of IS_B1_A2_B2_A1_A1_B1

### Relational analysis result of IS_B1_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8300407, upper bound: 0.8235950
time: 3.45 seconds

## Relational analysis of IS_B1_A2_B2_A1_A1_B2

### Relational analysis result of IS_B1_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7494670, upper bound: 0.7782256
time: 3.16 seconds

## BFS IS instance: IS_B1_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -7.3235788, -5.1089621, -7.2903109, -5.1862478, -2.1373310, 2.1813488
1: 1.8869526, 3.5654130, 1.8706965, 3.5463653, -1.5086048, 1.4938035
2: -4.9323349, -3.2130599, -4.9204917, -3.2798326, -1.4536939, 1.4882754
3: -11.0321484, -8.8801498, -11.0039759, -8.8929081, -1.9268956, 1.8647606
4: -5.6840653, -3.8783946, -5.5557218, -3.7552865, -1.9287789, 1.6773272
5: -9.1285744, -7.3414121, -9.0555630, -7.3248386, -1.8037357, 1.7141509
6: -6.6242204, -4.3484054, -6.5047956, -4.3752627, -1.9837990, 1.8816421
7: -8.8051662, -7.4050827, -8.7486906, -7.4113126, -1.2969162, 1.3436079
8: 1.0079465, 2.5101247, 1.0089264, 2.5872927, -1.4685807, 1.4002023
9: -9.4627237, -7.1924620, -9.3513823, -7.2289605, -2.1026597, 2.0578153

Time for backsubstitution: 5.73 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 1683
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 1935

## Relational analysis of IS_B1_A2_B2_A1_A2_A1

### Relational analysis result of IS_B1_A2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8127968, upper bound: 0.8750838
time: 3.25 seconds

## Relational analysis of IS_B1_A2_B2_A1_A2_A2

### Relational analysis result of IS_B1_A2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7494670, upper bound: 0.7782255
time: 3.24 seconds

## BFS IS instance: IS_B1_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -7.2792935, -5.1588993, -7.2903109, -5.1862478, -2.0930457, 2.1314116
1: 1.8940930, 3.5624146, 1.8706965, 3.5463653, -1.4941530, 1.5100768
2: -4.9353504, -3.2096086, -4.9204917, -3.2798326, -1.4561086, 1.4906144
3: -11.0193272, -8.8940620, -11.0039759, -8.8929081, -1.8279860, 1.8058658
4: -5.6920609, -3.8853910, -5.5557218, -3.7552865, -1.9367745, 1.6703308
5: -9.1074286, -7.3485203, -9.0555630, -7.3248386, -1.7825899, 1.7070427
6: -6.5972781, -4.3349371, -6.5047956, -4.3752627, -1.9485309, 1.9123690
7: -8.8276272, -7.4145799, -8.7486906, -7.4113126, -1.3085701, 1.2280648
8: 0.9962602, 2.5029044, 1.0089264, 2.5872927, -1.4566417, 1.3819933
9: -9.4471207, -7.2084980, -9.3513823, -7.2289605, -2.0819685, 2.0321968

Time for backsubstitution: 5.74 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 1935

## Relational analysis of IS_B1_A2_B2_A2_A1_B1

### Relational analysis result of IS_B1_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8300407, upper bound: 0.8257753
time: 3.35 seconds

## Relational analysis of IS_B1_A2_B2_A2_A1_B2

### Relational analysis result of IS_B1_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7494670, upper bound: 0.7815084
time: 3.22 seconds

## BFS IS instance: IS_B1_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -7.3267088, -5.1076736, -7.2903109, -5.1862478, -2.1404610, 2.1826372
1: 1.8897409, 3.5675426, 1.8706965, 3.5463653, -1.5069110, 1.5082257
2: -4.9364781, -3.2077653, -4.9204917, -3.2798326, -1.4556704, 1.4939638
3: -11.0399714, -8.8803959, -11.0039759, -8.8929081, -1.9405637, 1.8670387
4: -5.6888371, -3.8773637, -5.5557218, -3.7552865, -1.9335506, 1.6783581
5: -9.1346483, -7.3137503, -9.0555630, -7.3248386, -1.8098097, 1.7418127
6: -6.6255054, -4.3326507, -6.5047956, -4.3752627, -1.9708743, 1.9222233
7: -8.8360739, -7.3943453, -8.7486906, -7.4113126, -1.3082826, 1.3543453
8: 1.0010557, 2.4959664, 1.0089264, 2.5872927, -1.5248392, 1.4018407
9: -9.4632568, -7.1904984, -9.3513823, -7.2289605, -2.1112678, 2.0597966

Time for backsubstitution: 5.76 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 1935

## Relational analysis of IS_B1_A2_B2_A2_A2_B1

### Relational analysis result of IS_B1_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8300407, upper bound: 0.8257754
time: 3.33 seconds

## Relational analysis of IS_B1_A2_B2_A2_A2_B2

### Relational analysis result of IS_B1_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7494670, upper bound: 0.7815083
time: 3.16 seconds

## BFS IS instance: IS_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -7.2883701, -5.1846838, -7.3174481, -5.1406412, -2.1477289, 2.1327643
1: 1.9466350, 3.5629015, 1.9478123, 3.5487800, -1.4043298, 1.4345345
2: -4.9566669, -3.2875462, -4.9230566, -3.2950504, -1.4039555, 1.3913875
3: -11.0318794, -8.8928223, -11.0342321, -8.8800392, -1.8811402, 1.9175987
4: -5.5839472, -3.8463845, -5.5826550, -3.8751731, -1.7087741, 1.7356918
5: -9.0527601, -7.3194895, -9.0770321, -7.3389201, -1.7138400, 1.7575426
6: -6.5037718, -4.2940197, -6.5184116, -4.3607473, -1.8261433, 1.9058969
7: -8.8417425, -7.4146304, -8.8159456, -7.4106708, -1.3887715, 1.2334914
8: 0.9823718, 2.5220494, 1.0015287, 2.5288486, -1.3914714, 1.4103827
9: -9.4325314, -7.4215832, -9.4099588, -7.4096508, -1.8350043, 1.8177292

Time for backsubstitution: 5.77 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 655
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 1725

## Relational analysis of IS_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9054458, upper bound: 0.9744456
time: 3.76 seconds

## Relational analysis of IS_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9054458, upper bound: 0.9747683
time: 3.71 seconds

## BFS IS instance: IS_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -7.2883701, -5.1846838, -7.3205829, -5.1393690, -2.1490011, 2.1358991
1: 1.9466350, 3.5629015, 1.9504740, 3.5511718, -1.4159698, 1.4306564
2: -4.9566669, -3.2875462, -4.9272966, -3.2900596, -1.4092340, 1.3933918
3: -11.0318794, -8.8928223, -11.0422583, -8.8802795, -1.8834052, 1.9326224
4: -5.5839472, -3.8463845, -5.5875826, -3.8733571, -1.7105901, 1.7351525
5: -9.0527601, -7.3194895, -9.0834208, -7.3132272, -1.7395329, 1.7639313
6: -6.5037718, -4.2940197, -6.5189385, -4.3449941, -1.8647394, 1.8906956
7: -8.8417425, -7.4146304, -8.8472214, -7.3997459, -1.4016583, 1.2380085
8: 0.9823718, 2.5220494, 0.9936895, 2.5149250, -1.3926022, 1.4651945
9: -9.4325314, -7.4215832, -9.4105520, -7.4076591, -1.8371663, 1.8264070

Time for backsubstitution: 5.80 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 655
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 1683
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 1725

## Relational analysis of IS_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9031350, upper bound: 0.9744460
time: 3.67 seconds

## Relational analysis of IS_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9031350, upper bound: 0.9747684
time: 3.77 seconds

## BFS IS instance: IS_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -7.2918048, -5.1862307, -7.3171606, -5.1409645, -2.1508403, 2.1309299
1: 1.8706703, 3.5473242, 1.9481609, 3.5447855, -1.4838657, 1.4403641
2: -4.9242573, -3.2798150, -4.9184065, -3.2952547, -1.3960783, 1.4358596
3: -11.0040150, -8.8928928, -11.0313597, -8.8804293, -1.8652172, 1.9229698
4: -5.5557842, -3.7440040, -5.5752535, -3.8752432, -1.6805410, 1.8312495
5: -9.0556183, -7.3206935, -9.0759878, -7.3391242, -1.7164941, 1.7552943
6: -6.5048437, -4.3718333, -6.5179276, -4.3713183, -1.8417020, 1.8627112
7: -8.7486925, -7.4096055, -8.8051300, -7.4108057, -1.3378868, 1.2917976
8: 1.0088367, 2.5907869, 1.0063457, 2.5287046, -1.3958330, 1.4678013
9: -9.3555136, -7.2289448, -9.3959236, -7.4096866, -1.8453732, 2.0566313

Time for backsubstitution: 5.80 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 655
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 1725

## Relational analysis of IS_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8842574, upper bound: 0.9612677
time: 3.39 seconds

## Relational analysis of IS_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8842574, upper bound: 0.9612677
time: 3.26 seconds

## BFS IS instance: IS_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -7.2918048, -5.1862307, -7.3202953, -5.1396923, -2.1521125, 2.1340647
1: 1.8706703, 3.5473242, 1.9508147, 3.5467598, -1.4983416, 1.4364777
2: -4.9242573, -3.2798150, -4.9224286, -3.2902627, -1.4016829, 1.4376450
3: -11.0040150, -8.8928928, -11.0391827, -8.8806753, -1.8674936, 1.9367132
4: -5.5557842, -3.7440040, -5.5801616, -3.8734264, -1.6823578, 1.8300359
5: -9.0556183, -7.3206935, -9.0824203, -7.3134298, -1.7421885, 1.7617269
6: -6.5048437, -4.3718333, -6.5184808, -4.3555636, -1.8829117, 1.8475022
7: -8.7486925, -7.4096055, -8.8360310, -7.3998842, -1.3488083, 1.3030374
8: 1.0088367, 2.5907869, 0.9988918, 2.5147781, -1.3969729, 1.5239208
9: -9.3555136, -7.2289448, -9.3965168, -7.4076958, -1.8475366, 2.0652411

Time for backsubstitution: 5.80 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 1683
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 1725

## Relational analysis of IS_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8841186, upper bound: 0.9612678
time: 3.66 seconds

## Relational analysis of IS_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8841186, upper bound: 0.9612677
time: 3.69 seconds

## BFS IS instance: IS_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -7.2868624, -5.1847034, -7.3238783, -5.1086454, -2.1782169, 2.1391749
1: 1.9466591, 3.5619581, 1.8866072, 3.5694194, -1.4143260, 1.5022264
2: -4.9528751, -3.2875631, -4.9366636, -3.2128606, -1.4959648, 1.4092138
3: -11.0318432, -8.8928366, -11.0350170, -8.8797626, -1.8806858, 1.9215240
4: -5.5838881, -3.8599126, -5.6920261, -3.8782740, -1.7056141, 1.8321135
5: -9.0527153, -7.3236380, -9.1296043, -7.3412037, -1.7115116, 1.8059664
6: -6.5037260, -4.2974043, -6.6246901, -4.3378649, -1.8648448, 2.0265260
7: -8.8417397, -7.4164562, -8.8159819, -7.4049392, -1.3962703, 1.2385533
8: 0.9824457, 2.5185556, 1.0031295, 2.5102730, -1.3945270, 1.4114020
9: -9.4294062, -7.4215980, -9.4781694, -7.1924286, -2.0469117, 1.8637910

Time for backsubstitution: 5.84 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 655
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 1935

## Relational analysis of IS_B2_A1_B2_A1_B1_B1

### Relational analysis result of IS_B2_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.9065114, upper bound: 0.8530360
time: 3.89 seconds

## Relational analysis of IS_B2_A1_B2_A1_B1_B2

### Relational analysis result of IS_B2_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8158663, upper bound: 0.7934066
time: 3.52 seconds

## BFS IS instance: IS_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -7.2868624, -5.1847034, -7.3270097, -5.1073532, -2.1795092, 2.1423063
1: 1.9466591, 3.5619581, 1.8894062, 3.5719640, -1.4259214, 1.5005283
2: -4.9528751, -3.2875631, -4.9406886, -3.2075667, -1.5013180, 1.4114070
3: -11.0318432, -8.8928366, -11.0430441, -8.8800011, -1.8829513, 1.9364738
4: -5.5838881, -3.8599126, -5.6967745, -3.8772445, -1.7066436, 1.8368618
5: -9.0527153, -7.3236380, -9.1356792, -7.3135424, -1.7391729, 1.8120413
6: -6.5037260, -4.2974043, -6.6259532, -4.3221083, -1.9028635, 2.0136032
7: -8.8417397, -7.4164562, -8.8472643, -7.3941998, -1.4083655, 1.2432804
8: 0.9824457, 2.5185556, 0.9958587, 2.4961128, -1.3961570, 1.4660969
9: -9.4294062, -7.4215980, -9.4787025, -7.1904650, -2.0488925, 1.8724673

Time for backsubstitution: 5.92 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 1683
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 1935

## Relational analysis of IS_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8567408, upper bound: 0.8562876
time: 3.63 seconds

## Relational analysis of IS_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8169538, upper bound: 0.7934065
time: 3.48 seconds

## BFS IS instance: IS_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -7.2903109, -5.1862478, -7.3235788, -5.1089621, -2.1813488, 2.1373310
1: 1.8706965, 3.5463653, 1.8869526, 3.5654130, -1.4938035, 1.5086045
2: -4.9204917, -3.2798326, -4.9323349, -3.2130599, -1.4882755, 1.4536939
3: -11.0039759, -8.8929081, -11.0321484, -8.8801498, -1.8647609, 1.9268956
4: -5.5557218, -3.7552865, -5.6840653, -3.8783946, -1.6773272, 1.9287789
5: -9.0555630, -7.3248386, -9.1285744, -7.3414121, -1.7141509, 1.8037357
6: -6.5047956, -4.3752627, -6.6242204, -4.3484054, -1.8816423, 1.9837990
7: -8.7486906, -7.4113126, -8.8051662, -7.4050827, -1.3436079, 1.2969162
8: 1.0089264, 2.5872927, 1.0079465, 2.5101247, -1.4002023, 1.4685810
9: -9.3513823, -7.2289605, -9.4627237, -7.1924620, -2.0578151, 2.1026597

Time for backsubstitution: 5.80 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 1935

## Relational analysis of IS_B2_A1_B2_A2_B1_B1

### Relational analysis result of IS_B2_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8750837, upper bound: 0.8127971
time: 3.98 seconds

## Relational analysis of IS_B2_A1_B2_A2_B1_B2

### Relational analysis result of IS_B2_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7782257, upper bound: 0.7494669
time: 3.24 seconds

## BFS IS instance: IS_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -7.2903109, -5.1862478, -7.3267088, -5.1076736, -2.1826372, 2.1404610
1: 1.8706965, 3.5463653, 1.8897409, 3.5675426, -1.5082262, 1.5069110
2: -4.9204917, -3.2798326, -4.9364781, -3.2077653, -1.4939640, 1.4556704
3: -11.0039759, -8.8929081, -11.0399714, -8.8803959, -1.8670387, 1.9405642
4: -5.5557218, -3.7552865, -5.6888371, -3.8773637, -1.6783581, 1.9335506
5: -9.0555630, -7.3248386, -9.1346483, -7.3137503, -1.7418127, 1.8098097
6: -6.5047956, -4.3752627, -6.6255054, -4.3326507, -1.9222231, 1.9708743
7: -8.7486906, -7.4113126, -8.8360739, -7.3943453, -1.3543453, 1.3082826
8: 1.0089264, 2.5872927, 1.0010557, 2.4959664, -1.4018404, 1.5248394
9: -9.3513823, -7.2289605, -9.4632568, -7.1904984, -2.0597968, 2.1112676

Time for backsubstitution: 5.84 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 1683
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 1935

## Relational analysis of IS_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8257753, upper bound: 0.8300405
time: 3.30 seconds

## Relational analysis of IS_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7815084, upper bound: 0.7494669
time: 3.26 seconds

## BFS IS instance: IS_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -7.3342299, -5.1349053, -7.3174481, -5.1406412, -2.1200795, 2.1067240
1: 1.9427099, 3.5675063, 1.9478123, 3.5487800, -1.4666319, 1.4882615
2: -4.9561667, -3.2879825, -4.9230566, -3.2950504, -1.4026439, 1.3921978
3: -11.0532627, -8.8795404, -11.0342321, -8.8800392, -1.9705358, 1.9520597
4: -5.5809526, -3.8421807, -5.5826550, -3.8751731, -1.7057796, 1.7404742
5: -9.0798616, -7.2846737, -9.0770321, -7.3389201, -1.7409415, 1.7923584
6: -6.5299716, -4.2894926, -6.5184116, -4.3607473, -1.8489835, 1.9125125
7: -8.8528118, -7.3950086, -8.8159456, -7.4106708, -1.3808601, 1.3757989
8: 0.9866581, 2.5151229, 1.0015287, 2.5288486, -1.4650197, 1.4371362
9: -9.4480572, -7.4038544, -9.4099588, -7.4096508, -1.8598206, 1.8377826

Time for backsubstitution: 5.87 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 655
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 1683
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 1725

## Relational analysis of IS_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8880345, upper bound: 0.9880249
time: 4.09 seconds

## Relational analysis of IS_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8880345, upper bound: 0.9882713
time: 4.15 seconds

## BFS IS instance: IS_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -7.3342299, -5.1349053, -7.3205829, -5.1393690, -2.1204224, 2.1293390
1: 1.9427099, 3.5675063, 1.9504740, 3.5511718, -1.4842846, 1.4792049
2: -4.9561667, -3.2879825, -4.9272966, -3.2900596, -1.4077506, 1.3938824
3: -11.0532627, -8.8795404, -11.0422583, -8.8802795, -1.9671841, 1.9645300
4: -5.5809526, -3.8421807, -5.5875826, -3.8733571, -1.7075956, 1.7454019
5: -9.0798616, -7.2846737, -9.0834208, -7.3132272, -1.7666345, 1.7987471
6: -6.5299716, -4.2894926, -6.5189385, -4.3449941, -1.8875675, 1.8976307
7: -8.8528118, -7.3950086, -8.8472214, -7.3997459, -1.4000168, 1.3815987
8: 0.9866581, 2.5151229, 0.9936895, 2.5149250, -1.4702563, 1.4692440
9: -9.4480572, -7.4038544, -9.4105520, -7.4076591, -1.8619945, 1.8464608

Time for backsubstitution: 5.88 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 1683
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 1725

## Relational analysis of IS_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8838170, upper bound: 0.9880247
time: 4.05 seconds

## Relational analysis of IS_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8838170, upper bound: 0.9882713
time: 4.09 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 14.24 seconds
IS_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 14.24
Output dim: 1, lower bound: -0.9744456, upper bound: 0.8994768
IS_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 14.24
Output dim: 1, lower bound: -0.9744456, upper bound: 0.9054462
IS_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 14.24
Output dim: 1, lower bound: -0.9744456, upper bound: 0.8994782
IS_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 14.24
Output dim: 1, lower bound: -0.9744456, upper bound: 0.9054461
IS_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 14.24
Output dim: 1, lower bound: -0.9744456, upper bound: 0.8940714
IS_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 14.24
Output dim: 1, lower bound: -0.9744456, upper bound: 0.9031353
IS_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 14.24
Output dim: 1, lower bound: -0.9744456, upper bound: 0.8940728
IS_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 14.24
Output dim: 1, lower bound: -0.9744456, upper bound: 0.9031367
IS_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 14.24
Output dim: 1, lower bound: -0.9612677, upper bound: 0.8742171
IS_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 14.24
Output dim: 1, lower bound: -0.9612677, upper bound: 0.8842575
IS_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 14.24
Output dim: 1, lower bound: -0.9612677, upper bound: 0.8742171
IS_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 14.24
Output dim: 1, lower bound: -0.9612677, upper bound: 0.8842575
IS_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 14.24
Output dim: 1, lower bound: -0.9612677, upper bound: 0.8700824
IS_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 14.24
Output dim: 1, lower bound: -0.9612677, upper bound: 0.8841187
IS_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 14.24
Output dim: 1, lower bound: -0.9612677, upper bound: 0.8700824
IS_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 14.24
Output dim: 1, lower bound: -0.9612677, upper bound: 0.8841185
IS_B1_A2_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 14.24
Output dim: 1, lower bound: -0.8530358, upper bound: 0.9057481
IS_B1_A2_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 14.24
Output dim: 1, lower bound: -0.7934066, upper bound: 0.8158663
IS_B1_A2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 14.24
Output dim: 1, lower bound: -0.8530358, upper bound: 0.9065118
IS_B1_A2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 14.24
Output dim: 1, lower bound: -0.7934066, upper bound: 0.8158662
IS_B1_A2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 14.24
Output dim: 1, lower bound: -0.8562878, upper bound: 0.8567411
IS_B1_A2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 14.24
Output dim: 1, lower bound: -0.7934066, upper bound: 0.8169553
IS_B1_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 14.24
Output dim: 1, lower bound: -0.8562878, upper bound: 0.8567406
IS_B1_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 14.24
Output dim: 1, lower bound: -0.7934066, upper bound: 0.8169553
IS_B1_A2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 14.24
Output dim: 1, lower bound: -0.8300407, upper bound: 0.8235950
IS_B1_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 14.24
Output dim: 1, lower bound: -0.7494670, upper bound: 0.7782256
IS_B1_A2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 14.24
Output dim: 1, lower bound: -0.8127968, upper bound: 0.8750838
IS_B1_A2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 14.24
Output dim: 1, lower bound: -0.7494670, upper bound: 0.7782255
IS_B1_A2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 14.24
Output dim: 1, lower bound: -0.8300407, upper bound: 0.8257753
IS_B1_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 14.24
Output dim: 1, lower bound: -0.7494670, upper bound: 0.7815084
IS_B1_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 14.24
Output dim: 1, lower bound: -0.8300407, upper bound: 0.8257754
IS_B1_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 14.24
Output dim: 1, lower bound: -0.7494670, upper bound: 0.7815083
IS_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 14.24
Output dim: 1, lower bound: -0.9054458, upper bound: 0.9744456
IS_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 14.24
Output dim: 1, lower bound: -0.9054458, upper bound: 0.9747683
IS_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 14.24
Output dim: 1, lower bound: -0.9031350, upper bound: 0.9744460
IS_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 14.24
Output dim: 1, lower bound: -0.9031350, upper bound: 0.9747684
IS_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 14.24
Output dim: 1, lower bound: -0.8842574, upper bound: 0.9612677
IS_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 14.24
Output dim: 1, lower bound: -0.8842574, upper bound: 0.9612677
IS_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 14.24
Output dim: 1, lower bound: -0.8841186, upper bound: 0.9612678
IS_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 14.24
Output dim: 1, lower bound: -0.8841186, upper bound: 0.9612677
IS_B2_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 14.24
Output dim: 1, lower bound: -0.9065114, upper bound: 0.8530360
IS_B2_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 14.24
Output dim: 1, lower bound: -0.8158663, upper bound: 0.7934066
IS_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 14.24
Output dim: 1, lower bound: -0.8567408, upper bound: 0.8562876
IS_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 14.24
Output dim: 1, lower bound: -0.8169538, upper bound: 0.7934065
IS_B2_A1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 14.24
Output dim: 1, lower bound: -0.8750837, upper bound: 0.8127971
IS_B2_A1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 14.24
Output dim: 1, lower bound: -0.7782257, upper bound: 0.7494669
IS_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 14.24
Output dim: 1, lower bound: -0.8257753, upper bound: 0.8300405
IS_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 14.24
Output dim: 1, lower bound: -0.7815084, upper bound: 0.7494669
IS_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 14.24
Output dim: 1, lower bound: -0.8880345, upper bound: 0.9880249
IS_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 14.24
Output dim: 1, lower bound: -0.8880345, upper bound: 0.9882713
IS_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 14.24
Output dim: 1, lower bound: -0.8838170, upper bound: 0.9880247
IS_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 14.24
Output dim: 1, lower bound: -0.8838170, upper bound: 0.9882713
IS_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 14.24
Output dim: 1, lower bound: -0.8640902, upper bound: 0.9771827
IS_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 14.24
Output dim: 1, lower bound: -0.8605408, upper bound: 0.9771830
IS_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 14.24
Output dim: 1, lower bound: -0.9474879, upper bound: 0.9535115
IS_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 14.24
Output dim: 1, lower bound: -0.9483478, upper bound: 0.9535100
IS_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 14.24
Output dim: 1, lower bound: -0.9228046, upper bound: 0.9383472
IS_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 14.24
Output dim: 1, lower bound: -0.9232987, upper bound: 0.9383474
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
type: B, layer: 3, pos: 1928
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 1683
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 655
type: B, layer: 3, pos: 655
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.42 seconds

### Candidate
type: B, layer: 3, pos: 1928

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8444436, upper bound: 0.8543586
time: 4.52 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8543565, upper bound: 0.8543586
time: 5.03 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 9.98 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 9.98
Output dim: 1, lower bound: -0.8444436, upper bound: 0.8543586
IS_B2, status: Status.UNKNOWN, split count: 1, time: 9.98
Output dim: 1, lower bound: -0.8543565, upper bound: 0.8543586

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -7.3399820, -5.1205239, -7.2890615, -5.1841650, -1.8294921, 1.8177083
1: 1.9373870, 3.5871906, 1.9458289, 3.5798233, -1.2817743, 1.2689335
2: -4.9612403, -3.2775478, -4.9594469, -3.2859809, -1.2157462, 1.2233412
3: -11.0716782, -8.8745327, -11.0325260, -8.8911295, -1.6468410, 1.5795703
4: -5.6263137, -3.8402209, -5.6086173, -3.8460803, -1.5439324, 1.5773695
5: -9.0869417, -7.2697096, -9.0567703, -7.3194046, -1.7218280, 1.7870607
6: -6.5549726, -4.2857571, -6.5046525, -4.2907782, -1.6446648, 1.6112244
7: -8.8569775, -7.3833494, -8.8444023, -7.4142337, -1.0864391, 1.2407672
8: 0.9689927, 2.5435100, 0.9678707, 2.5229740, -1.2563066, 1.2103741
9: -9.4920349, -7.4001732, -9.4739857, -7.4211969, -1.6468105, 1.6428249

Time for backsubstitution: 5.80 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1683
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 655
type: B, layer: 3, pos: 655
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 1683

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8117850, upper bound: 0.8143894
time: 4.02 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8074072, upper bound: 0.8163624
time: 4.15 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -7.3397312, -5.1092644, -7.3349719, -5.1344824, -1.7900319, 1.8964176
1: 1.9375088, 3.5880408, 1.9418974, 3.5842206, -1.3163171, 1.2834496
2: -4.9613876, -3.2778080, -4.9584117, -3.2864370, -1.2162693, 1.2230279
3: -11.0744076, -8.8746071, -11.0543442, -8.8779135, -1.7158203, 1.6882687
4: -5.6256208, -3.8399391, -5.6055589, -3.8417616, -1.5949187, 1.5851812
5: -9.0873299, -7.2640867, -9.0845490, -7.2844133, -1.8029165, 1.8204622
6: -6.5574317, -4.2857304, -6.5309668, -4.2874689, -1.6724148, 1.6433034
7: -8.8569002, -7.3808432, -8.8546734, -7.3944950, -1.2297592, 1.2449226
8: 0.9688907, 2.5406384, 0.9718418, 2.5160675, -1.2885430, 1.2665670
9: -9.4920416, -7.3960786, -9.4891787, -7.4034848, -1.6665959, 1.6731017

Time for backsubstitution: 5.75 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1928
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 655
type: B, layer: 3, pos: 655
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 1683
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 1928

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8543587, upper bound: 0.8444411
time: 4.48 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8543587, upper bound: 0.8543563
time: 4.52 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 14.98 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 14.98
Output dim: 1, lower bound: -0.8117850, upper bound: 0.8143894
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 14.98
Output dim: 1, lower bound: -0.8074072, upper bound: 0.8163624
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 14.98
Output dim: 1, lower bound: -0.8543587, upper bound: 0.8444411
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 14.98
Output dim: 1, lower bound: -0.8543587, upper bound: 0.8543563

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -7.3392220, -5.1209040, -7.2890615, -5.1841650, -1.8288188, 1.8170173
1: 1.9382122, 3.5703845, 1.9458289, 3.5798233, -1.2809699, 1.2501268
2: -4.9589825, -3.2790785, -4.9594469, -3.2859809, -1.2066503, 1.2222426
3: -11.0708904, -8.8761711, -11.0325260, -8.8911295, -1.6447716, 1.5783753
4: -5.6017275, -3.8406565, -5.6086173, -3.8460803, -1.5181756, 1.5752609
5: -9.0822611, -7.2699533, -9.0567703, -7.3194046, -1.7164936, 1.7868171
6: -6.5539737, -4.2877932, -6.5046525, -4.2907782, -1.6437397, 1.6082029
7: -8.8551168, -7.3838763, -8.8444023, -7.4142337, -1.0776739, 1.2395716
8: 0.9838052, 2.5426064, 0.9678707, 2.5229740, -1.2466063, 1.2095225
9: -9.4509773, -7.4005437, -9.4739857, -7.4211969, -1.5972242, 1.6424439

Time for backsubstitution: 5.78 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 655
type: B, layer: 3, pos: 655
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 1928
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 655

## Relational analysis of IS_B1_A1_A1

### Relational analysis result of IS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7951858, upper bound: 0.7986033
time: 4.25 seconds

## Relational analysis of IS_B1_A1_A2

### Relational analysis result of IS_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7951858, upper bound: 0.7995272
time: 4.56 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -7.3410983, -5.1235886, -7.2886086, -5.1846800, -1.8308477, 1.8159633
1: 1.8638844, 3.5542510, 1.9464214, 3.5709906, -1.3616948, 1.2533641
2: -4.9236002, -3.2718682, -4.9503660, -3.2863712, -1.2014370, 1.2645227
3: -11.0433722, -8.8762436, -11.0259533, -8.8918371, -1.6309443, 1.5829740
4: -5.5712175, -3.7486205, -5.5933900, -3.8461533, -1.5525646, 1.6724007
5: -9.0864019, -7.2714024, -9.0542841, -7.3197122, -1.7285905, 1.7828817
6: -6.5481467, -4.3765736, -6.5038838, -4.3104677, -1.6676145, 1.5698082
7: -8.7572041, -7.3834553, -8.8222694, -7.4144001, -1.0610576, 1.2946157
8: 1.0125523, 2.6112700, 0.9779625, 2.5227032, -1.2456262, 1.2784870
9: -9.3761730, -7.2065544, -9.4441957, -7.4212685, -1.6035030, 1.8731933

Time for backsubstitution: 5.79 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 655
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 1928
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1683
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 655

## Relational analysis of IS_B1_A2_A1

### Relational analysis result of IS_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7900850, upper bound: 0.8012475
time: 4.06 seconds

## Relational analysis of IS_B1_A2_A2

### Relational analysis result of IS_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7900850, upper bound: 0.8009722
time: 3.92 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -7.2890615, -5.1841650, -7.3349719, -5.1344824, -1.8229713, 1.8259525
1: 1.9458289, 3.5798233, 1.9418974, 3.5842206, -1.2524271, 1.2664778
2: -4.9594469, -3.2859809, -4.9584117, -3.2864370, -1.2154949, 1.2137380
3: -11.0325260, -8.8911295, -11.0543442, -8.8779135, -1.5748963, 1.6524167
4: -5.6086173, -3.8460803, -5.6055589, -3.8417616, -1.5761533, 1.5298975
5: -9.0567703, -7.3194046, -9.0845490, -7.2844133, -1.7723570, 1.7022233
6: -6.5046525, -4.2907782, -6.5309668, -4.2874689, -1.6096425, 1.6359553
7: -8.8444023, -7.4142337, -8.8546734, -7.3944950, -1.2335365, 1.0703919
8: 0.9678707, 2.5229740, 0.9718418, 2.5160675, -1.2030125, 1.2539196
9: -9.4739857, -7.4211969, -9.4891787, -7.4034848, -1.6457586, 1.6449480

Time for backsubstitution: 5.78 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 1683
type: B, layer: 3, pos: 655
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 1683

## Relational analysis of IS_B2_A1_B1

### Relational analysis result of IS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8048755, upper bound: 0.8117855
time: 4.50 seconds

## Relational analysis of IS_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8074071, upper bound: 0.8074077
time: 4.59 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -7.3349719, -5.1344824, -7.3349719, -5.1344824, -1.7863393, 1.7863398
1: 1.9418974, 3.5842206, 1.9418974, 3.5842206, -1.3050637, 1.3050635
2: -4.9584117, -3.2864370, -4.9584117, -3.2864370, -1.2141483, 1.2141482
3: -11.0543442, -8.8779135, -11.0543442, -8.8779135, -1.6854858, 1.6854858
4: -5.6055589, -3.8417616, -5.6055589, -3.8417616, -1.5837264, 1.5837264
5: -9.0845490, -7.2844133, -9.0845490, -7.2844133, -1.8001356, 1.8001356
6: -6.5309668, -4.2874689, -6.5309668, -4.2874689, -1.6416910, 1.6416910
7: -8.8546734, -7.3944950, -8.8546734, -7.3944950, -1.2279625, 1.2279625
8: 0.9718418, 2.5160675, 0.9718418, 2.5160675, -1.2856467, 1.2856467
9: -9.4891787, -7.4034848, -9.4891787, -7.4034848, -1.6646967, 1.6646967

Time for backsubstitution: 5.77 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 655
type: B, layer: 3, pos: 655
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 1683

## Relational analysis of IS_B2_A2_B1

### Relational analysis result of IS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8048755, upper bound: 0.8196999
time: 4.02 seconds

## Relational analysis of IS_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8074071, upper bound: 0.8074072
time: 6.15 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 16.17 seconds
IS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 16.17
Output dim: 1, lower bound: -0.7951858, upper bound: 0.7986033
IS_B1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 16.17
Output dim: 1, lower bound: -0.7951858, upper bound: 0.7995272
IS_B1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 16.17
Output dim: 1, lower bound: -0.7900850, upper bound: 0.8012475
IS_B1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 16.17
Output dim: 1, lower bound: -0.7900850, upper bound: 0.8009722
IS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 16.17
Output dim: 1, lower bound: -0.8048755, upper bound: 0.8117855
IS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 16.17
Output dim: 1, lower bound: -0.8074071, upper bound: 0.8074077
IS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 16.17
Output dim: 1, lower bound: -0.8048755, upper bound: 0.8196999
IS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 16.17
Output dim: 1, lower bound: -0.8074071, upper bound: 0.8074072

## BFS IS instance: IS_B1_A1_A1

### Backsubstitution after applying IS history:
0: -7.3334265, -5.1228933, -7.2881575, -5.1842208, -1.8135405, 1.8079703
1: 1.9397063, 3.5569043, 1.9468918, 3.5773711, -1.2759397, 1.2414839
2: -4.9523854, -3.2858217, -4.9588413, -3.2867174, -1.2012348, 1.2134632
3: -11.0510597, -8.8781319, -11.0294714, -8.8916798, -1.6258364, 1.5719414
4: -5.5904279, -3.8430443, -5.6069069, -3.8461537, -1.5017166, 1.5659990
5: -9.0754185, -7.3077898, -9.0566378, -7.3248167, -1.6866813, 1.7488480
6: -6.5569901, -4.3220739, -6.5043769, -4.2957664, -1.6483178, 1.5881686
7: -8.8156128, -7.4005265, -8.8377171, -7.4144425, -1.0483153, 1.2131405
8: 1.0133190, 2.5498700, 0.9746165, 2.5229220, -1.2075782, 1.1907542
9: -9.4497604, -7.4040480, -9.4738035, -7.4215803, -1.5910487, 1.6377058

Time for backsubstitution: 5.72 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 1928
type: B, layer: 3, pos: 1683
type: B, layer: 3, pos: 655
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 423

## Relational analysis of IS_B1_A1_A1_B1

### Relational analysis result of IS_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7843858, upper bound: 0.7889452
time: 4.22 seconds

## Relational analysis of IS_B1_A1_A1_B2

### Relational analysis result of IS_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7859529, upper bound: 0.7893599
time: 3.97 seconds

## BFS IS instance: IS_B1_A1_A2

### Backsubstitution after applying IS history:
0: -7.3360157, -5.1210303, -7.2890615, -5.1841650, -1.8272605, 1.8081758
1: 1.9419346, 3.5591700, 1.9458289, 3.5798233, -1.2777865, 1.2492511
2: -4.9566231, -3.2811241, -4.9594469, -3.2859809, -1.2052317, 1.2170340
3: -11.0605583, -8.8789234, -11.0325260, -8.8911295, -1.6384335, 1.5765834
4: -5.5950804, -3.8411756, -5.6086173, -3.8460803, -1.5016432, 1.5748274
5: -9.0816441, -7.2820106, -9.0567703, -7.3194046, -1.7156882, 1.7747598
6: -6.5528793, -4.3064871, -6.5046525, -4.2907782, -1.6414113, 1.6165025
7: -8.8476954, -7.3854976, -8.8444023, -7.4142337, -1.0438120, 1.2388895
8: 1.0050373, 2.5423679, 0.9678707, 2.5229740, -1.2434816, 1.2051098
9: -9.4497356, -7.4019985, -9.4739857, -7.4211969, -1.5996652, 1.6402819

Time for backsubstitution: 5.70 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 655
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 1928
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 655

## Relational analysis of IS_B1_A1_A2_B1

### Relational analysis result of IS_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7937043, upper bound: 0.7995273
time: 4.25 seconds

## Relational analysis of IS_B1_A1_A2_B2

### Relational analysis result of IS_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7937043, upper bound: 0.7995284
time: 4.96 seconds

## BFS IS instance: IS_B1_A2_A1

### Backsubstitution after applying IS history:
0: -7.3352957, -5.1255655, -7.2877016, -5.1847363, -1.8154688, 1.8068275
1: 1.8607626, 3.5432196, 1.9474838, 3.5692923, -1.3568659, 1.2438729
2: -4.9180245, -3.2785726, -4.9497581, -3.2871132, -1.1972311, 1.2554855
3: -11.0240793, -8.8781872, -11.0228996, -8.8923874, -1.6119475, 1.5762444
4: -5.5635476, -3.7503653, -5.5917206, -3.8462248, -1.5369349, 1.6619225
5: -9.0792170, -7.3092556, -9.0541534, -7.3251233, -1.6992660, 1.7448978
6: -6.5484381, -4.4098649, -6.5036058, -4.3154545, -1.6726408, 1.5495934
7: -8.7217102, -7.3989773, -8.8155851, -7.4146099, -1.0308828, 1.2758970
8: 1.0369835, 2.6228752, 0.9847088, 2.5226493, -1.2074237, 1.2554770
9: -9.3750210, -7.2101297, -9.4440041, -7.4216518, -1.5975046, 1.8681231

Time for backsubstitution: 5.71 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 1928
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1683
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 423

## Relational analysis of IS_B1_A2_A1_B1

### Relational analysis result of IS_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7793541, upper bound: 0.7919021
time: 3.86 seconds

## Relational analysis of IS_B1_A2_A1_B2

### Relational analysis result of IS_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7808032, upper bound: 0.7919528
time: 3.86 seconds

## BFS IS instance: IS_B1_A2_A2

### Backsubstitution after applying IS history:
0: -7.3378868, -5.1237164, -7.2886086, -5.1846800, -1.8292179, 1.8070326
1: 1.8676064, 3.5407629, 1.9464214, 3.5709906, -1.3586154, 1.2521813
2: -4.9212313, -3.2738965, -4.9503660, -3.2863712, -1.2000182, 1.2591012
3: -11.0324068, -8.8790722, -11.0259533, -8.8918371, -1.6245027, 1.5811505
4: -5.5643601, -3.7490971, -5.5933900, -3.8461533, -1.5340619, 1.6719348
5: -9.0857639, -7.2834587, -9.0542841, -7.3197122, -1.7277894, 1.7708254
6: -6.5470810, -4.3937221, -6.5038838, -4.3104677, -1.6649632, 1.5778787
7: -8.7504454, -7.3850207, -8.8222694, -7.4144001, -1.0283234, 1.2939808
8: 1.0337715, 2.6110373, 0.9779625, 2.5227032, -1.2435808, 1.2735109
9: -9.3748379, -7.2079363, -9.4441957, -7.4212685, -1.6035271, 1.8709002

Time for backsubstitution: 5.72 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 655
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1683
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 655

## Relational analysis of IS_B1_A2_A2_B1

### Relational analysis result of IS_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7888266, upper bound: 0.8009724
time: 3.77 seconds

## Relational analysis of IS_B1_A2_A2_B2

### Relational analysis result of IS_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7888266, upper bound: 0.8009728
time: 3.75 seconds

## BFS IS instance: IS_B2_A1_B1

### Backsubstitution after applying IS history:
0: -7.2890615, -5.1841650, -7.3342299, -5.1349053, -1.8225236, 1.8252771
1: 1.9458289, 3.5798233, 1.9427099, 3.5675063, -1.2339060, 1.2656500
2: -4.9594469, -3.2859809, -4.9561667, -3.2879825, -1.2143080, 1.2046359
3: -11.0325260, -8.8911295, -11.0532627, -8.8795404, -1.5736341, 1.6500182
4: -5.6086173, -3.8460803, -5.5809526, -3.8421807, -1.5740252, 1.5060279
5: -9.0567703, -7.3194046, -9.0798616, -7.2846737, -1.7720966, 1.6967573
6: -6.5046525, -4.2907782, -6.5299716, -4.2894926, -1.6066289, 1.6350687
7: -8.8444023, -7.4142337, -8.8528118, -7.3950086, -1.2322979, 1.0602436
8: 0.9678707, 2.5229740, 0.9866581, 2.5151229, -1.2021749, 1.2441421
9: -9.4739857, -7.4211969, -9.4480572, -7.4038544, -1.6453638, 1.5954838

Time for backsubstitution: 5.71 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 655
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 655

## Relational analysis of IS_B2_A1_B1_B1

### Relational analysis result of IS_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7986033, upper bound: 0.7951865
time: 4.04 seconds

## Relational analysis of IS_B2_A1_B1_B2

### Relational analysis result of IS_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7995264, upper bound: 0.7951862
time: 4.47 seconds

## BFS IS instance: IS_B2_A1_B2

### Backsubstitution after applying IS history:
0: -7.2886086, -5.1846800, -7.3359895, -5.1373329, -1.8207092, 1.8271418
1: 1.9464214, 3.5709906, 1.8684154, 3.5498276, -1.2374320, 1.3443613
2: -4.9503660, -3.2863712, -4.9204550, -3.2813711, -1.2544842, 1.1993566
3: -11.0259533, -8.8918371, -11.0284929, -8.8796349, -1.5776491, 1.6371102
4: -5.5933900, -3.8461533, -5.5508881, -3.7506237, -1.6703777, 1.5397124
5: -9.0542841, -7.3197122, -9.0838490, -7.2861376, -1.7681465, 1.7074490
6: -6.5038838, -4.3104677, -6.5258422, -4.3784766, -1.5681677, 1.6589808
7: -8.8222694, -7.4144001, -8.7548981, -7.3989048, -1.2827656, 1.0429697
8: 0.9779625, 2.5227032, 1.0155959, 2.5790672, -1.2597344, 1.2430220
9: -9.4441957, -7.4212685, -9.3728790, -7.2114639, -1.8567646, 1.6014032

Time for backsubstitution: 5.70 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 655
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 655

## Relational analysis of IS_B2_A1_B2_B1

### Relational analysis result of IS_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8012467, upper bound: 0.7900857
time: 3.79 seconds

## Relational analysis of IS_B2_A1_B2_B2

### Relational analysis result of IS_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8009716, upper bound: 0.7900858
time: 3.53 seconds

## BFS IS instance: IS_B2_A2_B1

### Backsubstitution after applying IS history:
0: -7.3349719, -5.1344824, -7.3342299, -5.1349053, -1.7856469, 1.7857623
1: 1.9418974, 3.5842206, 1.9427099, 3.5675063, -1.2868190, 1.3042216
2: -4.9584117, -3.2864370, -4.9561667, -3.2879825, -1.2130318, 1.2050462
3: -11.0543442, -8.8779135, -11.0532627, -8.8795404, -1.6840715, 1.6830878
4: -5.6055589, -3.8417616, -5.5809526, -3.8421807, -1.5813828, 1.5562668
5: -9.0845490, -7.2844133, -9.0798616, -7.2846737, -1.7998753, 1.7954483
6: -6.5309668, -4.2874689, -6.5299716, -4.2894926, -1.6357074, 1.6408043
7: -8.8546734, -7.3944950, -8.8528118, -7.3950086, -1.2267480, 1.2170124
8: 0.9718418, 2.5160675, 0.9866581, 2.5151229, -1.2847171, 1.2765322
9: -9.4891787, -7.4034848, -9.4480572, -7.4038544, -1.6643236, 1.6146805

Time for backsubstitution: 5.83 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 655
type: B, layer: 3, pos: 655
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 655

## Relational analysis of IS_B2_A2_B1_A1

### Relational analysis result of IS_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7890557, upper bound: 0.8063631
time: 4.08 seconds

## Relational analysis of IS_B2_A2_B1_A2

### Relational analysis result of IS_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7890557, upper bound: 0.8046896
time: 4.12 seconds

## BFS IS instance: IS_B2_A2_B2

### Backsubstitution after applying IS history:
0: -7.3343830, -5.1351452, -7.3359895, -5.1373329, -1.7847009, 1.7889805
1: 1.9425921, 3.5751405, 1.8684154, 3.5498276, -1.2872043, 1.3735950
2: -4.9483581, -3.2868440, -4.9204550, -3.2813711, -1.2539356, 1.1995693
3: -11.0480003, -8.8787317, -11.0284929, -8.8796349, -1.6863384, 1.6699500
4: -5.5903006, -3.8419304, -5.5508881, -3.7506237, -1.6742334, 1.5877337
5: -9.0824890, -7.2848530, -9.0838490, -7.2861376, -1.7963514, 1.7989960
6: -6.5300188, -4.3092752, -6.5258422, -4.3784766, -1.5987773, 1.6627045
7: -8.8316011, -7.3947744, -8.7548981, -7.3989048, -1.2802262, 1.1972518
8: 0.9825697, 2.5157638, 1.0155959, 2.5790672, -1.3321817, 1.2724783
9: -9.4592772, -7.4035540, -9.3728790, -7.2114639, -1.8783803, 1.6208045

Time for backsubstitution: 5.84 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 655
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 2832
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 655

## Relational analysis of IS_B2_A2_B2_B1

### Relational analysis result of IS_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7936809, upper bound: 0.8011927
time: 3.78 seconds

## Relational analysis of IS_B2_A2_B2_B2

### Relational analysis result of IS_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7916956, upper bound: 0.8011919
time: 4.13 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 13.97 seconds
IS_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 13.97
Output dim: 1, lower bound: -0.7843858, upper bound: 0.7889452
IS_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 13.97
Output dim: 1, lower bound: -0.7859529, upper bound: 0.7893599
IS_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 13.97
Output dim: 1, lower bound: -0.7937043, upper bound: 0.7995273
IS_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 13.97
Output dim: 1, lower bound: -0.7937043, upper bound: 0.7995284
IS_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 13.97
Output dim: 1, lower bound: -0.7793541, upper bound: 0.7919021
IS_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 13.97
Output dim: 1, lower bound: -0.7808032, upper bound: 0.7919528
IS_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 13.97
Output dim: 1, lower bound: -0.7888266, upper bound: 0.8009724
IS_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 13.97
Output dim: 1, lower bound: -0.7888266, upper bound: 0.8009728
IS_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 13.97
Output dim: 1, lower bound: -0.7986033, upper bound: 0.7951865
IS_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 13.97
Output dim: 1, lower bound: -0.7995264, upper bound: 0.7951862
IS_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 13.97
Output dim: 1, lower bound: -0.8012467, upper bound: 0.7900857
IS_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 13.97
Output dim: 1, lower bound: -0.8009716, upper bound: 0.7900858
IS_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 13.97
Output dim: 1, lower bound: -0.7890557, upper bound: 0.8063631
IS_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 13.97
Output dim: 1, lower bound: -0.7890557, upper bound: 0.8046896
IS_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 13.97
Output dim: 1, lower bound: -0.7936809, upper bound: 0.8011927
IS_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 13.97
Output dim: 1, lower bound: -0.7916956, upper bound: 0.8011919

## BFS IS instance: IS_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -7.3328156, -5.1230278, -7.2846489, -5.1848550, -1.8121591, 1.8042793
1: 1.9431057, 3.5568769, 1.9647248, 3.5772171, -1.2690928, 1.2174463
2: -4.9512644, -3.2890999, -4.9530039, -3.3054171, -1.1785364, 1.2060101
3: -11.0509968, -8.8790007, -11.0294476, -8.8962717, -1.6199551, 1.5710742
4: -5.5896926, -3.8431087, -5.6027932, -3.8462076, -1.4987192, 1.5613995
5: -9.0751343, -7.3095584, -9.0551052, -7.3341613, -1.6727610, 1.7455468
6: -6.5533133, -4.3225908, -6.4838457, -4.2982903, -1.6377418, 1.5617812
7: -8.8148327, -7.4006176, -8.8335352, -7.4146147, -1.0472465, 1.2078731
8: 1.0158653, 2.5497303, 0.9885411, 2.5221534, -1.2041125, 1.1811428
9: -9.4496746, -7.4050083, -9.4735899, -7.4267883, -1.5813365, 1.6341460

Time for backsubstitution: 5.80 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 655
type: A, layer: 3, pos: 1928
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 655

## Relational analysis of IS_B1_A1_A1_B1_B1

### Relational analysis result of IS_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7828519, upper bound: 0.7889457
time: 3.85 seconds

## Relational analysis of IS_B1_A1_A1_B1_B2

### Relational analysis result of IS_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7828519, upper bound: 0.7889467
time: 4.16 seconds

## BFS IS instance: IS_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -7.3314476, -5.1229463, -7.2783546, -5.1893396, -1.8083782, 1.8028288
1: 1.9431906, 3.5568571, 1.9632540, 3.5738282, -1.2763255, 1.2248316
2: -4.9522066, -3.2886200, -4.9664812, -3.3001566, -1.1822677, 1.2243508
3: -11.0510006, -8.8788214, -11.0283031, -8.8950758, -1.6216526, 1.5706372
4: -5.5893226, -3.8431115, -5.6016359, -3.8466101, -1.4980640, 1.5610671
5: -9.0750055, -7.3087854, -9.0569544, -7.3292274, -1.6740127, 1.7481689
6: -6.5524945, -4.3221555, -6.4833550, -4.3118458, -1.6363668, 1.5567875
7: -8.8154564, -7.4006243, -8.8370733, -7.4146242, -1.0478427, 1.2111349
8: 1.0145950, 2.5496759, 0.9804764, 2.5315232, -1.2021739, 1.1893423
9: -9.4496508, -7.4053879, -9.4713564, -7.4280224, -1.5796595, 1.6339502

Time for backsubstitution: 5.74 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1928
type: B, layer: 3, pos: 1683
type: B, layer: 3, pos: 655
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 1928

## Relational analysis of IS_B1_A1_A1_B2_A1

### Relational analysis result of IS_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7859529, upper bound: 0.7866612
time: 3.98 seconds

## Relational analysis of IS_B1_A1_A1_B2_A2

### Relational analysis result of IS_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7859529, upper bound: 0.7893597
time: 4.11 seconds

## BFS IS instance: IS_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -7.3360157, -5.1210303, -7.2831717, -5.1860595, -1.8036995, 1.7951016
1: 1.9419346, 3.5591700, 1.9461946, 3.5677295, -1.2694676, 1.2389693
2: -4.9566231, -3.2811241, -4.9548240, -3.2925885, -1.1967502, 1.2156789
3: -11.0605583, -8.8789234, -11.0118694, -8.8928661, -1.6404853, 1.5593989
4: -5.5950804, -3.8411756, -5.5981255, -3.8481121, -1.5110726, 1.5579884
5: -9.0816441, -7.2820106, -9.0446062, -7.3571277, -1.6719246, 1.7625957
6: -6.5528793, -4.3064871, -6.5057964, -4.3236704, -1.6176491, 1.5771708
7: -8.8476954, -7.3854976, -8.7990742, -7.4342251, -1.0474935, 1.1953082
8: 1.0050373, 2.5423679, 1.0038567, 2.5262966, -1.2157907, 1.1569374
9: -9.4497356, -7.4019985, -9.4727840, -7.4243088, -1.5884223, 1.6364009

Time for backsubstitution: 5.73 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 1928
type: B, layer: 3, pos: 1683
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 423

## Relational analysis of IS_B1_A1_A2_B1_B1

### Relational analysis result of IS_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7828519, upper bound: 0.7896757
time: 3.88 seconds

## Relational analysis of IS_B1_A1_A2_B1_B2

### Relational analysis result of IS_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7844715, upper bound: 0.7902948
time: 4.11 seconds

## BFS IS instance: IS_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -7.3360157, -5.1210303, -7.2855878, -5.1842670, -1.8217063, 1.8197703
1: 1.9419346, 3.5591700, 1.9496813, 3.5686789, -1.2723989, 1.2459238
2: -4.9566231, -3.2811241, -4.9570961, -3.2880132, -1.1992893, 1.2155267
3: -11.0605583, -8.8789234, -11.0224342, -8.8939648, -1.6360130, 1.5631628
4: -5.5950804, -3.8411756, -5.6020403, -3.8464110, -1.5013137, 1.5559573
5: -9.0816441, -7.2820106, -9.0561609, -7.3312216, -1.6662302, 1.7741504
6: -6.5528793, -4.3064871, -6.5038824, -4.3086281, -1.6496229, 1.6148727
7: -8.8476954, -7.3854976, -8.8383360, -7.4155598, -1.0433469, 1.1989472
8: 1.0050373, 2.5423679, 0.9881859, 2.5227408, -1.2420762, 1.1951348
9: -9.4497356, -7.4019985, -9.4726963, -7.4228458, -1.5972526, 1.6424477

Time for backsubstitution: 5.72 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 1928
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 423

## Relational analysis of IS_B1_A1_A2_B2_A1

### Relational analysis result of IS_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7838293, upper bound: 0.7886803
time: 4.67 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2

### Relational analysis result of IS_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7844714, upper bound: 0.7902961
time: 4.51 seconds

## BFS IS instance: IS_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -7.3346815, -5.1257014, -7.2841921, -5.1853704, -1.8140974, 1.8031383
1: 1.8641927, 3.5431864, 1.9653170, 3.5691369, -1.3501382, 1.2198377
2: -4.9168887, -3.2818151, -4.9439034, -3.3058186, -1.1745124, 1.2479414
3: -11.0240202, -8.8790646, -11.0228701, -8.8969622, -1.6060653, 1.5753374
4: -5.5627766, -3.7504187, -5.5876284, -3.8462799, -1.5339537, 1.6573229
5: -9.0789204, -7.3110199, -9.0526304, -7.3344679, -1.6853576, 1.7416105
6: -6.5447588, -4.4103842, -6.4830751, -4.3179798, -1.6620517, 1.5232542
7: -8.7209301, -7.3990741, -8.8114071, -7.4147811, -1.0298150, 1.2706351
8: 1.0395288, 2.6227388, 0.9986320, 2.5218811, -1.2039900, 1.2461331
9: -9.3749256, -7.2110853, -9.4438000, -7.4268608, -1.5878015, 1.8645473

Time for backsubstitution: 5.78 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 655
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 1683
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 655

## Relational analysis of IS_B1_A2_A1_B1_B1

### Relational analysis result of IS_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7782047, upper bound: 0.7919013
time: 3.94 seconds

## Relational analysis of IS_B1_A2_A1_B1_B2

### Relational analysis result of IS_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7782047, upper bound: 0.7919033
time: 4.14 seconds

## BFS IS instance: IS_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -7.3333139, -5.1256185, -7.2778945, -5.1898537, -1.8103070, 1.8016894
1: 1.8642519, 3.5431631, 1.9638524, 3.5657437, -1.3572602, 1.2272243
2: -4.9178438, -3.2813234, -4.9573951, -3.3005514, -1.1782608, 1.2663729
3: -11.0240231, -8.8788862, -11.0217276, -8.8957863, -1.6077638, 1.5749149
4: -5.5623951, -3.7504215, -5.5864768, -3.8466840, -1.5333056, 1.6569862
5: -9.0787897, -7.3102474, -9.0544624, -7.3295341, -1.6866169, 1.7442150
6: -6.5439415, -4.4099464, -6.4825797, -4.3315344, -1.6607740, 1.5182190
7: -8.7215557, -7.3990784, -8.8149462, -7.4147921, -1.0304108, 1.2738967
8: 1.0382614, 2.6226859, 0.9905672, 2.5312634, -1.2020369, 1.2541351
9: -9.3748989, -7.2114639, -9.4415627, -7.4280930, -1.5861177, 1.8643808

Time for backsubstitution: 5.70 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1928
type: B, layer: 3, pos: 655
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 1928

## Relational analysis of IS_B1_A2_A1_B2_A1

### Relational analysis result of IS_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7808032, upper bound: 0.7873925
time: 4.33 seconds

## Relational analysis of IS_B1_A2_A1_B2_A2

### Relational analysis result of IS_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7808032, upper bound: 0.7919528
time: 4.26 seconds

## BFS IS instance: IS_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -7.3378868, -5.1237164, -7.2827125, -5.1865745, -1.8056831, 1.7939458
1: 1.8676064, 3.5407629, 1.9468184, 3.5596476, -1.3502169, 1.2421255
2: -4.9212313, -3.2738965, -4.9455986, -3.2929935, -1.1915319, 1.2580107
3: -11.0324068, -8.8790722, -11.0054550, -8.8935566, -1.6267643, 1.5639629
4: -5.5643601, -3.7490971, -5.5842242, -3.8481855, -1.5456295, 1.6552141
5: -9.0857639, -7.2834587, -9.0422268, -7.3574333, -1.6840248, 1.7587681
6: -6.5470810, -4.3937221, -6.5050187, -4.3433590, -1.6419415, 1.5378275
7: -8.7504454, -7.3850207, -8.7770510, -7.4343929, -1.0308888, 1.2512457
8: 1.0337715, 2.6110373, 1.0125518, 2.5260081, -1.2154188, 1.2247529
9: -9.3748379, -7.2079363, -9.4431295, -7.4243798, -1.5947723, 1.8672752

Time for backsubstitution: 5.68 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 1928
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1683
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 423

## Relational analysis of IS_B1_A2_A2_B1_B1

### Relational analysis result of IS_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7782047, upper bound: 0.7913871
time: 3.94 seconds

## Relational analysis of IS_B1_A2_A2_B1_B2

### Relational analysis result of IS_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7795343, upper bound: 0.7916859
time: 3.91 seconds

## BFS IS instance: IS_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -7.3378868, -5.1237164, -7.2851295, -5.1847796, -1.8236628, 1.8186340
1: 1.8676064, 3.5407629, 1.9502730, 3.5597296, -1.3527217, 1.2488458
2: -4.9212313, -3.2738965, -4.9480128, -3.2884166, -1.1952882, 1.2575941
3: -11.0324068, -8.8790722, -11.0158596, -8.8946724, -1.6220779, 1.5676916
4: -5.5643601, -3.7490971, -5.5868244, -3.8464837, -1.5337324, 1.6523263
5: -9.0857639, -7.2834587, -9.0536833, -7.3315282, -1.6797705, 1.7702246
6: -6.5470810, -4.3937221, -6.5031157, -4.3280997, -1.6747918, 1.5762489
7: -8.7504454, -7.3850207, -8.8162041, -7.4157281, -1.0278220, 1.2611656
8: 1.0337715, 2.6110373, 0.9982748, 2.5224695, -1.2421794, 1.2581358
9: -9.3748379, -7.2079363, -9.4428596, -7.4229183, -1.6011145, 1.8724909

Time for backsubstitution: 5.69 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 1928
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1683
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 423

## Relational analysis of IS_B1_A2_A2_B2_A1

### Relational analysis result of IS_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7792046, upper bound: 0.7903795
time: 3.82 seconds

## Relational analysis of IS_B1_A2_A2_B2_A2

### Relational analysis result of IS_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7795342, upper bound: 0.7916853
time: 4.55 seconds

## BFS IS instance: IS_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -7.2881575, -5.1842208, -7.3286157, -5.1363916, -1.8167872, 1.8100624
1: 1.9468918, 3.5773711, 1.9441571, 3.5531807, -1.2243843, 1.2646043
2: -4.9588413, -3.2867174, -4.9495296, -3.2945380, -1.2055683, 1.1991709
3: -11.0294714, -8.8916798, -11.0336456, -8.8816414, -1.5674734, 1.6305122
4: -5.6069069, -3.8461537, -5.5696673, -3.8445563, -1.5647750, 1.4883809
5: -9.0566378, -7.3248167, -9.0729828, -7.3224711, -1.7341666, 1.6669474
6: -6.5043769, -4.2957664, -6.5283980, -4.3237524, -1.5865664, 1.6411037
7: -8.8377171, -7.4144425, -8.8134422, -7.4072409, -1.2118649, 1.0285516
8: 0.9746165, 2.5229220, 1.0161257, 2.5287914, -1.1870878, 1.2051363
9: -9.4738035, -7.4215803, -9.4467335, -7.4073257, -1.6407387, 1.5891733

Time for backsubstitution: 5.72 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 655
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 423

## Relational analysis of IS_B2_A1_B1_B1_A1

### Relational analysis result of IS_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7889447, upper bound: 0.7843868
time: 4.19 seconds

## Relational analysis of IS_B2_A1_B1_B1_A2

### Relational analysis result of IS_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7893593, upper bound: 0.7859535
time: 4.13 seconds

## BFS IS instance: IS_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -7.2890615, -5.1841650, -7.3317451, -5.1350422, -1.8170018, 1.8240647
1: 1.9458289, 3.5798233, 1.9460990, 3.5563092, -1.2322695, 1.2629218
2: -4.9594469, -3.2859809, -4.9537902, -3.2895765, -1.2094843, 1.2032125
3: -11.0325260, -8.8911295, -11.0418968, -8.8818893, -1.5723009, 1.6430235
4: -5.6086173, -3.8460803, -5.5743446, -3.8427057, -1.5735698, 1.4889162
5: -9.0567703, -7.3194046, -9.0792437, -7.2967434, -1.7600269, 1.6959596
6: -6.5046525, -4.2907782, -6.5288715, -4.3079991, -1.6151209, 1.6326478
7: -8.8444023, -7.4142337, -8.8453779, -7.3960986, -1.2317841, 1.0239215
8: 0.9678707, 2.5229740, 1.0076389, 2.5148697, -1.1979172, 1.2412558
9: -9.4739857, -7.4211969, -9.4472132, -7.4053216, -1.6431513, 1.5982051

Time for backsubstitution: 5.71 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 655
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 1683
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 655

## Relational analysis of IS_B2_A1_B1_B2_A1

### Relational analysis result of IS_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7995264, upper bound: 0.7937053
time: 4.34 seconds

## Relational analysis of IS_B2_A1_B1_B2_A2

### Relational analysis result of IS_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7995264, upper bound: 0.7951867
time: 4.39 seconds

## BFS IS instance: IS_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -7.2877016, -5.1847363, -7.3303652, -5.1388168, -1.8149724, 1.8118272
1: 1.9474838, 3.5692923, 1.8651626, 3.5391002, -1.2275822, 1.3436294
2: -4.9497581, -3.2871132, -4.9148488, -3.2879658, -1.2457979, 1.1949828
3: -11.0228996, -8.8923874, -11.0095682, -8.8816748, -1.5710940, 1.6175971
4: -5.5917206, -3.8462248, -5.5389109, -3.7524118, -1.6599116, 1.5227275
5: -9.0541534, -7.3251233, -9.0766144, -7.3239007, -1.7302527, 1.6778760
6: -6.5036058, -4.3154545, -6.5225439, -4.4117665, -1.5478849, 1.6710196
7: -8.8155851, -7.4146099, -8.7195406, -7.4092679, -1.2678730, 1.0111227
8: 0.9847088, 2.5226493, 1.0399756, 2.5968561, -1.2479331, 1.2048414
9: -9.4440041, -7.4216518, -9.3716259, -7.2150040, -1.8517478, 1.5952880

Time for backsubstitution: 5.74 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 2832
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 423

## Relational analysis of IS_B2_A1_B2_B1_A1

### Relational analysis result of IS_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7919014, upper bound: 0.7793546
time: 3.86 seconds

## Relational analysis of IS_B2_A1_B2_B1_A2

### Relational analysis result of IS_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7919523, upper bound: 0.7808038
time: 3.93 seconds

## BFS IS instance: IS_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -7.2886086, -5.1846800, -7.3334985, -5.1374702, -1.8151855, 1.8258643
1: 1.9464214, 3.5709906, 1.8718026, 3.5372758, -1.2358916, 1.3417192
2: -4.9503660, -3.2863712, -4.9180689, -3.2829709, -1.2492807, 1.1979318
3: -11.0259533, -8.8918371, -11.0164356, -8.8820229, -1.5761776, 1.6300640
4: -5.5933900, -3.8461533, -5.5444059, -3.7511063, -1.6698818, 1.5204155
5: -9.0542841, -7.3197122, -9.0832100, -7.2982144, -1.7560697, 1.7066512
6: -6.5038838, -4.3104677, -6.5247412, -4.3954382, -1.5764251, 1.6565495
7: -8.8222694, -7.4144001, -8.7481422, -7.3999834, -1.2823229, 1.0083926
8: 0.9779625, 2.5227032, 1.0365620, 2.5788250, -1.2548685, 1.2412875
9: -9.4441957, -7.4212685, -9.3719721, -7.2128544, -1.8544660, 1.6017156

Time for backsubstitution: 5.73 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 2832
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 655

## Relational analysis of IS_B2_A1_B2_B2_A1

### Relational analysis result of IS_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8009716, upper bound: 0.7888272
time: 3.84 seconds

## Relational analysis of IS_B2_A1_B2_B2_A2

### Relational analysis result of IS_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.8009716, upper bound: 0.7900857
time: 3.72 seconds

## BFS IS instance: IS_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -7.3293533, -5.1359677, -7.3333926, -5.1349697, -1.7712283, 1.7739544
1: 1.9433136, 3.5706356, 1.9437702, 3.5641651, -1.2910225, 1.2953782
2: -4.9517732, -3.2929888, -4.9555593, -3.2887537, -1.2083611, 1.1964078
3: -11.0349464, -8.8800135, -11.0504093, -8.8801031, -1.6648579, 1.6816940
4: -5.5942645, -3.8441346, -5.5792894, -3.8422790, -1.5644736, 1.5497789
5: -9.0775299, -7.3222146, -9.0797281, -7.2901263, -1.7874036, 1.7575135
6: -6.5294418, -4.3217249, -6.5296593, -4.2946677, -1.6434500, 1.6212850
7: -8.8159628, -7.4067430, -8.8464317, -7.3952632, -1.1858153, 1.1905785
8: 1.0006075, 2.5297356, 0.9932017, 2.5150647, -1.2379532, 1.2574799
9: -9.4875069, -7.4069538, -9.4478693, -7.4042974, -1.6582401, 1.6099734

Time for backsubstitution: 5.69 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 1683
type: B, layer: 3, pos: 655
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 423

## Relational analysis of IS_B2_A2_B1_A1_B1

### Relational analysis result of IS_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7781062, upper bound: 0.7967099
time: 4.16 seconds

## Relational analysis of IS_B2_A2_B1_A1_B2

### Relational analysis result of IS_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7798083, upper bound: 0.7970947
time: 4.51 seconds

## BFS IS instance: IS_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -7.3324833, -5.1346207, -7.3342299, -5.1349053, -1.7897320, 1.7758894
1: 1.9452822, 3.5730186, 1.9427099, 3.5675063, -1.2841473, 1.3084874
2: -4.9560342, -3.2880180, -4.9561667, -3.2879825, -1.2116086, 1.1998851
3: -11.0429764, -8.8802528, -11.0532627, -8.8795404, -1.6745052, 1.6812820
4: -5.5990276, -3.8422863, -5.5809526, -3.8421807, -1.5659981, 1.5558233
5: -9.0839310, -7.2964821, -9.0798616, -7.2846737, -1.7992573, 1.7643499
6: -6.5298676, -4.3059726, -6.5299716, -4.2894926, -1.6332865, 1.6523340
7: -8.8472385, -7.3955851, -8.8528118, -7.3950086, -1.1845229, 1.2165244
8: 0.9928179, 2.5158205, 0.9866581, 2.5151229, -1.2600658, 1.2717717
9: -9.4883165, -7.4049478, -9.4480572, -7.4038544, -1.6669180, 1.6123497

Time for backsubstitution: 5.72 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 655
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 2832
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 655

## Relational analysis of IS_B2_A2_B1_A2_B1

### Relational analysis result of IS_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7875273, upper bound: 0.8046895
time: 4.51 seconds

## Relational analysis of IS_B2_A2_B1_A2_B2

### Relational analysis result of IS_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7875273, upper bound: 0.8046902
time: 4.72 seconds

## BFS IS instance: IS_B2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -7.3335438, -5.1352105, -7.3303652, -5.1388168, -1.7728353, 1.7745278
1: 1.9436522, 3.5720253, 1.8651626, 3.5391002, -1.2780023, 1.3828819
2: -4.9477510, -3.2876179, -4.9148488, -3.2879658, -1.2452686, 1.1954685
3: -11.0452347, -8.8792906, -11.0095682, -8.8816748, -1.6846128, 1.6507301
4: -5.5886369, -3.8420300, -5.5389109, -3.7524118, -1.6648412, 1.5722709
5: -9.0823517, -7.2903090, -9.0766144, -7.3239007, -1.7584510, 1.7863054
6: -6.5297079, -4.3144512, -6.5225439, -4.4117665, -1.5795841, 1.6750815
7: -8.8252192, -7.3950291, -8.7195406, -7.4092679, -1.2616103, 1.1563580
8: 0.9891138, 2.5157070, 1.0399756, 2.5968561, -1.3122222, 1.2253652
9: -9.4590797, -7.4039993, -9.3716259, -7.2150040, -1.8733361, 1.6146975

Time for backsubstitution: 5.71 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 423

## Relational analysis of IS_B2_A2_B2_B1_A1

### Relational analysis result of IS_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7844132, upper bound: 0.7908007
time: 3.85 seconds

## Relational analysis of IS_B2_A2_B2_B1_A2

### Relational analysis result of IS_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7843819, upper bound: 0.7919014
time: 3.59 seconds

## BFS IS instance: IS_B2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -7.3343830, -5.1351452, -7.3334985, -5.1374702, -1.7748289, 1.7927809
1: 1.9425921, 3.5751405, 1.8718026, 3.5372758, -1.2911892, 1.3709669
2: -4.9483581, -3.2868440, -4.9180689, -3.2829709, -1.2491474, 1.1981456
3: -11.0480003, -8.8787317, -11.0164356, -8.8820229, -1.6845193, 1.6606789
4: -5.5903006, -3.8419304, -5.5444059, -3.7511063, -1.6737833, 1.5710921
5: -9.0824890, -7.2848530, -9.0832100, -7.2982144, -1.7646966, 1.7983570
6: -6.5300188, -4.3092752, -6.5247412, -4.3954382, -1.6133008, 1.6602726
7: -8.8316011, -7.3947744, -8.7481422, -7.3999834, -1.2797837, 1.1548593
8: 0.9825697, 2.5157638, 1.0365620, 2.5788250, -1.3265617, 1.2476916
9: -9.4592772, -7.4035540, -9.3719721, -7.2128544, -1.8759682, 1.6211221

Time for backsubstitution: 5.69 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 655
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1683
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 2832
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 655

## Relational analysis of IS_B2_A2_B2_B2_A1

### Relational analysis result of IS_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7916957, upper bound: 0.7983141
time: 3.94 seconds

## Relational analysis of IS_B2_A2_B2_B2_A2

### Relational analysis result of IS_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7916957, upper bound: 0.8011917
time: 3.94 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 13.79 seconds
IS_B1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 1, lower bound: -0.7828519, upper bound: 0.7889457
IS_B1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 1, lower bound: -0.7828519, upper bound: 0.7889467
IS_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 1, lower bound: -0.7859529, upper bound: 0.7866612
IS_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 1, lower bound: -0.7859529, upper bound: 0.7893597
IS_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 1, lower bound: -0.7828519, upper bound: 0.7896757
IS_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 1, lower bound: -0.7844715, upper bound: 0.7902948
IS_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 1, lower bound: -0.7838293, upper bound: 0.7886803
IS_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 1, lower bound: -0.7844714, upper bound: 0.7902961
IS_B1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 1, lower bound: -0.7782047, upper bound: 0.7919013
IS_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 1, lower bound: -0.7782047, upper bound: 0.7919033
IS_B1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 1, lower bound: -0.7808032, upper bound: 0.7873925
IS_B1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 1, lower bound: -0.7808032, upper bound: 0.7919528
IS_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 1, lower bound: -0.7782047, upper bound: 0.7913871
IS_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 1, lower bound: -0.7795343, upper bound: 0.7916859
IS_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 1, lower bound: -0.7792046, upper bound: 0.7903795
IS_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 1, lower bound: -0.7795342, upper bound: 0.7916853
IS_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 1, lower bound: -0.7889447, upper bound: 0.7843868
IS_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 1, lower bound: -0.7893593, upper bound: 0.7859535
IS_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 1, lower bound: -0.7995264, upper bound: 0.7937053
IS_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 1, lower bound: -0.7995264, upper bound: 0.7951867
IS_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 1, lower bound: -0.7919014, upper bound: 0.7793546
IS_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 1, lower bound: -0.7919523, upper bound: 0.7808038
IS_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 1, lower bound: -0.8009716, upper bound: 0.7888272
IS_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 1, lower bound: -0.8009716, upper bound: 0.7900857
IS_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 1, lower bound: -0.7781062, upper bound: 0.7967099
IS_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 1, lower bound: -0.7798083, upper bound: 0.7970947
IS_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 1, lower bound: -0.7875273, upper bound: 0.8046895
IS_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 1, lower bound: -0.7875273, upper bound: 0.8046902
IS_B2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 1, lower bound: -0.7844132, upper bound: 0.7908007
IS_B2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 1, lower bound: -0.7843819, upper bound: 0.7919014
IS_B2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 1, lower bound: -0.7916957, upper bound: 0.7983141
IS_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.79
Output dim: 1, lower bound: -0.7916957, upper bound: 0.8011917

## BFS IS instance: IS_B1_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -7.3328156, -5.1230278, -7.2796669, -5.1865802, -1.8017750, 1.7925007
1: 1.9431057, 3.5568769, 1.9623249, 3.5675743, -1.2620723, 1.2217920
2: -4.9512644, -3.2890999, -4.9505796, -3.3112695, -1.1711831, 1.2030549
3: -11.0509968, -8.8790007, -11.0118446, -8.8969116, -1.6274495, 1.5563872
4: -5.5896926, -3.8431087, -5.5939922, -3.8481662, -1.4965882, 1.5471652
5: -9.0751343, -7.3095584, -9.0430088, -7.3657594, -1.6362152, 1.7334504
6: -6.5533133, -4.3225908, -6.4852734, -4.3255067, -1.6195502, 1.5633316
7: -8.8148327, -7.4006176, -8.7960920, -7.4343882, -1.0246849, 1.1721606
8: 1.0158653, 2.5497303, 1.0169563, 2.5255079, -1.1947517, 1.1438737
9: -9.4496746, -7.4050083, -9.4725695, -7.4290748, -1.5780263, 1.6305206

Time for backsubstitution: 5.73 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1928
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 1928

## Relational analysis of IS_B1_A1_A1_B1_B1_A1

### Relational analysis result of IS_B1_A1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7828519, upper bound: 0.7858038
time: 3.76 seconds

## Relational analysis of IS_B1_A1_A1_B1_B1_A2

### Relational analysis result of IS_B1_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7828519, upper bound: 0.7889457
time: 3.86 seconds

## BFS IS instance: IS_B1_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -7.3328156, -5.1230278, -7.2820778, -5.1848850, -1.8029084, 1.7926633
1: 1.9431057, 3.5568769, 1.9672694, 3.5685234, -1.2592444, 1.2155132
2: -4.9512644, -3.2890999, -4.9514408, -3.3067105, -1.1771822, 1.2050073
3: -11.0509968, -8.8790007, -11.0224085, -8.8985119, -1.6181731, 1.5645161
4: -5.5896926, -3.8431087, -5.5978460, -3.8464639, -1.4984608, 1.5587807
5: -9.0751343, -7.3095584, -9.0546236, -7.3404794, -1.6714563, 1.7450652
6: -6.5533133, -4.3225908, -6.4833522, -4.3111014, -1.6228480, 1.5593195
7: -8.8148327, -7.4006176, -8.8342781, -7.4157314, -1.0467744, 1.2084196
8: 1.0158653, 2.5497303, 1.0020771, 2.5219688, -1.1998301, 1.1726623
9: -9.4496746, -7.4050083, -9.4724846, -7.4279919, -1.5799642, 1.6307456

Time for backsubstitution: 5.69 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1928
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 1928

## Relational analysis of IS_B1_A1_A1_B1_B2_A1

### Relational analysis result of IS_B1_A1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7828519, upper bound: 0.7858036
time: 4.26 seconds

## Relational analysis of IS_B1_A1_A1_B1_B2_A2

### Relational analysis result of IS_B1_A1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7828519, upper bound: 0.7889467
time: 4.01 seconds

## BFS IS instance: IS_B1_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -7.2804942, -5.1866255, -7.2783546, -5.1893396, -1.7456174, 1.7478099
1: 1.9505093, 3.5500314, 1.9632540, 3.5738282, -1.2536631, 1.2092855
2: -4.9517307, -3.2969484, -4.9664812, -3.3001566, -1.1806288, 1.2150742
3: -11.0110836, -8.8952570, -11.0283031, -8.8950758, -1.4908051, 1.5091362
4: -5.5715785, -3.8484318, -5.6016359, -3.8466101, -1.4804010, 1.5141196
5: -9.0400429, -7.3581328, -9.0569544, -7.3292274, -1.6111965, 1.6136670
6: -6.5004959, -4.3275881, -6.4833550, -4.3118458, -1.5969973, 1.5499268
7: -8.7963982, -7.4346600, -8.8370733, -7.4146242, -1.0240257, 1.0435481
8: 1.0209060, 2.5251350, 0.9804764, 2.5315232, -1.1237037, 1.1623130
9: -9.4312639, -7.4260368, -9.4713564, -7.4280224, -1.5516884, 1.6099253

Time for backsubstitution: 5.71 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1683
type: B, layer: 3, pos: 655
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 1683

## Relational analysis of IS_B1_A1_A1_B2_A1_B1

### Relational analysis result of IS_B1_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7859529, upper bound: 0.7866613
time: 3.99 seconds

## Relational analysis of IS_B1_A1_A1_B2_A1_B2

### Relational analysis result of IS_B1_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7859529, upper bound: 0.7866612
time: 4.00 seconds

## BFS IS instance: IS_B1_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -7.3266349, -5.1364422, -7.2783546, -5.1893396, -1.8049006, 1.8116517
1: 1.9476366, 3.5531363, 1.9632540, 3.5738282, -1.2649887, 1.2077327
2: -4.9493523, -3.2973299, -4.9664812, -3.3001566, -1.1802065, 1.2164322
3: -11.0335913, -8.8823290, -11.0283031, -8.8950758, -1.6263309, 1.5661757
4: -5.5685644, -3.8446236, -5.6016359, -3.8466101, -1.4846783, 1.5598369
5: -9.0725670, -7.3234706, -9.0569544, -7.3292274, -1.6542850, 1.7334838
6: -6.5238981, -4.3238344, -6.4833550, -4.3118458, -1.6291375, 1.5551772
7: -8.8132887, -7.4073405, -8.8370733, -7.4146242, -1.0280783, 1.2098632
8: 1.0174031, 2.5285821, 0.9804764, 2.5315232, -1.1997249, 1.1856818
9: -9.4466248, -7.4086637, -9.4713564, -7.4280224, -1.5777836, 1.6369801

Time for backsubstitution: 5.67 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1683
type: B, layer: 3, pos: 655
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 1683

## Relational analysis of IS_B1_A1_A1_B2_A2_B1

### Relational analysis result of IS_B1_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7859529, upper bound: 0.7893596
time: 4.13 seconds

## Relational analysis of IS_B1_A1_A1_B2_A2_B2

### Relational analysis result of IS_B1_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7859529, upper bound: 0.7893597
time: 4.18 seconds

## BFS IS instance: IS_B1_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -7.3354044, -5.1211634, -7.2796669, -5.1865802, -1.8024592, 1.7914550
1: 1.9452155, 3.5591416, 1.9623249, 3.5675743, -1.2626655, 1.2170801
2: -4.9555264, -3.2844050, -4.9505796, -3.3112695, -1.1741016, 1.2090294
3: -11.0604944, -8.8797970, -11.0118446, -8.8969116, -1.6349669, 1.5585635
4: -5.5943289, -3.8412371, -5.5939922, -3.8481662, -1.5080609, 1.5534534
5: -9.0813513, -7.2837782, -9.0430088, -7.3657594, -1.6588254, 1.7592306
6: -6.5492005, -4.3070016, -6.4852734, -4.3255067, -1.6077523, 1.5508792
7: -8.8468771, -7.3855934, -8.7960920, -7.4343882, -1.0463409, 1.1912138
8: 1.0076237, 2.5422239, 1.0169563, 2.5255079, -1.2122755, 1.1482525
9: -9.4496479, -7.4029579, -9.4725695, -7.4290748, -1.5792499, 1.6328201

Time for backsubstitution: 5.70 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1928
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2832
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 1928

## Relational analysis of IS_B1_A1_A2_B1_B1_A1

### Relational analysis result of IS_B1_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7877630, upper bound: 0.7861271
time: 4.07 seconds

## Relational analysis of IS_B1_A1_A2_B1_B1_A2

### Relational analysis result of IS_B1_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7877630, upper bound: 0.7896755
time: 4.15 seconds

## BFS IS instance: IS_B1_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -7.3340349, -5.1210828, -7.2733736, -5.1911802, -1.7985415, 1.7899628
1: 1.9454198, 3.5591226, 1.9626236, 3.5641844, -1.2698579, 1.2222731
2: -4.9564505, -3.2839270, -4.9624848, -3.3058743, -1.1777864, 1.2265668
3: -11.0604973, -8.8796206, -11.0107021, -8.8962679, -1.6362100, 1.5581024
4: -5.5939531, -3.8412428, -5.5927997, -3.8485696, -1.5073919, 1.5530901
5: -9.0812216, -7.2830043, -9.0448875, -7.3615341, -1.6592417, 1.7618833
6: -6.5483794, -4.3065696, -6.4847736, -4.3397694, -1.6058838, 1.5459094
7: -8.8475332, -7.3856010, -8.7984476, -7.4343963, -1.0470073, 1.1933081
8: 1.0063210, 2.5421691, 1.0097055, 2.5348797, -1.2103562, 1.1555614
9: -9.4496260, -7.4033422, -9.4703455, -7.4307413, -1.5770845, 1.6326439

Time for backsubstitution: 5.82 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1928
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 1928

## Relational analysis of IS_B1_A1_A2_B1_B2_A1

### Relational analysis result of IS_B1_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7891247, upper bound: 0.7871565
time: 3.95 seconds

## Relational analysis of IS_B1_A1_A2_B1_B2_A2

### Relational analysis result of IS_B1_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7891247, upper bound: 0.7902947
time: 3.88 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -7.3325100, -5.1217799, -7.2849751, -5.1843829, -1.8182054, 1.8183424
1: 1.9603314, 3.5590067, 1.9529581, 3.5686536, -1.2462280, 1.2401502
2: -4.9505129, -3.2998633, -4.9559741, -3.2912869, -1.1917953, 1.1928060
3: -11.0601950, -8.8837204, -11.0224304, -8.8948288, -1.6336441, 1.5585532
4: -5.5908985, -3.8415303, -5.6012888, -3.8464189, -1.4966297, 1.5534325
5: -9.0800152, -7.2919965, -9.0558853, -7.3329115, -1.6607060, 1.7638888
6: -6.5317926, -4.3093510, -6.5003018, -4.3091202, -1.6235869, 1.6057723
7: -8.8432665, -7.3860493, -8.8375139, -7.4155893, -1.0390806, 1.1972232
8: 1.0196838, 2.5415764, 0.9906864, 2.5226026, -1.2321088, 1.1924528
9: -9.4492435, -7.4073968, -9.4726610, -7.4237895, -1.5938072, 1.6328478

Time for backsubstitution: 5.78 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1928
type: B, layer: 3, pos: 1683
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 1928

## Relational analysis of IS_B1_A1_A2_B2_A1_A1

### Relational analysis result of IS_B1_A1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7838293, upper bound: 0.7853033
time: 5.99 seconds

## Relational analysis of IS_B1_A1_A2_B2_A1_A2

### Relational analysis result of IS_B1_A1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7838293, upper bound: 0.7886813
time: 4.14 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -7.3262157, -5.1261721, -7.2836032, -5.1843109, -1.8166709, 1.8145680
1: 1.9583461, 3.5555530, 1.9531500, 3.5686336, -1.2549462, 1.2467453
2: -4.9642668, -3.2946908, -4.9569197, -3.2908022, -1.2100992, 1.1965092
3: -11.0591068, -8.8822861, -11.0224304, -8.8946657, -1.6332526, 1.5598748
4: -5.5897856, -3.8418553, -5.6009064, -3.8464265, -1.4968767, 1.5528362
5: -9.0818901, -7.2867408, -9.0557604, -7.3321466, -1.6667385, 1.7690196
6: -6.5314870, -4.3226299, -6.4994807, -4.3087063, -1.6178658, 1.6039395
7: -8.8470545, -7.3860059, -8.8381720, -7.4155955, -1.0424447, 1.1986642
8: 1.0112209, 2.5509768, 0.9894009, 2.5225487, -1.2419143, 1.1917946
9: -9.4470148, -7.4084339, -9.4726372, -7.4241886, -1.5943260, 1.6312327

Time for backsubstitution: 5.69 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1928
type: B, layer: 3, pos: 1683
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 1928

## Relational analysis of IS_B1_A1_A2_B2_A2_A1

### Relational analysis result of IS_B1_A1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7844714, upper bound: 0.7871565
time: 4.25 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2_A2

### Relational analysis result of IS_B1_A1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7844714, upper bound: 0.7902950
time: 4.11 seconds

## BFS IS instance: IS_B1_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -7.3346815, -5.1257014, -7.2792053, -5.1870947, -1.8037553, 1.7913382
1: 1.8641927, 3.5431864, 1.9629478, 3.5594928, -1.3430429, 1.2241848
2: -4.9168887, -3.2818151, -4.9413295, -3.3116751, -1.1671541, 1.2452632
3: -11.0240202, -8.8790646, -11.0054293, -8.8976021, -1.6135931, 1.5606503
4: -5.5627766, -3.7504187, -5.5801086, -3.8482392, -1.5318213, 1.6431384
5: -9.0789204, -7.3110199, -9.0406361, -7.3660674, -1.6488123, 1.7296162
6: -6.5447588, -4.4103842, -6.4844942, -4.3451953, -1.6445997, 1.5247912
7: -8.7209301, -7.3990741, -8.7740707, -7.4345551, -1.0072598, 1.2357676
8: 1.0395288, 2.6227388, 1.0256505, 2.5252213, -1.1946187, 1.2082571
9: -9.3749256, -7.2110853, -9.4429226, -7.4291468, -1.5844903, 1.8612080

Time for backsubstitution: 5.71 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1928
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1683
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 1928

## Relational analysis of IS_B1_A2_A1_B1_B1_A1

### Relational analysis result of IS_B1_A2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7782047, upper bound: 0.7869328
time: 4.30 seconds

## Relational analysis of IS_B1_A2_A1_B1_B1_A2

### Relational analysis result of IS_B1_A2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7782047, upper bound: 0.7919013
time: 3.85 seconds

## BFS IS instance: IS_B1_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -7.3346815, -5.1257014, -7.2816219, -5.1853986, -1.8048873, 1.7914999
1: 1.8641927, 3.5431864, 1.9678631, 3.5595744, -1.3394947, 1.2179041
2: -4.9168887, -3.2818151, -4.9423676, -3.3071125, -1.1731584, 1.2469295
3: -11.0240202, -8.8790646, -11.0158348, -8.8992233, -1.6042843, 1.5687814
4: -5.5627766, -3.7504187, -5.5826516, -3.8465366, -1.5336952, 1.6550157
5: -9.0789204, -7.3110199, -9.0521526, -7.3407874, -1.6840539, 1.7411327
6: -6.5447588, -4.4103842, -6.4825869, -4.3305831, -1.6475232, 1.5207911
7: -8.7209301, -7.3990741, -8.8121519, -7.4158998, -1.0293419, 1.2711804
8: 1.0395288, 2.6227388, 1.0121651, 2.5216990, -1.1997066, 1.2370398
9: -9.3749256, -7.2110853, -9.4426537, -7.4280615, -1.5864298, 1.8612287

Time for backsubstitution: 5.78 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1928
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1683
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 1928

## Relational analysis of IS_B1_A2_A1_B1_B2_A1

### Relational analysis result of IS_B1_A2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7782047, upper bound: 0.7869347
time: 4.35 seconds

## Relational analysis of IS_B1_A2_A1_B1_B2_A2

### Relational analysis result of IS_B1_A2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7782047, upper bound: 0.7919033
time: 4.05 seconds

## BFS IS instance: IS_B1_A2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -7.2839231, -5.1881456, -7.2778945, -5.1898537, -1.7492404, 1.7466636
1: 1.8713422, 3.5362015, 1.9638524, 3.5657437, -1.3321753, 1.2127581
2: -4.9193878, -3.2892485, -4.9573951, -3.3005514, -1.1765742, 1.2581375
3: -10.9840431, -8.8952923, -11.0217276, -8.8957863, -1.4764278, 1.5132749
4: -5.5474129, -3.7458131, -5.5864768, -3.8466840, -1.5167413, 1.5985756
5: -9.0432758, -7.3593464, -9.0544624, -7.3295341, -1.6272998, 1.6094933
6: -6.4969215, -4.4063244, -6.4825797, -4.3315344, -1.6043820, 1.5124311
7: -8.7058325, -7.4283772, -8.8149462, -7.4147921, -1.0059557, 1.1030974
8: 1.0403895, 2.5983539, 0.9905672, 2.5312634, -1.1227491, 1.2258935
9: -9.3546400, -7.2338943, -9.4415627, -7.4280930, -1.5575089, 1.8433781

Time for backsubstitution: 5.78 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 655

## Relational analysis of IS_B1_A2_A1_B2_A1_B1

### Relational analysis result of IS_B1_A2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7795343, upper bound: 0.7873909
time: 4.32 seconds

## Relational analysis of IS_B1_A2_A1_B2_A1_B2

### Relational analysis result of IS_B1_A2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7795343, upper bound: 0.7873903
time: 4.17 seconds

## BFS IS instance: IS_B1_A2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -7.3283820, -5.1388683, -7.2778945, -5.1898537, -1.8066673, 1.8098385
1: 1.8686485, 3.5390472, 1.9638524, 3.5657437, -1.3440204, 1.2109339
2: -4.9146671, -3.2907217, -4.9573951, -3.3005514, -1.1760178, 1.2566860
3: -11.0095186, -8.8823729, -11.0217276, -8.8957863, -1.6134148, 1.5697706
4: -5.5377655, -3.7524683, -5.5864768, -3.8466840, -1.5190372, 1.6549673
5: -9.0761871, -7.3248920, -9.0544624, -7.3295341, -1.6652331, 1.7295704
6: -6.5180511, -4.4118495, -6.4825797, -4.3315344, -1.6591580, 1.5165029
7: -8.7193851, -7.4093699, -8.8149462, -7.4147921, -1.0106509, 1.2658679
8: 1.0412526, 2.5966587, 0.9905672, 2.5312634, -1.1994481, 1.2465780
9: -9.3715048, -7.2163372, -9.4415627, -7.4280930, -1.5839014, 1.8480072

Time for backsubstitution: 5.72 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 655
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 655

## Relational analysis of IS_B1_A2_A1_B2_A2_B1

### Relational analysis result of IS_B1_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7795343, upper bound: 0.7919530
time: 3.86 seconds

## Relational analysis of IS_B1_A2_A1_B2_A2_B2

### Relational analysis result of IS_B1_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7795343, upper bound: 0.7919518
time: 4.11 seconds

## BFS IS instance: IS_B1_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -7.3372736, -5.1238489, -7.2792053, -5.1870947, -1.8044538, 1.7902935
1: 1.8708887, 3.5407310, 1.9629478, 3.5594928, -1.3434763, 1.2202382
2: -4.9201188, -3.2771420, -4.9413295, -3.3116751, -1.1688838, 1.2514250
3: -11.0323420, -8.8799314, -11.0054293, -8.8976021, -1.6212435, 1.5630887
4: -5.5635824, -3.7491500, -5.5801086, -3.8482392, -1.5426316, 1.6506081
5: -9.0854607, -7.2852230, -9.0406361, -7.3660674, -1.6709270, 1.7554131
6: -6.5434008, -4.3942294, -6.4844942, -4.3451953, -1.6320331, 1.5115778
7: -8.7496300, -7.3851194, -8.7740707, -7.4345551, -1.0297387, 1.2471571
8: 1.0363531, 2.6108999, 1.0256505, 2.5252213, -1.2119365, 1.2163219
9: -9.3747444, -7.2088923, -9.4429226, -7.4291468, -1.5856071, 1.8636827

Time for backsubstitution: 5.76 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1928
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1683
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 1928

## Relational analysis of IS_B1_A2_A2_B1_B1_A1

### Relational analysis result of IS_B1_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7819896, upper bound: 0.7842618
time: 4.04 seconds

## Relational analysis of IS_B1_A2_A2_B1_B1_A2

### Relational analysis result of IS_B1_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7819896, upper bound: 0.7913879
time: 3.98 seconds

## BFS IS instance: IS_B1_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -7.3359046, -5.1237674, -7.2729120, -5.1916962, -1.8005223, 1.7888088
1: 1.8710923, 3.5407085, 1.9632504, 3.5561004, -1.3506241, 1.2254331
2: -4.9210534, -3.2766500, -4.9532557, -3.3062730, -1.1725681, 1.2689263
3: -11.0323486, -8.8797817, -11.0042868, -8.8969631, -1.6224880, 1.5626421
4: -5.5632019, -3.7491534, -5.5789213, -3.8486433, -1.5419703, 1.6502321
5: -9.0853291, -7.2844553, -9.0424948, -7.3618431, -1.6713452, 1.7580395
6: -6.5425858, -4.3938022, -6.4839888, -4.3594584, -1.6302662, 1.5065691
7: -8.7502832, -7.3851252, -8.7764282, -7.4345636, -1.0304039, 1.2492514
8: 1.0350542, 2.6108465, 1.0184011, 2.5346045, -1.2100027, 1.2234516
9: -9.3747158, -7.2092738, -9.4406939, -7.4308133, -1.5834365, 1.8635345

Time for backsubstitution: 5.75 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1928
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 2832
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 1928

## Relational analysis of IS_B1_A2_A2_B1_B2_A1

### Relational analysis result of IS_B1_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7833256, upper bound: 0.7847126
time: 3.63 seconds

## Relational analysis of IS_B1_A2_A2_B1_B2_A2

### Relational analysis result of IS_B1_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7833256, upper bound: 0.7916869
time: 3.89 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -7.3343773, -5.1244688, -7.2845182, -5.1848984, -1.8200850, 1.8172030
1: 1.8860569, 3.5405769, 1.9535496, 3.5597022, -1.3265319, 1.2430890
2: -4.9149842, -3.2924318, -4.9468861, -3.2916915, -1.1875606, 1.2349350
3: -11.0320492, -8.8838024, -11.0158558, -8.8955364, -1.6197076, 1.5628467
4: -5.5600204, -3.7494016, -5.5860748, -3.8464928, -1.5291357, 1.6497922
5: -9.0840664, -7.2934589, -9.0534096, -7.3332181, -1.6743059, 1.7599506
6: -6.5259919, -4.3965669, -6.4995322, -4.3285933, -1.6487398, 1.5674708
7: -8.7460079, -7.3855929, -8.8153839, -7.4157572, -1.0234857, 1.2594907
8: 1.0484219, 2.6102691, 1.0007749, 2.5223308, -1.2323508, 1.2556509
9: -9.3742914, -7.2133403, -9.4428253, -7.4238605, -1.5977201, 1.8627639

Time for backsubstitution: 5.74 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1928
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1683
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 1928

## Relational analysis of IS_B1_A2_A2_B2_A1_A1

### Relational analysis result of IS_B1_A2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7792888, upper bound: 0.7832871
time: 3.99 seconds

## Relational analysis of IS_B1_A2_A2_B2_A1_A2

### Relational analysis result of IS_B1_A2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7792888, upper bound: 0.7903788
time: 4.02 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -7.3280840, -5.1288605, -7.2831469, -5.1848249, -1.8186059, 1.8134253
1: 1.8840437, 3.5371032, 1.9537413, 3.5596833, -1.3354311, 1.2496550
2: -4.9288845, -3.2871890, -4.9478383, -3.2912049, -1.2060115, 1.2386527
3: -11.0309591, -8.8824968, -11.0158577, -8.8953743, -1.6193161, 1.5642862
4: -5.5588794, -3.7497516, -5.5856948, -3.8464978, -1.5294008, 1.6491890
5: -9.0858688, -7.2881875, -9.0532808, -7.3324533, -1.6803479, 1.7650933
6: -6.5256376, -4.4099197, -6.4987097, -4.3281779, -1.6430712, 1.5653653
7: -8.7498140, -7.3855324, -8.8160429, -7.4157639, -1.0269091, 1.2609429
8: 1.0399542, 2.6197491, 0.9994898, 2.5222774, -1.2419877, 1.2551386
9: -9.3720007, -7.2143488, -9.4428005, -7.4242606, -1.5979707, 1.8613424

Time for backsubstitution: 5.72 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1928
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 1683
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 1928

## Relational analysis of IS_B1_A2_A2_B2_A2_A1

### Relational analysis result of IS_B1_A2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7796681, upper bound: 0.7847117
time: 3.91 seconds

## Relational analysis of IS_B1_A2_A2_B2_A2_A2

### Relational analysis result of IS_B1_A2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7796681, upper bound: 0.7916861
time: 3.85 seconds

## BFS IS instance: IS_B2_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -7.2846489, -5.1848550, -7.3280020, -5.1365242, -1.8130989, 1.8086722
1: 1.9647248, 3.5772171, 1.9475625, 3.5531535, -1.2003465, 1.2580314
2: -4.9530039, -3.3054171, -4.9484043, -3.2978125, -1.1980982, 1.1764728
3: -11.0294476, -8.8962717, -11.0335894, -8.8825102, -1.5666151, 1.6246338
4: -5.6027932, -3.8462076, -5.5689354, -3.8446200, -1.5601697, 1.4853449
5: -9.0551052, -7.3341613, -9.0726986, -7.3242474, -1.7308578, 1.6530304
6: -6.4838457, -4.2982903, -6.5247164, -4.3242731, -1.5601764, 1.6305358
7: -8.8335352, -7.4146147, -8.8126602, -7.4073324, -1.2066019, 1.0274847
8: 0.9885411, 2.5221534, 1.0186758, 2.5286412, -1.1774812, 1.2016664
9: -9.4735899, -7.4267883, -9.4466467, -7.4082885, -1.6371930, 1.5794585

Time for backsubstitution: 5.75 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 1683
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 655

## Relational analysis of IS_B2_A1_B1_B1_A1_A1

### Relational analysis result of IS_B2_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7889447, upper bound: 0.7828520
time: 4.32 seconds

## Relational analysis of IS_B2_A1_B1_B1_A1_A2

### Relational analysis result of IS_B2_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7889447, upper bound: 0.7843868
time: 4.24 seconds

## BFS IS instance: IS_B2_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -7.2783546, -5.1893396, -7.3266349, -5.1364422, -1.8116517, 1.8049004
1: 1.9632540, 3.5738282, 1.9476366, 3.5531363, -1.2077327, 1.2649889
2: -4.9664812, -3.3001566, -4.9493523, -3.2973299, -1.2164323, 1.1802065
3: -11.0283031, -8.8950758, -11.0335913, -8.8823290, -1.5661759, 1.6263309
4: -5.6016359, -3.8466101, -5.5685644, -3.8446236, -1.5598369, 1.4846785
5: -9.0569544, -7.3292274, -9.0725670, -7.3234706, -1.7334838, 1.6542850
6: -6.4833550, -4.3118458, -6.5238981, -4.3238344, -1.5551772, 1.6291375
7: -8.8370733, -7.4146242, -8.8132887, -7.4073405, -1.2098634, 1.0280786
8: 0.9804764, 2.5315232, 1.0174031, 2.5285821, -1.1856816, 1.1997252
9: -9.4713564, -7.4280224, -9.4466248, -7.4086637, -1.6369801, 1.5777833

Time for backsubstitution: 5.76 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 655
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 1683

## Relational analysis of IS_B2_A1_B1_B1_A2_A1

### Relational analysis result of IS_B2_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7893593, upper bound: 0.7859534
time: 4.22 seconds

## Relational analysis of IS_B2_A1_B1_B1_A2_A2

### Relational analysis result of IS_B2_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7893593, upper bound: 0.7859535
time: 4.22 seconds

## BFS IS instance: IS_B2_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -7.2831717, -5.1860595, -7.3317451, -5.1350422, -1.8039279, 1.8005126
1: 1.9461946, 3.5677295, 1.9460990, 3.5563092, -1.2253308, 1.2546031
2: -4.9548240, -3.2925885, -4.9537902, -3.2895765, -1.2077503, 1.1947310
3: -11.0118694, -8.8928661, -11.0418968, -8.8818893, -1.5551167, 1.6463017
4: -5.5981255, -3.8481121, -5.5743446, -3.8427057, -1.5567307, 1.4995561
5: -9.0446062, -7.3571277, -9.0792437, -7.2967434, -1.7478628, 1.6521959
6: -6.5057964, -4.3236704, -6.5288715, -4.3079991, -1.5757046, 1.6088858
7: -8.7990742, -7.4342251, -8.8453779, -7.3960986, -1.1882026, 1.0312974
8: 1.0038567, 2.5262966, 1.0076389, 2.5148697, -1.1497445, 1.2134645
9: -9.4727840, -7.4243088, -9.4472132, -7.4053216, -1.6392703, 1.5869622

Time for backsubstitution: 5.75 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 423

## Relational analysis of IS_B2_A1_B1_B2_A1_A1

### Relational analysis result of IS_B2_A1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7869965, upper bound: 0.7828538
time: 4.37 seconds

## Relational analysis of IS_B2_A1_B1_B2_A1_A2

### Relational analysis result of IS_B2_A1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7874132, upper bound: 0.7844734
time: 4.50 seconds

## BFS IS instance: IS_B2_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -7.2855878, -5.1842670, -7.3317451, -5.1350422, -1.8285918, 1.8185105
1: 1.9496813, 3.5686789, 1.9460990, 3.5563092, -1.2289426, 1.2605023
2: -4.9570961, -3.2880132, -4.9537902, -3.2895765, -1.2079771, 1.1972711
3: -11.0224342, -8.8939648, -11.0418968, -8.8818893, -1.5588756, 1.6406031
4: -5.6020403, -3.8464110, -5.5743446, -3.8427057, -1.5546994, 1.4885867
5: -9.0561609, -7.3312216, -9.0792437, -7.2967434, -1.7594175, 1.6464992
6: -6.5038824, -4.3086281, -6.5288715, -4.3079991, -1.6134911, 1.6418076
7: -8.8383360, -7.4155598, -8.8453779, -7.3960986, -1.1957653, 1.0234568
8: 0.9881859, 2.5227408, 1.0076389, 2.5148697, -1.1898408, 1.2398498
9: -9.4726963, -7.4228458, -9.4472132, -7.4053216, -1.6448720, 1.5957923

Time for backsubstitution: 5.76 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 423

## Relational analysis of IS_B2_A1_B1_B2_A2_A1

### Relational analysis result of IS_B2_A1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7869965, upper bound: 0.7843865
time: 4.15 seconds

## Relational analysis of IS_B2_A1_B1_B2_A2_A2

### Relational analysis result of IS_B2_A1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7874132, upper bound: 0.7859537
time: 4.14 seconds

## BFS IS instance: IS_B2_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -7.2841921, -5.1853704, -7.3297482, -5.1389503, -1.8112860, 1.8104525
1: 1.9653170, 3.5691369, 1.8685980, 3.5390697, -1.2035470, 1.3371682
2: -4.9439034, -3.3058186, -4.9137068, -3.2912123, -1.2382534, 1.1722822
3: -11.0228701, -8.8969622, -11.0095158, -8.8825531, -1.5701952, 1.6117167
4: -5.5876284, -3.8462799, -5.5381451, -3.7524643, -1.6553020, 1.5197005
5: -9.0526304, -7.3344679, -9.0763149, -7.3256583, -1.7269721, 1.6639705
6: -6.4830751, -4.3179798, -6.5188694, -4.4122901, -1.5215440, 1.6604581
7: -8.8114071, -7.4147811, -8.7187586, -7.4093647, -1.2626066, 1.0100570
8: 0.9986320, 2.5218811, 1.0425243, 2.5967135, -1.2385802, 1.2014041
9: -9.4438000, -7.4268608, -9.3715334, -7.2159615, -1.8481853, 1.5855825

Time for backsubstitution: 5.77 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 655
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2832
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 655

## Relational analysis of IS_B2_A1_B2_B1_A1_A1

### Relational analysis result of IS_B2_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7919014, upper bound: 0.7782054
time: 4.13 seconds

## Relational analysis of IS_B2_A1_B2_B1_A1_A2

### Relational analysis result of IS_B2_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7919014, upper bound: 0.7793546
time: 3.93 seconds

## BFS IS instance: IS_B2_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -7.2778945, -5.1898537, -7.3283820, -5.1388683, -1.8098383, 1.8066673
1: 1.9638524, 3.5657437, 1.8686485, 3.5390472, -1.2109342, 1.3440206
2: -4.9573951, -3.3005514, -4.9146671, -3.2907217, -1.2566860, 1.1760178
3: -11.0217276, -8.8957863, -11.0095186, -8.8823729, -1.5697708, 1.6134148
4: -5.5864768, -3.8466840, -5.5377655, -3.7524683, -1.6549673, 1.5190372
5: -9.0544624, -7.3295341, -9.0761871, -7.3248920, -1.7295704, 1.6652327
6: -6.4825797, -4.3315344, -6.5180511, -4.4118495, -1.5165029, 1.6591580
7: -8.8149462, -7.4147921, -8.7193851, -7.4093699, -1.2658682, 1.0106506
8: 0.9905672, 2.5312634, 1.0412526, 2.5966587, -1.2465780, 1.1994481
9: -9.4415627, -7.4280930, -9.3715048, -7.2163372, -1.8480070, 1.5839014

Time for backsubstitution: 5.77 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 655
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 1683
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2832
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 655

## Relational analysis of IS_B2_A1_B2_B1_A2_A1

### Relational analysis result of IS_B2_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7919523, upper bound: 0.7795350
time: 4.12 seconds

## Relational analysis of IS_B2_A1_B2_B1_A2_A2

### Relational analysis result of IS_B2_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7919523, upper bound: 0.7808038
time: 4.03 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 14.13 seconds
IS_B1_A1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 14.13
Output dim: 1, lower bound: -0.7828519, upper bound: 0.7858038
IS_B1_A1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 14.13
Output dim: 1, lower bound: -0.7828519, upper bound: 0.7889457
IS_B1_A1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 14.13
Output dim: 1, lower bound: -0.7828519, upper bound: 0.7858036
IS_B1_A1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 14.13
Output dim: 1, lower bound: -0.7828519, upper bound: 0.7889467
IS_B1_A1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 14.13
Output dim: 1, lower bound: -0.7859529, upper bound: 0.7866613
IS_B1_A1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 14.13
Output dim: 1, lower bound: -0.7859529, upper bound: 0.7866612
IS_B1_A1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 14.13
Output dim: 1, lower bound: -0.7859529, upper bound: 0.7893596
IS_B1_A1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 14.13
Output dim: 1, lower bound: -0.7859529, upper bound: 0.7893597
IS_B1_A1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 14.13
Output dim: 1, lower bound: -0.7877630, upper bound: 0.7861271
IS_B1_A1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 14.13
Output dim: 1, lower bound: -0.7877630, upper bound: 0.7896755
IS_B1_A1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 14.13
Output dim: 1, lower bound: -0.7891247, upper bound: 0.7871565
IS_B1_A1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 14.13
Output dim: 1, lower bound: -0.7891247, upper bound: 0.7902947
IS_B1_A1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 14.13
Output dim: 1, lower bound: -0.7838293, upper bound: 0.7853033
IS_B1_A1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 14.13
Output dim: 1, lower bound: -0.7838293, upper bound: 0.7886813
IS_B1_A1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 14.13
Output dim: 1, lower bound: -0.7844714, upper bound: 0.7871565
IS_B1_A1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 14.13
Output dim: 1, lower bound: -0.7844714, upper bound: 0.7902950
IS_B1_A2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 14.13
Output dim: 1, lower bound: -0.7782047, upper bound: 0.7869328
IS_B1_A2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 14.13
Output dim: 1, lower bound: -0.7782047, upper bound: 0.7919013
IS_B1_A2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 14.13
Output dim: 1, lower bound: -0.7782047, upper bound: 0.7869347
IS_B1_A2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 14.13
Output dim: 1, lower bound: -0.7782047, upper bound: 0.7919033
IS_B1_A2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 14.13
Output dim: 1, lower bound: -0.7795343, upper bound: 0.7873909
IS_B1_A2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 14.13
Output dim: 1, lower bound: -0.7795343, upper bound: 0.7873903
IS_B1_A2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 14.13
Output dim: 1, lower bound: -0.7795343, upper bound: 0.7919530
IS_B1_A2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 14.13
Output dim: 1, lower bound: -0.7795343, upper bound: 0.7919518
IS_B1_A2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 14.13
Output dim: 1, lower bound: -0.7819896, upper bound: 0.7842618
IS_B1_A2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 14.13
Output dim: 1, lower bound: -0.7819896, upper bound: 0.7913879
IS_B1_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 14.13
Output dim: 1, lower bound: -0.7833256, upper bound: 0.7847126
IS_B1_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 14.13
Output dim: 1, lower bound: -0.7833256, upper bound: 0.7916869
IS_B1_A2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 14.13
Output dim: 1, lower bound: -0.7792888, upper bound: 0.7832871
IS_B1_A2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 14.13
Output dim: 1, lower bound: -0.7792888, upper bound: 0.7903788
IS_B1_A2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 14.13
Output dim: 1, lower bound: -0.7796681, upper bound: 0.7847117
IS_B1_A2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 14.13
Output dim: 1, lower bound: -0.7796681, upper bound: 0.7916861
IS_B2_A1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 14.13
Output dim: 1, lower bound: -0.7889447, upper bound: 0.7828520
IS_B2_A1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 14.13
Output dim: 1, lower bound: -0.7889447, upper bound: 0.7843868
IS_B2_A1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 14.13
Output dim: 1, lower bound: -0.7893593, upper bound: 0.7859534
IS_B2_A1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 14.13
Output dim: 1, lower bound: -0.7893593, upper bound: 0.7859535
IS_B2_A1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 14.13
Output dim: 1, lower bound: -0.7869965, upper bound: 0.7828538
IS_B2_A1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 14.13
Output dim: 1, lower bound: -0.7874132, upper bound: 0.7844734
IS_B2_A1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 14.13
Output dim: 1, lower bound: -0.7869965, upper bound: 0.7843865
IS_B2_A1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 14.13
Output dim: 1, lower bound: -0.7874132, upper bound: 0.7859537
IS_B2_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 14.13
Output dim: 1, lower bound: -0.7919014, upper bound: 0.7782054
IS_B2_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 14.13
Output dim: 1, lower bound: -0.7919014, upper bound: 0.7793546
IS_B2_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 14.13
Output dim: 1, lower bound: -0.7919523, upper bound: 0.7795350
IS_B2_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 14.13
Output dim: 1, lower bound: -0.7919523, upper bound: 0.7808038
IS_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.13
Output dim: 1, lower bound: -0.8009716, upper bound: 0.7888272
IS_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.13
Output dim: 1, lower bound: -0.8009716, upper bound: 0.7900857
IS_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 14.13
Output dim: 1, lower bound: -0.7781062, upper bound: 0.7967099
IS_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 14.13
Output dim: 1, lower bound: -0.7798083, upper bound: 0.7970947
IS_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 14.13
Output dim: 1, lower bound: -0.7875273, upper bound: 0.8046895
IS_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 14.13
Output dim: 1, lower bound: -0.7875273, upper bound: 0.8046902
IS_B2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.13
Output dim: 1, lower bound: -0.7844132, upper bound: 0.7908007
IS_B2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.13
Output dim: 1, lower bound: -0.7843819, upper bound: 0.7919014
IS_B2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.13
Output dim: 1, lower bound: -0.7916957, upper bound: 0.7983141
IS_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.13
Output dim: 1, lower bound: -0.7916957, upper bound: 0.8011917
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
type: A, layer: 3, pos: 1928
type: B, layer: 3, pos: 1928
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 655
type: B, layer: 3, pos: 655
type: A, layer: 3, pos: 1683
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.40 seconds

### Candidate
type: A, layer: 3, pos: 1928

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7610339, upper bound: 0.7516602
time: 4.48 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7610339, upper bound: 0.7610360
time: 4.67 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 9.56 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 9.56
Output dim: 1, lower bound: -0.7610339, upper bound: 0.7516602
IS_A2, status: Status.UNKNOWN, split count: 1, time: 9.56
Output dim: 1, lower bound: -0.7610339, upper bound: 0.7610360

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -7.2890615, -5.1841650, -7.3393188, -5.1294498, -1.6966219, 1.7153702
1: 1.9458289, 3.5798233, 1.9379888, 3.5862699, -1.2037451, 1.2156601
2: -4.9594469, -3.2859809, -4.9607730, -3.2786198, -1.1525898, 1.1458054
3: -11.0325260, -8.8911295, -11.0674467, -8.8750515, -1.4746614, 1.5452528
4: -5.6086173, -3.8460803, -5.6241131, -3.8406239, -1.4985733, 1.4650264
5: -9.0567703, -7.3194046, -9.0862675, -7.2751775, -1.7524257, 1.6359439
6: -6.5046525, -4.2907782, -6.5497088, -4.2860427, -1.5178494, 1.5486357
7: -8.8444023, -7.4142337, -8.8567429, -7.3863902, -1.1752546, 1.0193634
8: 0.9678707, 2.5229740, 0.9694920, 2.5408931, -1.1376500, 1.1897023
9: -9.4739857, -7.4211969, -9.4915495, -7.4032288, -1.5565343, 1.5626278

Time for backsubstitution: 5.72 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 655
type: A, layer: 3, pos: 655
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 1683
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 655

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7495255, upper bound: 0.7387058
time: 4.25 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7490377, upper bound: 0.7387074
time: 4.27 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -7.3349719, -5.1344824, -7.3389273, -5.1125546, -1.7822762, 1.6732507
1: 1.9418974, 3.5842206, 1.9381919, 3.5875144, -1.2171316, 1.2477190
2: -4.9584117, -3.2864370, -4.9609857, -3.2790527, -1.1522651, 1.1463878
3: -11.0543442, -8.8779135, -11.0713587, -8.8751745, -1.5921278, 1.6194806
4: -5.6055589, -3.8417616, -5.6229496, -3.8402016, -1.5058966, 1.5137720
5: -9.0845490, -7.2844133, -9.0868454, -7.2667475, -1.8178015, 1.7376313
6: -6.5309668, -4.2874689, -6.5536671, -4.2860031, -1.5526075, 1.5799589
7: -8.8546734, -7.3944950, -8.8566122, -7.3828506, -1.1811280, 1.1661313
8: 0.9718418, 2.5160675, 0.9693451, 2.5366940, -1.1972260, 1.2206318
9: -9.4891787, -7.4034848, -9.4915638, -7.3970466, -1.5885048, 1.5823307

Time for backsubstitution: 5.69 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 1683
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 2832
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 3, pos: 655

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7495232, upper bound: 0.7490379
time: 4.45 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7490377, upper bound: 0.7490387
time: 4.78 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 15.13 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 15.13
Output dim: 1, lower bound: -0.7495255, upper bound: 0.7387058
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 15.13
Output dim: 1, lower bound: -0.7490377, upper bound: 0.7387074
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 15.13
Output dim: 1, lower bound: -0.7495232, upper bound: 0.7490379
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 15.13
Output dim: 1, lower bound: -0.7490377, upper bound: 0.7490387

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -7.2875981, -5.1842556, -7.3335228, -5.1314402, -1.6856966, 1.6996810
1: 1.9475536, 3.5763195, 1.9394684, 3.5735989, -1.1942477, 1.2095923
2: -4.9584646, -3.2871718, -4.9541702, -3.2853680, -1.1435432, 1.1400683
3: -11.0275803, -8.8920231, -11.0476074, -8.8770313, -1.4668202, 1.5260744
4: -5.6058760, -3.8461983, -5.6127901, -3.8430142, -1.4877353, 1.4477618
5: -9.0565567, -7.3281698, -9.0792789, -7.3130322, -1.7093530, 1.6022034
6: -6.5042028, -4.2988529, -6.5526276, -4.3203225, -1.4974127, 1.5497921
7: -8.8335810, -7.4145727, -8.8178692, -7.4035897, -1.1442266, 0.9908900
8: 0.9786372, 2.5228877, 0.9982953, 2.5476785, -1.1127794, 1.1516929
9: -9.4736881, -7.4218159, -9.4899902, -7.4067116, -1.5514596, 1.5560451

Time for backsubstitution: 5.70 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 1683
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 3, pos: 423

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7411704, upper bound: 0.7303591
time: 4.20 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7418940, upper bound: 0.7310698
time: 4.70 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -7.2886848, -5.1841755, -7.3360443, -5.1295762, -1.6853170, 1.7113602
1: 1.9462442, 3.5786259, 1.9417517, 3.5750635, -1.2007284, 1.2105210
2: -4.9591951, -3.2862043, -4.9584131, -3.2806931, -1.1465566, 1.1441221
3: -11.0314474, -8.8914347, -11.0571156, -8.8778381, -1.4717994, 1.5377116
4: -5.6077881, -3.8461154, -5.6175294, -3.8411427, -1.4975882, 1.4488192
5: -9.0567055, -7.3206682, -9.0856504, -7.2872386, -1.6985517, 1.6343164
6: -6.5045700, -4.2928648, -6.5486178, -4.3047538, -1.5223579, 1.5436521
7: -8.8437519, -7.4143763, -8.8493214, -7.3880291, -1.1738980, 0.9832489
8: 0.9704714, 2.5229492, 0.9907446, 2.5406594, -1.1308987, 1.1816730
9: -9.4738474, -7.4213772, -9.4902372, -7.4046836, -1.5539851, 1.5643306

Time for backsubstitution: 5.70 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 655
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 1928
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 2832
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 423

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7402883, upper bound: 0.7310521
time: 4.61 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7413975, upper bound: 0.7310697
time: 4.90 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -7.3336191, -5.1345882, -7.3331990, -5.1140375, -1.7756495, 1.6581821
1: 1.9436135, 3.5792115, 1.9396214, 3.5743942, -1.2073393, 1.2496920
2: -4.9574251, -3.2876773, -4.9543753, -3.2857461, -1.1432705, 1.1411066
3: -11.0497227, -8.8788223, -11.0517473, -8.8771696, -1.5891633, 1.6000447
4: -5.6028628, -3.8419204, -5.6116428, -3.8425837, -1.4978504, 1.4955299
5: -9.0843277, -7.2932420, -9.0798607, -7.3045611, -1.7787833, 1.7028403
6: -6.5304608, -4.2958465, -6.5566254, -4.3202782, -1.5328488, 1.5862844
7: -8.8443413, -7.3949065, -8.8178368, -7.3974042, -1.1528320, 1.1250803
8: 0.9824362, 2.5159755, 0.9981627, 2.5465207, -1.1808581, 1.1735671
9: -9.4888687, -7.4041996, -9.4899797, -7.4005394, -1.5836051, 1.5757504

Time for backsubstitution: 5.69 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 1683
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 655
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 423

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7411682, upper bound: 0.7402877
time: 4.17 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7418940, upper bound: 0.7413985
time: 4.78 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -7.3346052, -5.1344972, -7.3360071, -5.1126909, -1.7751660, 1.6747279
1: 1.9423087, 3.5829792, 1.9417739, 3.5762811, -1.2129958, 1.2429647
2: -4.9581556, -3.2866566, -4.9586234, -3.2809267, -1.1462250, 1.1446698
3: -11.0531273, -8.8782206, -11.0604248, -8.8777523, -1.5890121, 1.6071281
4: -5.6047125, -3.8418171, -5.6163845, -3.8407238, -1.5049129, 1.4947147
5: -9.0844812, -7.2857070, -9.0862284, -7.2788048, -1.7677898, 1.7357292
6: -6.5308480, -4.2894716, -6.5525799, -4.3046265, -1.5613995, 1.5736184
7: -8.8538771, -7.3946762, -8.8491888, -7.3842545, -1.1796756, 1.1214445
8: 0.9747910, 2.5160422, 0.9904752, 2.5364513, -1.1895955, 1.1921575
9: -9.4890308, -7.4036407, -9.4904556, -7.3985057, -1.5857530, 1.5841839

Time for backsubstitution: 5.73 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 1683
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 423

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7402883, upper bound: 0.7407392
time: 4.53 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7413975, upper bound: 0.7413996
time: 4.72 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 15.19 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 15.19
Output dim: 1, lower bound: -0.7411704, upper bound: 0.7303591
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 15.19
Output dim: 1, lower bound: -0.7418940, upper bound: 0.7310698
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 15.19
Output dim: 1, lower bound: -0.7402883, upper bound: 0.7310521
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 15.19
Output dim: 1, lower bound: -0.7413975, upper bound: 0.7310697
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 15.19
Output dim: 1, lower bound: -0.7411682, upper bound: 0.7402877
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 15.19
Output dim: 1, lower bound: -0.7418940, upper bound: 0.7413985
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 15.19
Output dim: 1, lower bound: -0.7402883, upper bound: 0.7407392
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 15.19
Output dim: 1, lower bound: -0.7413975, upper bound: 0.7413996

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -7.2840929, -5.1848755, -7.3325319, -5.1316557, -1.6819196, 1.6979239
1: 1.9651608, 3.5761659, 1.9449863, 3.5735531, -1.1703219, 1.2000048
2: -4.9527974, -3.3058679, -4.9523430, -3.2906866, -1.1337802, 1.1169415
3: -11.0275545, -8.8965683, -11.0475044, -8.8784389, -1.4654222, 1.5201135
4: -5.6017675, -3.8462534, -5.6115947, -3.8431149, -1.4829521, 1.4442837
5: -9.0550232, -7.3374333, -9.0788174, -7.3158984, -1.7010865, 1.5880246
6: -6.4836740, -4.3013015, -6.5466614, -4.3211660, -1.4705162, 1.5364320
7: -8.8295212, -7.4147453, -8.8166056, -7.4037371, -1.1389985, 0.9893403
8: 0.9924798, 2.5221205, 1.0024281, 2.5474486, -1.1031842, 1.1472173
9: -9.4734774, -7.4269762, -9.4898510, -7.4082537, -1.5468490, 1.5462151

Time for backsubstitution: 5.74 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 655
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 1928
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 1683

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7039806, upper bound: 0.7002201
time: 4.21 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7104348, upper bound: 0.6998785
time: 4.74 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -7.2777987, -5.1893730, -7.3305602, -5.1315184, -1.6805744, 1.6936173
1: 1.9639156, 3.5727756, 1.9446759, 3.5735273, -1.1775432, 1.2090945
2: -4.9661064, -3.3006127, -4.9539061, -3.2895508, -1.1535177, 1.1202725
3: -11.0264130, -8.8954210, -11.0475197, -8.8780613, -1.4652071, 1.5218148
4: -5.6006088, -3.8466563, -5.6111236, -3.8431096, -1.4827332, 1.4435046
5: -9.0568752, -7.3325777, -9.0786591, -7.3145170, -1.7088041, 1.5887456
6: -6.4831820, -4.3149338, -6.5458965, -4.3204436, -1.4669890, 1.5361891
7: -8.8329363, -7.4147539, -8.8176384, -7.4037375, -1.1420803, 0.9903388
8: 0.9844942, 2.5314889, 1.0002031, 2.5473852, -1.1101456, 1.1451213
9: -9.4712429, -7.4282565, -9.4898243, -7.4087157, -1.5473251, 1.5451035

Time for backsubstitution: 5.70 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 1683
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 1683

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7052742, upper bound: 0.7021808
time: 5.29 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7116052, upper bound: 0.7018387
time: 3.96 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -7.2876892, -5.1843677, -7.3325405, -5.1303186, -1.6834974, 1.7077866
1: 1.9515569, 3.5785811, 1.9600472, 3.5749040, -1.1922169, 1.1845589
2: -4.9573774, -3.2915156, -4.9523745, -3.2994268, -1.1233816, 1.1342762
3: -11.0314388, -8.8928337, -11.0567560, -8.8826084, -1.4673061, 1.5346570
4: -5.6065655, -3.8461313, -5.6133208, -3.8414922, -1.4945784, 1.4438679
5: -9.0562553, -7.3234110, -9.0840321, -7.2971902, -1.6820488, 1.6273322
6: -6.4987555, -4.2936511, -6.5275335, -4.3075805, -1.5104270, 1.5168858
7: -8.8424187, -7.4144249, -8.8449373, -7.3885818, -1.1715431, 0.9789498
8: 0.9745145, 2.5227265, 1.0053530, 2.5398874, -1.1270988, 1.1717212
9: -9.4737883, -7.4229097, -9.4897480, -7.4100556, -1.5440857, 1.5597954

Time for backsubstitution: 5.69 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 655
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1928
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 2832
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 1683

## Relational analysis of IS_A1_B2_B1_B1

### Relational analysis result of IS_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7035403, upper bound: 0.7010353
time: 4.14 seconds

## Relational analysis of IS_A1_B2_B1_B2

### Relational analysis result of IS_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7085879, upper bound: 0.7007225
time: 4.05 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -7.2857161, -5.1842437, -7.3262486, -5.1347194, -1.6792083, 1.7063293
1: 1.9514351, 3.5785573, 1.9581563, 3.5714524, -1.2006664, 1.1929417
2: -4.9589376, -3.2903762, -4.9660559, -3.2942176, -1.1267269, 1.1540422
3: -11.0314407, -8.8924837, -11.0556717, -8.8811970, -1.4685984, 1.5344272
4: -5.6060896, -3.8461382, -5.6121726, -3.8418148, -1.4938908, 1.4441676
5: -9.0561008, -7.3220491, -9.0858755, -7.2919693, -1.6825690, 1.6343598
6: -6.4979773, -4.2929778, -6.5272384, -4.3208852, -1.5097561, 1.5129604
7: -8.8435087, -7.4144306, -8.8486805, -7.3885422, -1.1734772, 0.9822819
8: 0.9722877, 2.5226622, 0.9969220, 2.5492597, -1.1265070, 1.1799953
9: -9.4737549, -7.4233875, -9.4875298, -7.4111185, -1.5432162, 1.5610952

Time for backsubstitution: 5.68 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 655
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 2832
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 1683

## Relational analysis of IS_A1_B2_B2_B1

### Relational analysis result of IS_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7056076, upper bound: 0.7021798
time: 3.90 seconds

## Relational analysis of IS_A1_B2_B2_B2

### Relational analysis result of IS_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7106007, upper bound: 0.7018395
time: 4.14 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -7.3301134, -5.1352873, -7.3322072, -5.1142530, -1.7718821, 1.6564002
1: 1.9613407, 3.5790586, 1.9451399, 3.5743499, -1.1823175, 1.2401962
2: -4.9518089, -3.3064027, -4.9525485, -3.2910638, -1.1336503, 1.1179733
3: -11.0494022, -8.8834324, -11.0516415, -8.8785744, -1.5861630, 1.5939708
4: -5.5987568, -3.8422747, -5.6104479, -3.8426867, -1.4930663, 1.4924748
5: -9.0827150, -7.3030348, -9.0794010, -7.3074303, -1.7706060, 1.6864972
6: -6.5093889, -4.2984815, -6.5506616, -4.3211207, -1.5061178, 1.5729105
7: -8.8402500, -7.3954601, -8.8165712, -7.3975482, -1.1475861, 1.1227393
8: 0.9967895, 2.5151615, 1.0022974, 2.5462871, -1.1709266, 1.1690547
9: -9.4883823, -7.4094501, -9.4898396, -7.4020934, -1.5791433, 1.5660434

Time for backsubstitution: 5.76 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1683
type: B, layer: 3, pos: 1683
type: B, layer: 3, pos: 1928
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 1683

## Relational analysis of IS_A2_B1_A1_A1

### Relational analysis result of IS_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7115229, upper bound: 0.7035474
time: 3.83 seconds

## Relational analysis of IS_A2_B1_A1_A2

### Relational analysis result of IS_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7107885, upper bound: 0.7087549
time: 3.93 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -7.3238225, -5.1397190, -7.3302355, -5.1141143, -1.7705274, 1.6520948
1: 1.9600258, 3.5756383, 1.9448309, 3.5743251, -1.1900551, 1.2491949
2: -4.9650745, -3.3011277, -4.9541121, -3.2899270, -1.1532500, 1.1213014
3: -11.0483112, -8.8821764, -11.0516567, -8.8781986, -1.5859141, 1.5957751
4: -5.5975990, -3.8425913, -5.6099782, -3.8426838, -1.4928479, 1.4917974
5: -9.0845661, -7.2980061, -9.0792437, -7.3060465, -1.7783313, 1.6868510
6: -6.5090842, -4.3119631, -6.5498962, -4.3204012, -1.5021610, 1.5726771
7: -8.8436985, -7.3954229, -8.8176041, -7.3975487, -1.1506958, 1.1246367
8: 0.9886131, 2.5245161, 1.0000739, 2.5462208, -1.1791303, 1.1669953
9: -9.4861822, -7.4106340, -9.4898167, -7.4025421, -1.5804288, 1.5648210

Time for backsubstitution: 5.73 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1683
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 655
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 1683

## Relational analysis of IS_A2_B1_A2_A1

### Relational analysis result of IS_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7126207, upper bound: 0.7056105
time: 3.78 seconds

## Relational analysis of IS_A2_B1_A2_A2

### Relational analysis result of IS_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7118708, upper bound: 0.7106705
time: 3.90 seconds

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: -7.3336139, -5.1347108, -7.3325033, -5.1134238, -1.7733483, 1.6709604
1: 1.9476371, 3.5829365, 1.9600971, 3.5761228, -1.2034285, 1.2171338
2: -4.9563713, -3.2919779, -4.9525690, -3.2996616, -1.1231000, 1.1348110
3: -11.0530367, -8.8796329, -11.0600338, -8.8825283, -1.5827832, 1.6040983
4: -5.6034918, -3.8419175, -5.6121769, -3.8410828, -1.5018945, 1.4898567
5: -9.0840139, -7.2885914, -9.0846071, -7.2887778, -1.7512999, 1.7274890
6: -6.5248833, -4.2903113, -6.5315351, -4.3074617, -1.5478935, 1.5468698
7: -8.8525448, -7.3948326, -8.8447914, -7.3847985, -1.1773233, 1.1160014
8: 0.9789696, 2.5158052, 1.0051031, 2.5356479, -1.1851120, 1.1820815
9: -9.4888926, -7.4051991, -9.4899597, -7.4038849, -1.5759146, 1.5796885

Time for backsubstitution: 5.70 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 655
type: B, layer: 3, pos: 1683
type: B, layer: 3, pos: 1928
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 1683

## Relational analysis of IS_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7087549, upper bound: 0.7042821
time: 4.09 seconds

## Relational analysis of IS_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7087549, upper bound: 0.7096442
time: 3.96 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: -7.3316393, -5.1345720, -7.3262105, -5.1178179, -1.7691345, 1.6696143
1: 1.9475203, 3.5829134, 1.9581842, 3.5726702, -1.2124820, 1.2254093
2: -4.9579000, -3.2908411, -4.9662676, -3.2944431, -1.1264048, 1.1545775
3: -11.0530472, -8.8792553, -11.0589514, -8.8811102, -1.5847492, 1.6038523
4: -5.6030169, -3.8419139, -5.6110268, -3.8414054, -1.5012088, 1.4896431
5: -9.0838528, -7.2872038, -9.0864468, -7.2835436, -1.7518458, 1.7352223
6: -6.5241189, -4.2895908, -6.5312185, -4.3207631, -1.5477347, 1.5429420
7: -8.8536329, -7.3948336, -8.8485470, -7.3847589, -1.1792564, 1.1193008
8: 0.9767070, 2.5157380, 0.9966650, 2.5450163, -1.1830637, 1.1905336
9: -9.4888668, -7.4056492, -9.4877396, -7.4049387, -1.5748372, 1.5809345

Time for backsubstitution: 5.72 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1683
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 1683

## Relational analysis of IS_A2_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7106702, upper bound: 0.7056105
time: 3.95 seconds

## Relational analysis of IS_A2_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7106702, upper bound: 0.7106713
time: 3.90 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 13.78 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 13.78
Output dim: 1, lower bound: -0.7039806, upper bound: 0.7002201
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 13.78
Output dim: 1, lower bound: -0.7104348, upper bound: 0.6998785
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 13.78
Output dim: 1, lower bound: -0.7052742, upper bound: 0.7021808
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 13.78
Output dim: 1, lower bound: -0.7116052, upper bound: 0.7018387
IS_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 13.78
Output dim: 1, lower bound: -0.7035403, upper bound: 0.7010353
IS_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 13.78
Output dim: 1, lower bound: -0.7085879, upper bound: 0.7007225
IS_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 13.78
Output dim: 1, lower bound: -0.7056076, upper bound: 0.7021798
IS_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 13.78
Output dim: 1, lower bound: -0.7106007, upper bound: 0.7018395
IS_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 13.78
Output dim: 1, lower bound: -0.7115229, upper bound: 0.7035474
IS_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 13.78
Output dim: 1, lower bound: -0.7107885, upper bound: 0.7087549
IS_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 13.78
Output dim: 1, lower bound: -0.7126207, upper bound: 0.7056105
IS_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 13.78
Output dim: 1, lower bound: -0.7118708, upper bound: 0.7106705
IS_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 13.78
Output dim: 1, lower bound: -0.7087549, upper bound: 0.7042821
IS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 13.78
Output dim: 1, lower bound: -0.7087549, upper bound: 0.7096442
IS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 13.78
Output dim: 1, lower bound: -0.7106702, upper bound: 0.7056105
IS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 13.78
Output dim: 1, lower bound: -0.7106702, upper bound: 0.7106713

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -7.2840929, -5.1848755, -7.3317785, -5.1320372, -1.6812601, 1.6972353
1: 1.9651608, 3.5761659, 1.9458365, 3.5560102, -1.1518850, 1.1992104
2: -4.9527974, -3.3058679, -4.9500904, -3.2922132, -1.1327221, 1.1074684
3: -11.0275545, -8.8965683, -11.0465784, -8.8800745, -1.4641590, 1.5180445
4: -5.6017675, -3.8462534, -5.5870266, -3.8435473, -1.4808440, 1.4192288
5: -9.0550232, -7.3374333, -9.0742741, -7.3161387, -1.6992788, 1.5826578
6: -6.4836740, -4.3013015, -6.5456443, -4.3231993, -1.4675756, 1.5354817
7: -8.8295212, -7.4147453, -8.8140860, -7.4042459, -1.1379328, 0.9797843
8: 0.9924798, 2.5221205, 1.0179391, 2.5465055, -1.1022613, 1.1361210
9: -9.4734774, -7.4269762, -9.4491062, -7.4086318, -1.5464571, 1.4964488

Time for backsubstitution: 5.75 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 655
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 1928
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 1683

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7039644, upper bound: 0.6945510
time: 4.53 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7039644, upper bound: 0.6998787
time: 4.17 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -7.2835102, -5.1855316, -7.3336687, -5.1346874, -1.6801004, 1.6990490
1: 1.9659145, 3.5658774, 1.8669190, 3.5422933, -1.1527953, 1.2785263
2: -4.9412832, -3.3063796, -4.9157066, -3.2848377, -1.1742442, 1.1044924
3: -11.0191870, -8.8974409, -11.0194988, -8.8801069, -1.4671612, 1.5048733
4: -5.5824647, -3.8463466, -5.5604291, -3.7508633, -1.5726881, 1.4486136
5: -9.0518799, -7.3378248, -9.0781479, -7.3176003, -1.6960173, 1.5935574
6: -6.4826941, -4.3263674, -6.5372138, -4.4109869, -1.4314923, 1.5543499
7: -8.8013496, -7.4149580, -8.7201862, -7.4021006, -1.1970971, 0.9669411
8: 1.0053258, 2.5217729, 1.0415802, 2.6194949, -1.1651362, 1.1347845
9: -9.4358206, -7.4270668, -9.3743877, -7.2146001, -1.7742531, 1.5010207

Time for backsubstitution: 5.73 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 655
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 1683
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 2832
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 655

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7104348, upper bound: 0.6988315
time: 4.80 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7104348, upper bound: 0.6998785
time: 4.87 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -7.2777987, -5.1893730, -7.3298063, -5.1318979, -1.6799121, 1.6929250
1: 1.9639156, 3.5727756, 1.9455245, 3.5559835, -1.1591058, 1.2083030
2: -4.9661064, -3.3006127, -4.9516521, -3.2910843, -1.1524596, 1.1108000
3: -11.0264130, -8.8954210, -11.0465927, -8.8796997, -1.4639421, 1.5197468
4: -5.6006088, -3.8466563, -5.5865655, -3.8435431, -1.4806247, 1.4184616
5: -9.0568752, -7.3325777, -9.0741138, -7.3147559, -1.7069983, 1.5833888
6: -6.4831820, -4.3149338, -6.5448809, -4.3224792, -1.4640484, 1.5352571
7: -8.8329363, -7.4147539, -8.8151188, -7.4042463, -1.1410139, 0.9807746
8: 0.9844942, 2.5314889, 1.0157156, 2.5464420, -1.1092203, 1.1340871
9: -9.4712429, -7.4282565, -9.4490795, -7.4090943, -1.5469432, 1.4953372

Time for backsubstitution: 5.70 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1683
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 2832
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 1683

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7052742, upper bound: 0.6964164
time: 5.50 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7052742, upper bound: 0.7018404
time: 4.75 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -7.2772150, -5.1900306, -7.3316975, -5.1345472, -1.6787505, 1.6947558
1: 1.9646740, 3.5624843, 1.8665662, 3.5422630, -1.1600173, 1.2875428
2: -4.9545622, -3.3011160, -4.9172902, -3.2836919, -1.1940737, 1.1078547
3: -11.0180426, -8.8963242, -11.0195112, -8.8797293, -1.4669728, 1.5065765
4: -5.5813127, -3.8467493, -5.5599594, -3.7508631, -1.5724511, 1.4478550
5: -9.0537090, -7.3329678, -9.0779896, -7.3162155, -1.7037354, 1.5942945
6: -6.4821978, -4.3400016, -6.5364556, -4.4102626, -1.4278908, 1.5542274
7: -8.8047676, -7.4149671, -8.7212191, -7.4020967, -1.2001777, 0.9679310
8: 0.9973407, 2.5311608, 1.0393596, 2.6194315, -1.1718946, 1.1327004
9: -9.4335823, -7.4283481, -9.3743563, -7.2150412, -1.7747893, 1.4998991

Time for backsubstitution: 5.77 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 1683
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 2832
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 1928

## Relational analysis of IS_A1_B1_A2_B2_B1

### Relational analysis result of IS_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7073261, upper bound: 0.7018386
time: 4.27 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2

### Relational analysis result of IS_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7073261, upper bound: 0.7018380
time: 4.61 seconds

## BFS IS instance: IS_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -7.2876892, -5.1843677, -7.3317862, -5.1306973, -1.6828074, 1.7071011
1: 1.9515569, 3.5785811, 1.9608703, 3.5580935, -1.1737511, 1.1837606
2: -4.9573774, -3.2915156, -4.9501200, -3.3009708, -1.1222892, 1.1251774
3: -11.0314388, -8.8928337, -11.0559683, -8.8842592, -1.4661140, 1.5325885
4: -5.6065655, -3.8461313, -5.5887079, -3.8419247, -1.4924722, 1.4182053
5: -9.0562553, -7.3234110, -9.0793400, -7.2974324, -1.6802454, 1.6219721
6: -6.4987555, -4.2936511, -6.5265293, -4.3096161, -1.5074863, 1.5159373
7: -8.8424187, -7.4144249, -8.8430796, -7.3891039, -1.1703453, 0.9694786
8: 0.9745145, 2.5227265, 1.0201669, 2.5389609, -1.1262531, 1.1602633
9: -9.4737883, -7.4229097, -9.4487000, -7.4104295, -1.5437036, 1.5103562

Time for backsubstitution: 5.76 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 1683
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1928
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 2832
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 655

## Relational analysis of IS_A1_B2_B1_B1_A1

### Relational analysis result of IS_A1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7013528, upper bound: 0.6998319
time: 4.51 seconds

## Relational analysis of IS_A1_B2_B1_B1_A2

### Relational analysis result of IS_A1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7013528, upper bound: 0.7010354
time: 4.07 seconds

## BFS IS instance: IS_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -7.2871108, -5.1850238, -7.3336797, -5.1333785, -1.6817336, 1.7088988
1: 1.9523101, 3.5671887, 1.8865840, 3.5396194, -1.1751323, 1.2633622
2: -4.9458294, -3.2920184, -4.9145927, -3.2934473, -1.1640956, 1.1210165
3: -11.0230713, -8.8937340, -11.0278225, -8.8843079, -1.4692116, 1.5194001
4: -5.5871902, -3.8462226, -5.5578289, -3.7497990, -1.5856438, 1.4454181
5: -9.0531063, -7.3238020, -9.0834894, -7.2988968, -1.6770115, 1.6323476
6: -6.4977789, -4.3187189, -6.5208421, -4.3968172, -1.4715691, 1.5342834
7: -8.8142433, -7.4146390, -8.7458220, -7.3882999, -1.2218394, 0.9581885
8: 0.9873629, 2.5223799, 1.0488839, 2.6076536, -1.1937020, 1.1592143
9: -9.4361315, -7.4229994, -9.3737803, -7.2162337, -1.7716041, 1.5128291

Time for backsubstitution: 5.73 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 1928
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 1683
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 655

## Relational analysis of IS_A1_B2_B1_B2_A1

### Relational analysis result of IS_A1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7070655, upper bound: 0.6996673
time: 4.03 seconds

## Relational analysis of IS_A1_B2_B1_B2_A2

### Relational analysis result of IS_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7070655, upper bound: 0.7007226
time: 4.06 seconds

## BFS IS instance: IS_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -7.2857161, -5.1842437, -7.3254919, -5.1350980, -1.6785178, 1.7056530
1: 1.9514351, 3.5785573, 1.9589834, 3.5546386, -1.1821997, 1.1921458
2: -4.9589376, -3.2903762, -4.9638028, -3.2957942, -1.1256018, 1.1449399
3: -11.0314407, -8.8924837, -11.0548811, -8.8828478, -1.4674015, 1.5323572
4: -5.6060896, -3.8461382, -5.5875945, -3.8422475, -1.4917836, 1.4185627
5: -9.0561008, -7.3220491, -9.0812263, -7.2922115, -1.6807632, 1.6290317
6: -6.4979773, -4.2929778, -6.5262251, -4.3229232, -1.5068150, 1.5120301
7: -8.8435087, -7.4144306, -8.8468180, -7.3890629, -1.1722801, 0.9727788
8: 0.9722877, 2.5226622, 1.0117378, 2.5483556, -1.1257262, 1.1687331
9: -9.4737549, -7.4233875, -9.4464760, -7.4114919, -1.5428355, 1.5116389

Time for backsubstitution: 5.72 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 1683
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 2832
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 655

## Relational analysis of IS_A1_B2_B2_B1_A1

### Relational analysis result of IS_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7033748, upper bound: 0.7009783
time: 4.62 seconds

## Relational analysis of IS_A1_B2_B2_B1_A2

### Relational analysis result of IS_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7033748, upper bound: 0.7021798
time: 4.05 seconds

## BFS IS instance: IS_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -7.2851343, -5.1849003, -7.3273830, -5.1377716, -1.6774349, 1.7075055
1: 1.9521894, 3.5671630, 1.8846698, 3.5361454, -1.1835446, 1.2715311
2: -4.9473982, -3.2908797, -4.9284234, -3.2881997, -1.1674066, 1.1407439
3: -11.0230703, -8.8933849, -11.0267334, -8.8830233, -1.4706149, 1.5191660
4: -5.5867152, -3.8462319, -5.5566864, -3.7501471, -1.5849414, 1.4457955
5: -9.0529499, -7.3224425, -9.0852928, -7.2936592, -1.6775489, 1.6394062
6: -6.4970007, -4.3180451, -6.5204949, -4.4102063, -1.4706321, 1.5304151
7: -8.8153353, -7.4146428, -8.7495785, -7.3882422, -1.2237840, 0.9615786
8: 0.9851346, 2.5223169, 1.0404515, 2.6171303, -1.1932809, 1.1675112
9: -9.4361010, -7.4234767, -9.3714933, -7.2172661, -1.7708182, 1.5138416

Time for backsubstitution: 5.76 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 655
type: B, layer: 3, pos: 1928
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 655

## Relational analysis of IS_A1_B2_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7090649, upper bound: 0.7007920
time: 4.60 seconds

## Relational analysis of IS_A1_B2_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7090649, upper bound: 0.7018394
time: 4.12 seconds

## BFS IS instance: IS_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -7.3293734, -5.1357098, -7.3322072, -5.1142530, -1.7712264, 1.6557069
1: 1.9621582, 3.5619431, 1.9451399, 3.5743499, -1.1815002, 1.2219739
2: -4.9495640, -3.3079512, -4.9525485, -3.2910638, -1.1245484, 1.1168774
3: -11.0483208, -8.8850670, -11.0516415, -8.8785744, -1.5837650, 1.5925536
4: -5.5741711, -3.8426971, -5.6104479, -3.8426867, -1.4655824, 1.4903476
5: -9.0780220, -7.3032994, -9.0794010, -7.3074303, -1.7655582, 1.6846771
6: -6.5083890, -4.3005090, -6.5506616, -4.3211207, -1.5052266, 1.5676229
7: -8.8383942, -7.3959737, -8.8165712, -7.3975482, -1.1374218, 1.1215250
8: 1.0116034, 2.5141950, 1.0022974, 2.5462871, -1.1609540, 1.1681263
9: -9.4472685, -7.4098272, -9.4898396, -7.4020934, -1.5296938, 1.5656645

Time for backsubstitution: 5.70 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 1928

## Relational analysis of IS_A2_B1_A1_A1_B1

### Relational analysis result of IS_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7048782, upper bound: 0.7035407
time: 4.13 seconds

## Relational analysis of IS_A2_B1_A1_A1_B2

### Relational analysis result of IS_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7048782, upper bound: 0.7035493
time: 4.42 seconds

## BFS IS instance: IS_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -7.3311281, -5.1381435, -7.3314404, -5.1150970, -1.7728209, 1.6546562
1: 1.8879123, 3.5469038, 1.9460695, 3.5638936, -1.2585545, 1.2208974
2: -4.9137144, -3.3010609, -4.9401054, -3.2915709, -1.1202590, 1.1577096
3: -11.0238266, -8.8851194, -11.0433464, -8.8796177, -1.5713568, 1.5934978
4: -5.5437498, -3.7510743, -5.5910745, -3.8429091, -1.4915919, 1.5838916
5: -9.0819340, -7.3047142, -9.0766554, -7.3079805, -1.7708936, 1.6808653
6: -6.5042467, -4.3894296, -6.5494318, -4.3488593, -1.5242238, 1.5333953
7: -8.7407045, -7.3999038, -8.7881861, -7.3978901, -1.1241317, 1.1720662
8: 1.0399361, 2.5781798, 1.0149312, 2.5458913, -1.1586430, 1.2143168
9: -9.3720179, -7.2174630, -9.4524078, -7.4021873, -1.5337586, 1.7752306

Time for backsubstitution: 5.73 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1928
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 1928

## Relational analysis of IS_A2_B1_A1_A2_B1

### Relational analysis result of IS_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7035005, upper bound: 0.7085884
time: 3.90 seconds

## Relational analysis of IS_A2_B1_A1_A2_B2

### Relational analysis result of IS_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7035005, upper bound: 0.7087569
time: 4.27 seconds

## BFS IS instance: IS_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -7.3230782, -5.1401405, -7.3302355, -5.1141143, -1.7698774, 1.6514020
1: 1.9608455, 3.5585098, 1.9448309, 3.5743251, -1.1892354, 1.2309713
2: -4.9628320, -3.3027072, -4.9541121, -3.2899270, -1.1441436, 1.1201729
3: -11.0472288, -8.8838129, -11.0516567, -8.8781986, -1.5835156, 1.5943503
4: -5.5730443, -3.8430152, -5.6099782, -3.8426838, -1.4653811, 1.4896679
5: -9.0799141, -7.2982659, -9.0792437, -7.3060465, -1.7733154, 1.6850314
6: -6.5080800, -4.3139939, -6.5498962, -4.3204012, -1.5012884, 1.5673902
7: -8.8418398, -7.3959351, -8.8176041, -7.3975487, -1.1405263, 1.1234236
8: 1.0034294, 2.5235710, 1.0000739, 2.5462208, -1.1693549, 1.1660779
9: -9.4450645, -7.4110050, -9.4898167, -7.4025421, -1.5309646, 1.5644503

Time for backsubstitution: 5.77 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 1928

## Relational analysis of IS_A2_B1_A2_A1_B1

### Relational analysis result of IS_A2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7060512, upper bound: 0.7056082
time: 4.15 seconds

## Relational analysis of IS_A2_B1_A2_A1_B2

### Relational analysis result of IS_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7060512, upper bound: 0.7056102
time: 6.42 seconds

## BFS IS instance: IS_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -7.3248305, -5.1425691, -7.3294678, -5.1149578, -1.7714720, 1.6503417
1: 1.8865724, 3.5434356, 1.9457593, 3.5638685, -1.2661080, 1.2298610
2: -4.9271216, -3.2957506, -4.9416995, -3.2904348, -1.1398239, 1.1610479
3: -11.0227299, -8.8839960, -11.0433626, -8.8792400, -1.5711069, 1.5953107
4: -5.5425911, -3.7514179, -5.5906057, -3.8429060, -1.4914412, 1.5832009
5: -9.0837364, -7.2997026, -9.0764952, -7.3065982, -1.7771382, 1.6812372
6: -6.5038967, -4.4029579, -6.5486703, -4.3481379, -1.5202513, 1.5330396
7: -8.7440338, -7.3998437, -8.7892218, -7.3978891, -1.1272616, 1.1738732
8: 1.0318408, 2.5876389, 1.0127072, 2.5458274, -1.1668768, 1.2123516
9: -9.3697548, -7.2186165, -9.4523821, -7.4026365, -1.5347762, 1.7742181

Time for backsubstitution: 5.83 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1928
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 1683
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 1928

## Relational analysis of IS_A2_B1_A2_A2_B1

### Relational analysis result of IS_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7046061, upper bound: 0.7106010
time: 4.34 seconds

## Relational analysis of IS_A2_B1_A2_A2_B2

### Relational analysis result of IS_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7046061, upper bound: 0.7106706
time: 4.51 seconds

## BFS IS instance: IS_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -7.3328705, -5.1351347, -7.3325033, -5.1134238, -1.7726994, 1.6702986
1: 1.9484518, 3.5662255, 1.9600971, 3.5761228, -1.2026155, 1.1988897
2: -4.9541268, -3.2935309, -4.9525690, -3.2996616, -1.1137280, 1.1337126
3: -11.0519543, -8.8812637, -11.0600338, -8.8825283, -1.5803843, 1.6027155
4: -5.5788908, -3.8423386, -5.6121769, -3.8410828, -1.4744267, 1.4877281
5: -9.0793257, -7.2888489, -9.0846071, -7.2887778, -1.7463188, 1.7256656
6: -6.5238867, -4.2923388, -6.5315351, -4.3074617, -1.5469742, 1.5415812
7: -8.8506851, -7.3953485, -8.8447914, -7.3847985, -1.1671581, 1.1148043
8: 0.9937854, 2.5148540, 1.0051031, 2.5356479, -1.1750598, 1.1811278
9: -9.4477739, -7.4055734, -9.4899597, -7.4038849, -1.5264366, 1.5793233

Time for backsubstitution: 5.78 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 655
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 655

## Relational analysis of IS_A2_B2_B1_A1_A1

### Relational analysis result of IS_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7072215, upper bound: 0.7020897
time: 6.46 seconds

## Relational analysis of IS_A2_B2_B1_A1_A2

### Relational analysis result of IS_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7072215, upper bound: 0.7042821
time: 4.44 seconds

## BFS IS instance: IS_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -7.3346300, -5.1375637, -7.3317366, -5.1142673, -1.7743149, 1.6691089
1: 1.8741560, 3.5477097, 1.9610002, 3.5645695, -1.2801764, 1.1981783
2: -4.9183860, -3.2868643, -4.9398065, -3.3001697, -1.1104755, 1.1741304
3: -11.0271158, -8.8813429, -11.0515461, -8.8835688, -1.5678730, 1.6033301
4: -5.5488062, -3.7507606, -5.5927677, -3.8413038, -1.4998245, 1.5805755
5: -9.0832891, -7.2902904, -9.0819654, -7.2893252, -1.7518606, 1.7218246
6: -6.5197530, -4.3811698, -6.5303488, -4.3349423, -1.5711031, 1.5071144
7: -8.7527733, -7.3992519, -8.8154249, -7.3851538, -1.1538527, 1.1730437
8: 1.0227218, 2.5788093, 1.0187454, 2.5352550, -1.1726563, 1.2276313
9: -9.3725710, -7.2131720, -9.4523630, -7.4039783, -1.5305030, 1.7888622

Time for backsubstitution: 5.72 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 655
type: B, layer: 3, pos: 1928
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 1683
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 655

## Relational analysis of IS_A2_B2_B1_A2_A1

### Relational analysis result of IS_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7072215, upper bound: 0.7081091
time: 7.65 seconds

## Relational analysis of IS_A2_B2_B1_A2_A2

### Relational analysis result of IS_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7072215, upper bound: 0.7096441
time: 4.27 seconds

## BFS IS instance: IS_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -7.3308973, -5.1349964, -7.3262105, -5.1178179, -1.7684808, 1.6689525
1: 1.9483345, 3.5662010, 1.9581842, 3.5726702, -1.2116637, 1.2071712
2: -4.9556561, -3.2923970, -4.9662676, -3.2944431, -1.1170003, 1.1534760
3: -11.0519657, -8.8808880, -11.0589514, -8.8811102, -1.5823503, 1.6024680
4: -5.5784273, -3.8423340, -5.6110268, -3.8414054, -1.4737463, 1.4875145
5: -9.0791645, -7.2874627, -9.0864468, -7.2835436, -1.7468753, 1.7333994
6: -6.5231242, -4.2916183, -6.5312185, -4.3207631, -1.5468340, 1.5376542
7: -8.8517742, -7.3953476, -8.8485470, -7.3847589, -1.1690898, 1.1181040
8: 0.9915228, 2.5147858, 0.9966650, 2.5450163, -1.1730835, 1.1895664
9: -9.4477491, -7.4060221, -9.4877396, -7.4049387, -1.5253582, 1.5805585

Time for backsubstitution: 5.75 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1928
type: A, layer: 3, pos: 655
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 1928

## Relational analysis of IS_A2_B2_B2_A1_B1

### Relational analysis result of IS_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7021785, upper bound: 0.7056081
time: 3.93 seconds

## Relational analysis of IS_A2_B2_B2_A1_B2

### Relational analysis result of IS_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7021785, upper bound: 0.7056100
time: 4.41 seconds

## BFS IS instance: IS_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -7.3326540, -5.1374249, -7.3254414, -5.1186600, -1.7701364, 1.6677608
1: 1.8740382, 3.5476820, 1.9590950, 3.5611157, -1.2891104, 1.2065558
2: -4.9199400, -3.2857192, -4.9535074, -3.2949426, -1.1138151, 1.1940084
3: -11.0271273, -8.8810215, -11.0504637, -8.8821754, -1.5698419, 1.6030793
4: -5.5483341, -3.7507589, -5.5916262, -3.8416262, -1.4991593, 1.5803437
5: -9.0831318, -7.2889194, -9.0837727, -7.2840910, -1.7524204, 1.7295570
6: -6.5189953, -4.3804560, -6.5300226, -4.3482647, -1.5708013, 1.5032094
7: -8.7538605, -7.3992467, -8.8191814, -7.3851128, -1.1557851, 1.1761537
8: 1.0204625, 2.5787458, 1.0103059, 2.5446434, -1.1706312, 1.2358825
9: -9.3725424, -7.2136145, -9.4501390, -7.4050317, -1.5294147, 1.7901444

Time for backsubstitution: 5.75 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1928
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 1928

## Relational analysis of IS_A2_B2_B2_A2_B1

### Relational analysis result of IS_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7018384, upper bound: 0.7106017
time: 4.08 seconds

## Relational analysis of IS_A2_B2_B2_A2_B2

### Relational analysis result of IS_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7018384, upper bound: 0.7106721
time: 4.30 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 14.35 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.35
Output dim: 1, lower bound: -0.7039644, upper bound: 0.6945510
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.35
Output dim: 1, lower bound: -0.7039644, upper bound: 0.6998787
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.35
Output dim: 1, lower bound: -0.7104348, upper bound: 0.6988315
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.35
Output dim: 1, lower bound: -0.7104348, upper bound: 0.6998785
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.35
Output dim: 1, lower bound: -0.7052742, upper bound: 0.6964164
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.35
Output dim: 1, lower bound: -0.7052742, upper bound: 0.7018404
IS_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 14.35
Output dim: 1, lower bound: -0.7073261, upper bound: 0.7018386
IS_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 14.35
Output dim: 1, lower bound: -0.7073261, upper bound: 0.7018380
IS_A1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.35
Output dim: 1, lower bound: -0.7013528, upper bound: 0.6998319
IS_A1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.35
Output dim: 1, lower bound: -0.7013528, upper bound: 0.7010354
IS_A1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.35
Output dim: 1, lower bound: -0.7070655, upper bound: 0.6996673
IS_A1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.35
Output dim: 1, lower bound: -0.7070655, upper bound: 0.7007226
IS_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.35
Output dim: 1, lower bound: -0.7033748, upper bound: 0.7009783
IS_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.35
Output dim: 1, lower bound: -0.7033748, upper bound: 0.7021798
IS_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.35
Output dim: 1, lower bound: -0.7090649, upper bound: 0.7007920
IS_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.35
Output dim: 1, lower bound: -0.7090649, upper bound: 0.7018394
IS_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 14.35
Output dim: 1, lower bound: -0.7048782, upper bound: 0.7035407
IS_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 14.35
Output dim: 1, lower bound: -0.7048782, upper bound: 0.7035493
IS_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 14.35
Output dim: 1, lower bound: -0.7035005, upper bound: 0.7085884
IS_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 14.35
Output dim: 1, lower bound: -0.7035005, upper bound: 0.7087569
IS_A2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 14.35
Output dim: 1, lower bound: -0.7060512, upper bound: 0.7056082
IS_A2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 14.35
Output dim: 1, lower bound: -0.7060512, upper bound: 0.7056102
IS_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 14.35
Output dim: 1, lower bound: -0.7046061, upper bound: 0.7106010
IS_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 14.35
Output dim: 1, lower bound: -0.7046061, upper bound: 0.7106706
IS_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 14.35
Output dim: 1, lower bound: -0.7072215, upper bound: 0.7020897
IS_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 14.35
Output dim: 1, lower bound: -0.7072215, upper bound: 0.7042821
IS_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 14.35
Output dim: 1, lower bound: -0.7072215, upper bound: 0.7081091
IS_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 14.35
Output dim: 1, lower bound: -0.7072215, upper bound: 0.7096441
IS_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 14.35
Output dim: 1, lower bound: -0.7021785, upper bound: 0.7056081
IS_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 14.35
Output dim: 1, lower bound: -0.7021785, upper bound: 0.7056100
IS_A2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 14.35
Output dim: 1, lower bound: -0.7018384, upper bound: 0.7106017
IS_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 14.35
Output dim: 1, lower bound: -0.7018384, upper bound: 0.7106721

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -7.2833986, -5.1853943, -7.3317785, -5.1320372, -1.6807079, 1.6966252
1: 1.9659669, 3.5585103, 1.9458365, 3.5560102, -1.1510191, 1.1801717
2: -4.9500875, -3.3074389, -4.9500904, -3.2922132, -1.1235280, 1.1063547
3: -11.0269098, -8.8982687, -11.0465784, -8.8800745, -1.4618120, 1.5167785
4: -5.5771642, -3.8465567, -5.5870266, -3.8435473, -1.4533477, 1.4165828
5: -9.0509996, -7.3375187, -9.0742741, -7.3161387, -1.6940136, 1.5804391
6: -6.4827929, -4.3048172, -6.5456443, -4.3231993, -1.4668107, 1.5297062
7: -8.8270559, -7.4151440, -8.8140860, -7.4042459, -1.1261830, 0.9778819
8: 1.0072279, 2.5211749, 1.0179391, 2.5465055, -1.0915196, 1.1352036
9: -9.4320173, -7.4273677, -9.4491062, -7.4086318, -1.4955733, 1.4960713

Time for backsubstitution: 5.68 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 655
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 1928
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 655

## Relational analysis of IS_A1_B1_A1_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7039806, upper bound: 0.6985191
time: 4.13 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7039806, upper bound: 0.6999813
time: 4.20 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -7.2868319, -5.1869459, -7.3317785, -5.1320372, -1.6848021, 1.6955249
1: 1.8901472, 3.5443907, 1.9458365, 3.5560102, -1.2341862, 1.1760211
2: -4.9176245, -3.2994232, -4.9500904, -3.2922132, -1.1264815, 1.1517618
3: -10.9990692, -8.8983040, -11.0465784, -8.8800745, -1.4443512, 1.5243802
4: -5.5497370, -3.7441819, -5.5870266, -3.8435473, -1.4377160, 1.5211918
5: -9.0538235, -7.3387403, -9.0742741, -7.3161387, -1.7009125, 1.5795536
6: -6.4837899, -4.3836350, -6.5456443, -4.3231993, -1.4923701, 1.4821970
7: -8.7337952, -7.4101396, -8.8140860, -7.4042459, -1.0952120, 1.0439751
8: 1.0330763, 2.5899358, 1.0179391, 2.5465055, -1.0718000, 1.1990232
9: -9.3549261, -7.2349167, -9.4491062, -7.4086318, -1.4865201, 1.7430210

Time for backsubstitution: 5.76 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 655
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1928
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 655

## Relational analysis of IS_A1_B1_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7039806, upper bound: 0.6990939
time: 4.36 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7039806, upper bound: 0.7002201
time: 4.19 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -7.2790813, -5.1872377, -7.3336687, -5.1346874, -1.6690669, 1.6891356
1: 1.9631176, 3.5572872, 1.8669190, 3.5422933, -1.1574912, 1.2722049
2: -4.9389129, -3.3117843, -4.9157066, -3.2848377, -1.1718049, 1.0978160
3: -11.0036755, -8.8977928, -11.0194988, -8.8801069, -1.4539785, 1.5126414
4: -5.5763206, -3.8482594, -5.5604291, -3.7508633, -1.5603571, 1.4465261
5: -9.0399961, -7.3661509, -9.0781479, -7.3176003, -1.6722465, 1.5608878
6: -6.4842815, -4.3505721, -6.5372138, -4.4109869, -1.4333515, 1.5400326
7: -8.7681818, -7.4346018, -8.7201862, -7.4021006, -1.1664851, 0.9444511
8: 1.0280247, 2.5251436, 1.0415802, 2.6194949, -1.1329604, 1.1257012
9: -9.4350624, -7.4291658, -9.3743877, -7.2146001, -1.7712352, 1.4982507

Time for backsubstitution: 5.78 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1928
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 1683
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 2832
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 1928

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7061653, upper bound: 0.6988306
time: 4.31 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7061653, upper bound: 0.6988298
time: 4.78 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -7.2814980, -5.1855392, -7.3336687, -5.1346874, -1.6692290, 1.6902676
1: 1.9680226, 3.5571313, 1.8669190, 3.5422933, -1.1512103, 1.2684388
2: -4.9399509, -3.3072233, -4.9157066, -3.2848377, -1.1733441, 1.1038214
3: -11.0140400, -8.8994160, -11.0194988, -8.8801069, -1.4621096, 1.5033226
4: -5.5785041, -3.8465581, -5.5604291, -3.7508633, -1.5723114, 1.4483995
5: -9.0514851, -7.3408704, -9.0781479, -7.3176003, -1.6953044, 1.5961299
6: -6.4823723, -4.3357177, -6.5372138, -4.4109869, -1.4293532, 1.5428624
7: -8.8061085, -7.4159441, -8.7201862, -7.4021006, -1.2016666, 0.9665315
8: 1.0149193, 2.5216250, 1.0415802, 2.6194949, -1.1617409, 1.1307919
9: -9.4347954, -7.4280825, -9.3743877, -7.2146001, -1.7711909, 1.5001900

Time for backsubstitution: 5.72 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 1683
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2832
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 1928

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7061653, upper bound: 0.6998791
time: 4.39 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7061653, upper bound: 0.6998781
time: 4.68 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -7.2771025, -5.1898918, -7.3298063, -5.1318979, -1.6793599, 1.6923161
1: 1.9647274, 3.5551076, 1.9455245, 3.5559835, -1.1582379, 1.1892633
2: -4.9633274, -3.3022144, -4.9516521, -3.2910843, -1.1432610, 1.1096534
3: -11.0257654, -8.8971291, -11.0465927, -8.8796997, -1.4615960, 1.5184731
4: -5.5760350, -3.8469610, -5.5865655, -3.8435431, -1.4531450, 1.4158161
5: -9.0528870, -7.3326626, -9.0741138, -7.3147559, -1.7017713, 1.5811687
6: -6.4822941, -4.3184657, -6.5448809, -4.3224792, -1.4632840, 1.5294840
7: -8.8304853, -7.4151525, -8.8151188, -7.4042463, -1.1292584, 0.9788737
8: 0.9990449, 2.5305686, 1.0157156, 2.5464420, -1.0984125, 1.1331806
9: -9.4297781, -7.4286432, -9.4490795, -7.4090943, -1.4960489, 1.4949667

Time for backsubstitution: 5.69 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1928
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 1928

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7024777, upper bound: 0.7017814
time: 4.24 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7024777, upper bound: 0.7017815
time: 4.60 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -7.2805395, -5.1914387, -7.3298063, -5.1318979, -1.6834579, 1.6912179
1: 1.8887887, 3.5409245, 1.9455245, 3.5559835, -1.2411962, 1.1851001
2: -4.9309592, -3.2941353, -4.9516521, -3.2910843, -1.1461403, 1.1550725
3: -10.9979239, -8.8972855, -11.0465927, -8.8796997, -1.4441357, 1.5260940
4: -5.5485573, -3.7446012, -5.5865655, -3.8435431, -1.4374228, 1.5204256
5: -9.0556583, -7.3338675, -9.0741138, -7.3147559, -1.7087908, 1.5803032
6: -6.4832544, -4.3972726, -6.5448809, -4.3224792, -1.4888487, 1.4816756
7: -8.7372360, -7.4101353, -8.8151188, -7.4042463, -1.0980098, 1.0449789
8: 1.0250878, 2.5994048, 1.0157156, 2.5464420, -1.0786097, 1.1971369
9: -9.3526182, -7.2360959, -9.4490795, -7.4090943, -1.4868860, 1.7420077

Time for backsubstitution: 5.75 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 655
type: B, layer: 3, pos: 1928
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 1725

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6347132, upper bound: 0.6760194
time: 4.24 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6783254, upper bound: 0.6754001
time: 4.01 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -7.2772150, -5.1900306, -7.2829370, -5.1881685, -1.6323566, 1.6341903
1: 1.9646740, 3.5624843, 1.8730595, 3.5361743, -1.1467872, 1.2647808
2: -4.9545622, -3.3011160, -4.9193020, -3.2906146, -1.1868832, 1.1065243
3: -11.0180426, -8.8963242, -10.9840412, -8.8956442, -1.4058568, 1.3723536
4: -5.5813127, -3.8467493, -5.5468421, -3.7458224, -1.5158815, 1.4334741
5: -9.0537090, -7.3329678, -9.0430632, -7.3598051, -1.5233817, 1.5359516
6: -6.4821978, -4.3400016, -6.4947329, -4.4063621, -1.4223454, 1.5018177
7: -8.8047676, -7.4149671, -8.7057571, -7.4283953, -1.0290897, 0.9445739
8: 0.9973407, 2.5311608, 1.0409913, 2.5982599, -1.1478401, 1.0511889
9: -9.4335823, -7.4283481, -9.3546028, -7.2345548, -1.7567163, 1.4735579

Time for backsubstitution: 5.76 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 2832
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 1725

## Relational analysis of IS_A1_B1_A2_B2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6775855, upper bound: 0.6230388
time: 4.13 seconds

## Relational analysis of IS_A1_B1_A2_B2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6761803, upper bound: 0.6708026
time: 4.30 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -7.2772150, -5.1900306, -7.3273973, -5.1388950, -1.6966939, 1.6916184
1: 1.9646740, 3.5624843, 1.8703754, 3.5390201, -1.1449630, 1.2767587
2: -4.9545622, -3.3011160, -4.9145803, -3.2920890, -1.1854303, 1.1059595
3: -11.0180426, -8.8963242, -11.0094957, -8.8827181, -1.4623437, 1.5185289
4: -5.5813127, -3.8467493, -5.5372000, -3.7524958, -1.5708046, 1.4357398
5: -9.0537090, -7.3329678, -9.0759716, -7.3253841, -1.7073441, 1.5738802
6: -6.4821978, -4.3400016, -6.5158176, -4.4118905, -1.4264183, 1.5600624
7: -8.8047676, -7.4149671, -8.7193108, -7.4094200, -1.1954689, 0.9492691
8: 0.9973407, 2.5311608, 1.0418878, 2.5965610, -1.1685264, 1.1305456
9: -9.4335823, -7.4283481, -9.3714447, -7.2170000, -1.7613466, 1.4980240

Time for backsubstitution: 5.73 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 1725
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 2832
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 912
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 1725

## Relational analysis of IS_A1_B1_A2_B2_B2_B1

### Relational analysis result of IS_A1_B1_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6321623, upper bound: 0.6727050
time: 4.38 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2_B2

### Relational analysis result of IS_A1_B1_A2_B2_B2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.6761802, upper bound: 0.6708017
time: 4.48 seconds

## BFS IS instance: IS_A1_B2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -7.2821794, -5.1862516, -7.3317862, -5.1306973, -1.6713367, 1.6859810
1: 1.9517014, 3.5676856, 1.9608703, 3.5580935, -1.1649859, 1.1773643
2: -4.9529929, -3.2978957, -4.9501200, -3.3009708, -1.1217375, 1.1169673
3: -11.0118618, -8.8942585, -11.0559683, -8.8842592, -1.4499998, 1.5358562
4: -5.5969143, -3.8481274, -5.5887079, -3.8419247, -1.4761934, 1.4268377
5: -9.0441437, -7.3598676, -9.0793400, -7.2974324, -1.7016439, 1.5790362
6: -6.4999857, -4.3244600, -6.5265293, -4.3096161, -1.4717422, 1.4948478
7: -8.7978001, -7.4342694, -8.8430796, -7.3891039, -1.1275048, 0.9756973
8: 1.0078521, 2.5260673, 1.0201669, 2.5389609, -1.0802979, 1.1387935
9: -9.4727240, -7.4258413, -9.4487000, -7.4104295, -1.5402155, 1.4997308

Time for backsubstitution: 5.76 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1683
type: B, layer: 3, pos: 1928
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 1683

## Relational analysis of IS_A1_B2_B1_B1_A1_A1

### Relational analysis result of IS_A1_B2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7013528, upper bound: 0.6991473
time: 4.77 seconds

## Relational analysis of IS_A1_B2_B1_B1_A1_A2

### Relational analysis result of IS_A1_B2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7013528, upper bound: 0.6998319
time: 4.59 seconds

## BFS IS instance: IS_A1_B2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -7.2845945, -5.1844554, -7.3317862, -5.1306973, -1.6938725, 1.7021410
1: 1.9549999, 3.5686369, 1.9608703, 3.5580935, -1.1707628, 1.1784532
2: -4.9552717, -3.2933257, -4.9501200, -3.3009708, -1.1209414, 1.1188884
3: -11.0224285, -8.8953648, -11.0559683, -8.8842592, -1.4525490, 1.5304332
4: -5.6008191, -3.8464265, -5.5887079, -3.8419247, -1.4746609, 1.4179096
5: -9.0557156, -7.3339643, -9.0793400, -7.2974324, -1.6793580, 1.5705571
6: -6.4980736, -4.3094282, -6.5265293, -4.3096161, -1.5060320, 1.5237536
7: -8.8370037, -7.4156089, -8.8430796, -7.3891039, -1.1279142, 0.9690647
8: 0.9922414, 2.5225163, 1.0201669, 2.5389609, -1.1127181, 1.1590116
9: -9.4726381, -7.4243765, -9.4487000, -7.4104295, -1.5459776, 1.5082183

Time for backsubstitution: 5.75 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1683
type: B, layer: 3, pos: 1928
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 1683

## Relational analysis of IS_A1_B2_B1_B1_A2_A1

### Relational analysis result of IS_A1_B2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7013528, upper bound: 0.7006409
time: 4.28 seconds

## Relational analysis of IS_A1_B2_B1_B1_A2_A2

### Relational analysis result of IS_A1_B2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7013528, upper bound: 0.7010354
time: 4.06 seconds

## BFS IS instance: IS_A1_B2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -7.2815962, -5.1869092, -7.3336797, -5.1333785, -1.6702404, 1.6878133
1: 1.9524932, 3.5573988, 1.8865840, 3.5396194, -1.1667576, 1.2568855
2: -4.9413404, -3.2984104, -4.9145927, -3.2934473, -1.1638508, 1.1128001
3: -11.0036945, -8.8951387, -11.0278225, -8.8843079, -1.4530935, 1.5228662
4: -5.5792270, -3.8482208, -5.5578289, -3.7497990, -1.5696707, 1.4556146
5: -9.0411186, -7.3602591, -9.0834894, -7.2988968, -1.6984072, 1.5894108
6: -6.4989963, -4.3495278, -6.5208421, -4.3968172, -1.4351182, 1.5142753
7: -8.7698812, -7.4344859, -8.7458220, -7.3882999, -1.1800756, 0.9636612
8: 1.0189223, 2.5257001, 1.0488839, 2.6076536, -1.1470277, 1.1373389
9: -9.4352093, -7.4259310, -9.3737803, -7.2162337, -1.7684894, 1.5042284

Time for backsubstitution: 5.73 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1928
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 1928

## Relational analysis of IS_A1_B2_B1_B2_A1_B1

### Relational analysis result of IS_A1_B2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7019656, upper bound: 0.6996657
time: 4.26 seconds

## Relational analysis of IS_A1_B2_B1_B2_A1_B2

### Relational analysis result of IS_A1_B2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7019656, upper bound: 0.6996653
time: 4.27 seconds

## BFS IS instance: IS_A1_B2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -7.2840114, -5.1851106, -7.3336797, -5.1333785, -1.6926904, 1.7039375
1: 1.9557540, 3.5572417, 1.8865840, 3.5396194, -1.1721439, 1.2569702
2: -4.9437437, -3.2938371, -4.9145927, -3.2934473, -1.1627822, 1.1156982
3: -11.0140572, -8.8962669, -11.0278225, -8.8843079, -1.4556084, 1.5172367
4: -5.5814562, -3.8465190, -5.5578289, -3.7497990, -1.5669680, 1.4451244
5: -9.0525684, -7.3343534, -9.0834894, -7.2988968, -1.6761241, 1.5822496
6: -6.4970922, -4.3340368, -6.5208421, -4.3968172, -1.4701157, 1.5436518
7: -8.8088303, -7.4158206, -8.7458220, -7.3882999, -1.1865857, 0.9577417
8: 1.0050845, 2.5221715, 1.0488839, 2.6076536, -1.1743579, 1.1579700
9: -9.4349442, -7.4244657, -9.3737803, -7.2162337, -1.7732067, 1.5106900

Time for backsubstitution: 5.75 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1928
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 1465
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 2832
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 1269
type: B, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 1928

## Relational analysis of IS_A1_B2_B1_B2_A2_B1

### Relational analysis result of IS_A1_B2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7019656, upper bound: 0.7007238
time: 3.96 seconds

## Relational analysis of IS_A1_B2_B1_B2_A2_B2

### Relational analysis result of IS_A1_B2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7019656, upper bound: 0.7007229
time: 4.48 seconds

## BFS IS instance: IS_A1_B2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -7.2802048, -5.1861267, -7.3254919, -5.1350980, -1.6671128, 1.6845284
1: 1.9513767, 3.5676613, 1.9589834, 3.5546386, -1.1731789, 1.1857498
2: -4.9545641, -3.2967567, -4.9638028, -3.2957942, -1.1250765, 1.1367263
3: -11.0118656, -8.8939056, -11.0548811, -8.8828478, -1.4512863, 1.5356216
4: -5.5964413, -3.8481336, -5.5875945, -3.8422475, -1.4755096, 1.4271863
5: -9.0439796, -7.3585072, -9.0812263, -7.2922115, -1.7022066, 1.5861011
6: -6.4992070, -4.3237858, -6.5262251, -4.3229232, -1.4711647, 1.4909151
7: -8.7988424, -7.4342756, -8.8468180, -7.3890629, -1.1293809, 0.9793723
8: 1.0056653, 2.5260029, 1.0117378, 2.5483556, -1.0797572, 1.1474323
9: -9.4726915, -7.4263105, -9.4464760, -7.4114919, -1.5393484, 1.5009766

Time for backsubstitution: 5.76 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1683
type: B, layer: 3, pos: 1928
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 2832
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 1683

## Relational analysis of IS_A1_B2_B2_B1_A1_A1

### Relational analysis result of IS_A1_B2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7033748, upper bound: 0.7003945
time: 4.48 seconds

## Relational analysis of IS_A1_B2_B2_B1_A1_A2

### Relational analysis result of IS_A1_B2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7033748, upper bound: 0.7009791
time: 4.31 seconds

## BFS IS instance: IS_A1_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -7.2826185, -5.1843328, -7.3254919, -5.1350980, -1.6896820, 1.7006922
1: 1.9548676, 3.5686111, 1.9589834, 3.5546386, -1.1792428, 1.1870553
2: -4.9568367, -3.2921872, -4.9638028, -3.2957942, -1.1242588, 1.1386473
3: -11.0224295, -8.8950138, -11.0548811, -8.8828478, -1.4539070, 1.5301971
4: -5.6003442, -3.8464336, -5.5875945, -3.8422475, -1.4739757, 1.4182680
5: -9.0555611, -7.3326054, -9.0812263, -7.2922115, -1.6798806, 1.5776196
6: -6.4972906, -4.3087454, -6.5262251, -4.3229232, -1.5053601, 1.5198238
7: -8.8380938, -7.4156132, -8.8468180, -7.3890629, -1.1298449, 0.9723647
8: 0.9900031, 2.5224524, 1.0117378, 2.5483556, -1.1120844, 1.1674740
9: -9.4726048, -7.4248533, -9.4464760, -7.4114919, -1.5451093, 1.5094833

Time for backsubstitution: 5.75 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1683
type: B, layer: 3, pos: 1928
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 1501
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 1683

## Relational analysis of IS_A1_B2_B2_B1_A2_A1

### Relational analysis result of IS_A1_B2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7033748, upper bound: 0.7017822
time: 3.98 seconds

## Relational analysis of IS_A1_B2_B2_B1_A2_A2

### Relational analysis result of IS_A1_B2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7033748, upper bound: 0.7021798
time: 4.02 seconds

## BFS IS instance: IS_A1_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -7.2796211, -5.1867824, -7.3273830, -5.1377716, -1.6660089, 1.6864142
1: 1.9521689, 3.5573735, 1.8846698, 3.5361454, -1.1749172, 1.2650547
2: -4.9429216, -3.2972713, -4.9284234, -3.2881997, -1.1671844, 1.1325239
3: -11.0036983, -8.8947887, -11.0267334, -8.8830233, -1.4544969, 1.5226321
4: -5.5787559, -3.8482265, -5.5566864, -3.7501471, -1.5689716, 1.4559820
5: -9.0409565, -7.3589010, -9.0852928, -7.2936592, -1.6989865, 1.5964746
6: -6.4982157, -4.3488503, -6.5204949, -4.4102063, -1.4342823, 1.5103836
7: -8.7709198, -7.4344902, -8.7495785, -7.3882422, -1.1819606, 0.9673660
8: 1.0167356, 2.5256367, 1.0404515, 2.6171303, -1.1465681, 1.1457937
9: -9.4351797, -7.4264045, -9.3714933, -7.2172661, -1.7677045, 1.5052059

Time for backsubstitution: 5.78 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1928
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 1683
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 1928

## Relational analysis of IS_A1_B2_B2_B2_A1_B1

### Relational analysis result of IS_A1_B2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7039330, upper bound: 0.7007942
time: 4.81 seconds

## Relational analysis of IS_A1_B2_B2_B2_A1_B2

### Relational analysis result of IS_A1_B2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7039330, upper bound: 0.7007916
time: 7.41 seconds

## BFS IS instance: IS_A1_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -7.2820349, -5.1849890, -7.3273830, -5.1377716, -1.6884885, 1.7025442
1: 1.9556217, 3.5572166, 1.8846698, 3.5361454, -1.1805882, 1.2657123
2: -4.9452963, -3.2926991, -4.9284234, -3.2881997, -1.1660986, 1.1355953
3: -11.0140581, -8.8959141, -11.0267334, -8.8830233, -1.4570837, 1.5170031
4: -5.5809851, -3.8465264, -5.5566864, -3.7501471, -1.5662684, 1.4455009
5: -9.0524120, -7.3329954, -9.0852928, -7.2936592, -1.6766658, 1.5893221
6: -6.4963160, -4.3333492, -6.5204949, -4.4102063, -1.4691777, 1.5397696
7: -8.8099203, -7.4158268, -8.7495785, -7.3882422, -1.1884773, 0.9611316
8: 1.0028486, 2.5221076, 1.0404515, 2.6171303, -1.1738470, 1.1662631
9: -9.4349098, -7.4249444, -9.3714933, -7.2172661, -1.7724214, 1.5116870

Time for backsubstitution: 5.77 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1928
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1725
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 430
type: A, layer: 3, pos: 1683
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 165
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 402
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 1509
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1159
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 1928

## Relational analysis of IS_A1_B2_B2_B2_A2_B1

### Relational analysis result of IS_A1_B2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7039330, upper bound: 0.7018384
time: 4.35 seconds

## Relational analysis of IS_A1_B2_B2_B2_A2_B2

### Relational analysis result of IS_A1_B2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7039330, upper bound: 0.7018383
time: 4.62 seconds

## BFS IS instance: IS_A2_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -7.3293734, -5.1357098, -7.2821794, -5.1862516, -1.6945944, 1.6921234
1: 1.9621582, 3.5619431, 1.9517014, 3.5676856, -1.1668339, 1.1603432
2: -4.9495640, -3.3079512, -4.9529929, -3.2978957, -1.1163599, 1.1156319
3: -11.0483208, -8.8850670, -11.0118618, -8.8942585, -1.5536919, 1.4472339
4: -5.5741711, -3.8426971, -5.5969143, -3.8481274, -1.4176393, 1.4755769
5: -9.0780220, -7.3032994, -9.0441437, -7.3598676, -1.5609183, 1.7030506
6: -6.5083890, -4.3005090, -6.4999857, -4.3244600, -1.4939728, 1.4986691
7: -8.8383942, -7.3959737, -8.7978001, -7.4342694, -0.9576480, 1.1238487
8: 1.0116034, 2.5141950, 1.0078521, 2.5260673, -1.1445754, 1.0805473
9: -9.4472685, -7.4098272, -9.4727240, -7.4258413, -1.5019498, 1.5481417

Time for backsubstitution: 5.76 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 655
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2139
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 1683

## Relational analysis of IS_A2_B1_A1_A1_B1_B1

### Relational analysis result of IS_A2_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7048720, upper bound: 0.7035407
time: 4.11 seconds

## Relational analysis of IS_A2_B1_A1_A1_B1_B2

### Relational analysis result of IS_A2_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7048720, upper bound: 0.7035405
time: 4.36 seconds

## BFS IS instance: IS_A2_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -7.3293734, -5.1357098, -7.3283610, -5.1361842, -1.6532207, 1.6526742
1: 1.9621582, 3.5619431, 1.9488406, 3.5705924, -1.2028909, 1.2140188
2: -4.9495640, -3.3079512, -4.9499407, -3.2983036, -1.1169724, 1.1149987
3: -11.0483208, -8.8850670, -11.0348539, -8.8814220, -1.5812240, 1.5628514
4: -5.5741711, -3.8426971, -5.5930715, -3.8442376, -1.4643211, 1.4823160
5: -9.0780220, -7.3032994, -9.0770683, -7.3250799, -1.6783686, 1.6820521
6: -6.5083890, -4.3005090, -6.5234733, -4.3225722, -1.5038199, 1.5381575
7: -8.8383942, -7.3959737, -8.8146935, -7.4068918, -1.1177583, 1.1200047
8: 1.0116034, 2.5141950, 1.0047455, 2.5294971, -1.1741619, 1.1656728
9: -9.4472685, -7.4098272, -9.4873714, -7.4085107, -1.5214181, 1.5639925

Time for backsubstitution: 5.81 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 655
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 662
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1159
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 1683

## Relational analysis of IS_A2_B1_A1_A1_B2_B1

### Relational analysis result of IS_A2_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7048741, upper bound: 0.7035456
time: 4.32 seconds

## Relational analysis of IS_A2_B1_A1_A1_B2_B2

### Relational analysis result of IS_A2_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7048741, upper bound: 0.7035493
time: 4.10 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -7.3311281, -5.1381435, -7.2815962, -5.1869092, -1.6962833, 1.6902666
1: 1.8879123, 3.5469038, 1.9524932, 3.5573988, -1.2441130, 1.1626289
2: -4.9137144, -3.3010609, -4.9413404, -3.2984104, -1.1121118, 1.1556027
3: -11.0238266, -8.8851194, -11.0036945, -8.8951387, -1.5415406, 1.4497240
4: -5.5437498, -3.7510743, -5.5792270, -3.8482208, -1.4456887, 1.5682881
5: -9.0819340, -7.3047142, -9.0411186, -7.3602591, -1.5698023, 1.6999159
6: -6.5042467, -4.3894296, -6.4989963, -4.3495278, -1.5142410, 1.4628315
7: -8.7407045, -7.3999038, -8.7698812, -7.4344859, -0.9452627, 1.1716652
8: 1.0399361, 2.5781798, 1.0189223, 2.5257001, -1.1423099, 1.1359570
9: -9.3720179, -7.2174630, -9.4352093, -7.4259310, -1.5060129, 1.7559290

Time for backsubstitution: 5.78 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 655
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 1683
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 2139
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 662
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: B, layer: 3, pos: 914
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 1250
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 912
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 655

## Relational analysis of IS_A2_B1_A1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7035005, upper bound: 0.7070659
time: 4.06 seconds

## Relational analysis of IS_A2_B1_A1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7035005, upper bound: 0.7085883
time: 4.21 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -7.3311281, -5.1381435, -7.3276100, -5.1370273, -1.6562004, 1.6516180
1: 1.8879123, 3.5469038, 1.9497554, 3.5601168, -1.2703979, 1.2129400
2: -4.9137144, -3.3010609, -4.9373994, -3.2988391, -1.1124666, 1.1557784
3: -11.0238266, -8.8851194, -11.0273151, -8.8824434, -1.5688372, 1.5638051
4: -5.5437498, -3.7510743, -5.5736685, -3.8444524, -1.4903002, 1.5720091
5: -9.0819340, -7.3047142, -9.0743361, -7.3256159, -1.6851435, 1.6782923
6: -6.5042467, -4.3894296, -6.5222111, -4.3503313, -1.5229025, 1.5037475
7: -8.7407045, -7.3999038, -8.7863102, -7.4072347, -1.1023731, 1.1706066
8: 1.0399361, 2.5781798, 1.0173869, 2.5291176, -1.1698058, 1.2119887
9: -9.3720179, -7.2174630, -9.4497252, -7.4086037, -1.5257590, 1.7735162

Time for backsubstitution: 5.76 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 655
type: B, layer: 3, pos: 1725
type: A, layer: 3, pos: 1725
type: B, layer: 3, pos: 1683
type: B, layer: 3, pos: 430
type: A, layer: 3, pos: 430
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 2139
type: A, layer: 3, pos: 2139
type: A, layer: 3, pos: 402
type: B, layer: 3, pos: 402
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 165
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 2320
type: A, layer: 3, pos: 2320
type: B, layer: 3, pos: 2536
type: A, layer: 3, pos: 2536
type: B, layer: 3, pos: 662
type: B, layer: 3, pos: 1501
type: A, layer: 3, pos: 1501
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 662
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 1465
type: A, layer: 3, pos: 1465
type: A, layer: 3, pos: 1509
type: B, layer: 3, pos: 1509
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 2382
type: B, layer: 3, pos: 914
type: A, layer: 3, pos: 914
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 1250
type: A, layer: 3, pos: 1250
type: B, layer: 3, pos: 912
type: A, layer: 3, pos: 912
type: B, layer: 3, pos: 1159
type: A, layer: 3, pos: 1159
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 627
type: A, layer: 3, pos: 627
type: B, layer: 3, pos: 1269
type: A, layer: 3, pos: 1269

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 655

## Relational analysis of IS_A2_B1_A1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7035005, upper bound: 0.7072235
time: 4.17 seconds

## Relational analysis of IS_A2_B1_A1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.7035005, upper bound: 0.7087569
time: 4.60 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 14.76 seconds
IS_A1_B1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 14.76
Output dim: 1, lower bound: -0.7039806, upper bound: 0.6985191
IS_A1_B1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 14.76
Output dim: 1, lower bound: -0.7039806, upper bound: 0.6999813
IS_A1_B1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 14.76
Output dim: 1, lower bound: -0.7039806, upper bound: 0.6990939
IS_A1_B1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 14.76
Output dim: 1, lower bound: -0.7039806, upper bound: 0.7002201
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 14.76
Output dim: 1, lower bound: -0.7061653, upper bound: 0.6988306
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 14.76
Output dim: 1, lower bound: -0.7061653, upper bound: 0.6988298
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 14.76
Output dim: 1, lower bound: -0.7061653, upper bound: 0.6998791
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 14.76
Output dim: 1, lower bound: -0.7061653, upper bound: 0.6998781
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 14.76
Output dim: 1, lower bound: -0.7024777, upper bound: 0.7017814
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 14.76
Output dim: 1, lower bound: -0.7024777, upper bound: 0.7017815
IS_A1_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 14.76
Output dim: 1, lower bound: -0.6347132, upper bound: 0.6760194
IS_A1_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 14.76
Output dim: 1, lower bound: -0.6783254, upper bound: 0.6754001
IS_A1_B1_A2_B2_B1_A1, status: Status.VERIFIED, split count: 6, time: 14.76
Output dim: 1, lower bound: -0.6775855, upper bound: 0.6230388
IS_A1_B1_A2_B2_B1_A2, status: Status.VERIFIED, split count: 6, time: 14.76
Output dim: 1, lower bound: -0.6761803, upper bound: 0.6708026
IS_A1_B1_A2_B2_B2_B1, status: Status.VERIFIED, split count: 6, time: 14.76
Output dim: 1, lower bound: -0.6321623, upper bound: 0.6727050
IS_A1_B1_A2_B2_B2_B2, status: Status.VERIFIED, split count: 6, time: 14.76
Output dim: 1, lower bound: -0.6761802, upper bound: 0.6708017
IS_A1_B2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 14.76
Output dim: 1, lower bound: -0.7013528, upper bound: 0.6991473
IS_A1_B2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 14.76
Output dim: 1, lower bound: -0.7013528, upper bound: 0.6998319
IS_A1_B2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 14.76
Output dim: 1, lower bound: -0.7013528, upper bound: 0.7006409
IS_A1_B2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 14.76
Output dim: 1, lower bound: -0.7013528, upper bound: 0.7010354
IS_A1_B2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 14.76
Output dim: 1, lower bound: -0.7019656, upper bound: 0.6996657
IS_A1_B2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 14.76
Output dim: 1, lower bound: -0.7019656, upper bound: 0.6996653
IS_A1_B2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 14.76
Output dim: 1, lower bound: -0.7019656, upper bound: 0.7007238
IS_A1_B2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 14.76
Output dim: 1, lower bound: -0.7019656, upper bound: 0.7007229
IS_A1_B2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 14.76
Output dim: 1, lower bound: -0.7033748, upper bound: 0.7003945
IS_A1_B2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 14.76
Output dim: 1, lower bound: -0.7033748, upper bound: 0.7009791
IS_A1_B2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 14.76
Output dim: 1, lower bound: -0.7033748, upper bound: 0.7017822
IS_A1_B2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 14.76
Output dim: 1, lower bound: -0.7033748, upper bound: 0.7021798
IS_A1_B2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 14.76
Output dim: 1, lower bound: -0.7039330, upper bound: 0.7007942
IS_A1_B2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 14.76
Output dim: 1, lower bound: -0.7039330, upper bound: 0.7007916
IS_A1_B2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 14.76
Output dim: 1, lower bound: -0.7039330, upper bound: 0.7018384
IS_A1_B2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 14.76
Output dim: 1, lower bound: -0.7039330, upper bound: 0.7018383
IS_A2_B1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 14.76
Output dim: 1, lower bound: -0.7048720, upper bound: 0.7035407
IS_A2_B1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 14.76
Output dim: 1, lower bound: -0.7048720, upper bound: 0.7035405
IS_A2_B1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 14.76
Output dim: 1, lower bound: -0.7048741, upper bound: 0.7035456
IS_A2_B1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 14.76
Output dim: 1, lower bound: -0.7048741, upper bound: 0.7035493
IS_A2_B1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 14.76
Output dim: 1, lower bound: -0.7035005, upper bound: 0.7070659
IS_A2_B1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 14.76
Output dim: 1, lower bound: -0.7035005, upper bound: 0.7085883
IS_A2_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 14.76
Output dim: 1, lower bound: -0.7035005, upper bound: 0.7072235
IS_A2_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 14.76
Output dim: 1, lower bound: -0.7035005, upper bound: 0.7087569
IS_A2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 14.76
Output dim: 1, lower bound: -0.7060512, upper bound: 0.7056082
IS_A2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 14.76
Output dim: 1, lower bound: -0.7060512, upper bound: 0.7056102
IS_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 14.76
Output dim: 1, lower bound: -0.7046061, upper bound: 0.7106010
IS_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 14.76
Output dim: 1, lower bound: -0.7046061, upper bound: 0.7106706
IS_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 14.76
Output dim: 1, lower bound: -0.7072215, upper bound: 0.7020897
IS_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 14.76
Output dim: 1, lower bound: -0.7072215, upper bound: 0.7042821
IS_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 14.76
Output dim: 1, lower bound: -0.7072215, upper bound: 0.7081091
IS_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 14.76
Output dim: 1, lower bound: -0.7072215, upper bound: 0.7096441
IS_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 14.76
Output dim: 1, lower bound: -0.7021785, upper bound: 0.7056081
IS_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 14.76
Output dim: 1, lower bound: -0.7021785, upper bound: 0.7056100
IS_A2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 14.76
Output dim: 1, lower bound: -0.7018384, upper bound: 0.7106017
IS_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 14.76
Output dim: 1, lower bound: -0.7018384, upper bound: 0.7106721
Binary search (step 2): status=Status.UNKNOWN, k_low=4, k_high=4, k_mid=4, eps_mid=0.0156250, abs_max=1.2436916828155518
rel_dist={1: [-0.7636376522603388, 0.7636396744031151]}

## Binary Search with IS_dual Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01171875
execution time: 2410.92 seconds
