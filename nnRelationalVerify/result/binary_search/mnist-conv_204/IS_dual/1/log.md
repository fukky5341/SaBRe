## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 1.4889943489
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-5.2582340, -2.7679250, -5.2582340, -2.7679250, -2.4903090, 2.4903090)
1: (-10.1559734, -7.2571516, -10.1559734, -7.2571516, -2.8988218, 2.8988218)
2: (-5.5816660, -2.7165861, -5.5816660, -2.7165861, -2.8650799, 2.8650799)
3: (-12.1876364, -8.9962378, -12.1876364, -8.9962378, -3.1913986, 3.1913986)
4: (-8.7789192, -5.4293895, -8.7789192, -5.4293895, -3.3495297, 3.3495297)
5: (-0.9330347, 1.5743690, -0.9330347, 1.5743690, -2.5074036, 2.5074036)
6: (5.1026025, 7.4659190, 5.1026025, 7.4659190, -2.3633165, 2.3633165)
7: (-18.8783417, -15.4095268, -18.8783417, -15.4095268, -3.3238997, 3.3238983)
8: (-1.6320400, 1.3817253, -1.6320400, 1.3817253, -3.0137653, 3.0137653)
9: (-8.8765841, -6.3745003, -8.8765841, -6.3745003, -2.5020838, 2.5020838)

## BASE Result
execution time: IAR + LP analysis = 13.26 + 33.71 = 46.97 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -1.8584840, upper bound: 1.8584796


# Binary Search by BASE starts (time budget: 3553.03 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.309645652770996
rel_dist={6: [-1.4944306407619283, 1.4944305738519086]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.VERIFIED, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=2.2005128860473633
rel_dist={6: [-1.2600079034819816, 1.2600075251228153]}

## Binary search (step 2) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=4, k_high=5, k_mid=4, eps_mid=0.0156250, abs_max=2.2368907928466797
rel_dist={6: [-1.346308779708549, 1.3463093624116622]}

## Binary search (step 3) starts
Candidate k: 5, corresponding eps: 0.0195312


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=5, k_high=5, k_mid=5, eps_mid=0.0195312, abs_max=2.273268222808838
rel_dist={6: [-1.423624394997221, 1.4236241266449756]}

## Binary Search Result
Binary search time: 193.70 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.01953125


# Individual Split (IS_dual) starts
Time budget: 3359.33 seconds

## Binary search (step 0) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 481
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 6196
type: B, layer: 1, pos: 6196
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 5817
type: B, layer: 1, pos: 5817
type: A, layer: 1, pos: 4584
type: B, layer: 1, pos: 4584
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 481

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6729517, upper bound: 1.6888901
time: 4.40 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6888884, upper bound: 1.6888886
time: 4.27 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 8.88 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 8.88
Output dim: 6, lower bound: -1.6729517, upper bound: 1.6888901
IS_A2, status: Status.UNKNOWN, split count: 1, time: 8.88
Output dim: 6, lower bound: -1.6888884, upper bound: 1.6888886

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -5.2482443, -2.7802007, -5.2582340, -2.7679250, -2.3773623, 2.3776040
1: -10.1429062, -7.2638254, -10.1559734, -7.2571516, -2.8569174, 2.8629413
2: -5.5605159, -2.7213488, -5.5816660, -2.7165861, -2.8439298, 2.8603172
3: -12.1751766, -9.0126343, -12.1876364, -8.9962378, -3.1121097, 3.1088104
4: -8.7701921, -5.4502583, -8.7789192, -5.4293895, -3.3297815, 3.3237314
5: -0.9283434, 1.5568354, -0.9330347, 1.5743690, -2.5027122, 2.4898701
6: 5.1297588, 7.4612336, 5.1026025, 7.4659190, -2.3361602, 2.3586311
7: -18.8547440, -15.4136238, -18.8783417, -15.4095268, -3.0131087, 3.0321178
8: -1.6276751, 1.3615103, -1.6320400, 1.3817253, -3.0094004, 2.9935503
9: -8.8711119, -6.3979683, -8.8765841, -6.3745003, -2.4570379, 2.4421535

Time for backsubstitution: 12.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 6196
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 6221
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 5817
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 4584
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 481

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6729517, upper bound: 1.6729512
time: 4.47 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6729517, upper bound: 1.6888887
time: 4.03 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -5.2933302, -2.7619328, -5.2582312, -2.7679286, -2.4213696, 2.4081404
1: -10.1672659, -7.2483907, -10.1559668, -7.2571554, -2.8799815, 2.8930144
2: -5.5987654, -2.6825676, -5.5816560, -2.7165885, -2.8821769, 2.8990884
3: -12.1994400, -8.9904881, -12.1876287, -8.9962454, -3.1368537, 3.1329446
4: -8.8240805, -5.4225054, -8.7789145, -5.4293957, -3.3815584, 3.3564091
5: -0.9681324, 1.5843486, -0.9330323, 1.5743618, -2.5424943, 2.5173807
6: 5.0850186, 7.5114970, 5.1026125, 7.4659171, -2.3808985, 2.4088845
7: -18.8848648, -15.3919125, -18.8783302, -15.4095278, -3.0475397, 3.0539160
8: -1.6544285, 1.3965702, -1.6320381, 1.3817186, -3.0361471, 3.0286083
9: -8.9248171, -6.3680735, -8.8765821, -6.3745098, -2.4890337, 2.4790154

Time for backsubstitution: 12.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6196
type: A, layer: 1, pos: 6196
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 5817
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 4584
type: B, layer: 1, pos: 4584
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6196

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6883087, upper bound: 1.6888885
time: 4.26 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6888879, upper bound: 1.6888885
time: 4.41 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 21.48 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 21.48
Output dim: 6, lower bound: -1.6729517, upper bound: 1.6729512
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 21.48
Output dim: 6, lower bound: -1.6729517, upper bound: 1.6888887
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 21.48
Output dim: 6, lower bound: -1.6883087, upper bound: 1.6888885
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 21.48
Output dim: 6, lower bound: -1.6888879, upper bound: 1.6888885

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -5.2482443, -2.7802007, -5.2482443, -2.7802007, -2.3631110, 2.3631110
1: -10.1429062, -7.2638254, -10.1429062, -7.2638254, -2.8461800, 2.8461790
2: -5.5605159, -2.7213488, -5.5605159, -2.7213488, -2.8391671, 2.8391671
3: -12.1751766, -9.0126343, -12.1751766, -9.0126343, -3.0947542, 3.0947542
4: -8.7701921, -5.4502583, -8.7701921, -5.4502583, -3.3062048, 3.3062053
5: -0.9283434, 1.5568354, -0.9283434, 1.5568354, -2.4851789, 2.4851789
6: 5.1297588, 7.4612336, 5.1297588, 7.4612336, -2.3314748, 2.3314748
7: -18.8547440, -15.4136238, -18.8547440, -15.4136238, -3.0066762, 3.0066757
8: -1.6276751, 1.3615103, -1.6276751, 1.3615103, -2.9891853, 2.9891853
9: -8.8711119, -6.3979683, -8.8711119, -6.3979683, -2.4322400, 2.4322400

Time for backsubstitution: 12.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6196
type: B, layer: 1, pos: 6196
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 5817
type: A, layer: 1, pos: 5817
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 4584
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6196

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6729515, upper bound: 1.6723774
time: 4.25 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6729513, upper bound: 1.6729562
time: 4.51 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -5.2482443, -2.7802007, -5.2933302, -2.7619328, -2.3819132, 2.4071255
1: -10.1429062, -7.2638254, -10.1672659, -7.2483907, -2.8621154, 2.8692479
2: -5.5605159, -2.7213488, -5.5987654, -2.6825676, -2.8779483, 2.8774166
3: -12.1751766, -9.0126343, -12.1994400, -8.9904881, -3.1180553, 3.1195040
4: -8.7701921, -5.4502583, -8.8240805, -5.4225054, -3.3353758, 3.3579946
5: -0.9283434, 1.5568354, -0.9681324, 1.5843486, -2.5126920, 2.5249677
6: 5.1297588, 7.4612336, 5.0850186, 7.5114970, -2.3817382, 2.3762150
7: -18.8547440, -15.4136238, -18.8848648, -15.3919125, -3.0284848, 3.0394011
8: -1.6276751, 1.3615103, -1.6544285, 1.3965702, -3.0242453, 3.0159388
9: -8.8711119, -6.3979683, -8.9248171, -6.3680735, -2.4630542, 2.4643064

Time for backsubstitution: 12.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6196
type: B, layer: 1, pos: 6196
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 5817
type: A, layer: 1, pos: 5817
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 4584
type: B, layer: 1, pos: 4584
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6196

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6729515, upper bound: 1.6883106
time: 4.40 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6729513, upper bound: 1.6888900
time: 4.22 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -5.2902765, -2.7659867, -5.2325315, -2.7892945, -2.3913255, 2.3788013
1: -10.1563911, -7.2499199, -10.1026020, -7.2868237, -2.8374543, 2.8410540
2: -5.5968347, -2.6902475, -5.5558786, -2.7576213, -2.8392134, 2.8656311
3: -12.1940699, -8.9918013, -12.1595335, -9.0108671, -3.0758333, 3.0967598
4: -8.8223429, -5.4270535, -8.7585583, -5.4531527, -3.3171005, 3.3069201
5: -0.9656922, 1.5818973, -0.9188534, 1.5550935, -2.5207858, 2.5007505
6: 5.0871520, 7.5102048, 5.1222076, 7.4559798, -2.3688278, 2.3879972
7: -18.8811684, -15.3954306, -18.8535347, -15.4281769, -3.0179620, 3.0171461
8: -1.6486995, 1.3943911, -1.6000094, 1.3574467, -3.0061462, 2.9944005
9: -8.9225187, -6.3767781, -8.8432541, -6.4185257, -2.4427292, 2.4412370

Time for backsubstitution: 12.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6221
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 5817
type: B, layer: 1, pos: 5817
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 4584

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6221

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6806334, upper bound: 1.6888796
time: 4.40 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6882981, upper bound: 1.6888786
time: 4.14 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -5.2933302, -2.7619328, -5.2582245, -2.7679403, -2.4111543, 2.4081359
1: -10.1672659, -7.2483907, -10.1559544, -7.2571578, -2.8799791, 2.8678837
2: -5.5987654, -2.6825676, -5.5816522, -2.7165978, -2.8821676, 2.8990846
3: -12.1994400, -8.9904881, -12.1876163, -8.9962482, -3.1368508, 3.1271057
4: -8.8240805, -5.4225054, -8.7789125, -5.4294071, -3.3841119, 3.3564072
5: -0.9681324, 1.5843486, -0.9330264, 1.5743580, -2.5424905, 2.5173750
6: 5.0850186, 7.5114970, 5.1026163, 7.4659147, -2.3808961, 2.4088807
7: -18.8848648, -15.3919125, -18.8783245, -15.4095373, -3.0542297, 3.0485878
8: -1.6544285, 1.3965702, -1.6320291, 1.3817148, -3.0361433, 3.0285993
9: -8.9248171, -6.3680735, -8.8765793, -6.3745332, -2.4578681, 2.4763465

Time for backsubstitution: 12.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6196
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6196

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6888881, upper bound: 1.6883091
time: 4.50 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6888879, upper bound: 1.6888883
time: 4.38 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 21.90 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 21.90
Output dim: 6, lower bound: -1.6729515, upper bound: 1.6723774
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 21.90
Output dim: 6, lower bound: -1.6729513, upper bound: 1.6729562
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 21.90
Output dim: 6, lower bound: -1.6729515, upper bound: 1.6883106
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 21.90
Output dim: 6, lower bound: -1.6729513, upper bound: 1.6888900
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 21.90
Output dim: 6, lower bound: -1.6806334, upper bound: 1.6888796
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 21.90
Output dim: 6, lower bound: -1.6882981, upper bound: 1.6888786
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 21.90
Output dim: 6, lower bound: -1.6888881, upper bound: 1.6883091
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 21.90
Output dim: 6, lower bound: -1.6888879, upper bound: 1.6888883

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -5.2224331, -2.8014963, -5.2451324, -2.7841792, -2.3338346, 2.3332942
1: -10.0900106, -7.2931690, -10.1323891, -7.2653742, -2.7946758, 2.8045092
2: -5.5347357, -2.7622347, -5.5586176, -2.7290115, -2.8057241, 2.7963829
3: -12.1475792, -9.0272722, -12.1701813, -9.0139275, -3.0593643, 3.0340536
4: -8.7499352, -5.4736094, -8.7685585, -5.4547720, -3.2442384, 3.2420082
5: -0.9145565, 1.5375628, -0.9259325, 1.5544188, -2.4689753, 2.4634953
6: 5.1479845, 7.4512987, 5.1317983, 7.4599595, -2.3119750, 2.3195004
7: -18.8300114, -15.4321060, -18.8512249, -15.4171371, -2.9702110, 2.9778061
8: -1.5962601, 1.3372712, -1.6220074, 1.3594251, -2.9556851, 2.9592786
9: -8.8377466, -6.4417171, -8.8688116, -6.4066362, -2.3945603, 2.3861361

Time for backsubstitution: 12.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 6196
type: A, layer: 1, pos: 5817
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 4584

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6221

## Relational analysis of IS_A1_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6729470, upper bound: 1.6647027
time: 4.60 seconds

## Relational analysis of IS_A1_B1_A1_A2

### Relational analysis result of IS_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6729470, upper bound: 1.6723669
time: 4.56 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -5.2482414, -2.7802112, -5.2482443, -2.7802007, -2.3631063, 2.3528938
1: -10.1428947, -7.2638259, -10.1429062, -7.2638254, -2.8210506, 2.8461757
2: -5.5605116, -2.7213585, -5.5605159, -2.7213488, -2.8391628, 2.8391573
3: -12.1751680, -9.0126381, -12.1751766, -9.0126343, -3.0891528, 3.0947514
4: -8.7701921, -5.4502707, -8.7701921, -5.4502583, -3.2996125, 3.3150082
5: -0.9283382, 1.5568302, -0.9283434, 1.5568354, -2.4851737, 2.4851737
6: 5.1297626, 7.4612293, 5.1297588, 7.4612336, -2.3314710, 2.3314705
7: -18.8547363, -15.4136333, -18.8547440, -15.4136238, -3.0014167, 3.0133152
8: -1.6276655, 1.3615065, -1.6276751, 1.3615103, -2.9891758, 2.9891815
9: -8.8711100, -6.3979945, -8.8711119, -6.3979683, -2.4306700, 2.4013464

Time for backsubstitution: 12.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6196
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 6221
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 4584
type: B, layer: 1, pos: 4584
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6196

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6723778, upper bound: 1.6729596
time: 5.13 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6723777, upper bound: 1.6729576
time: 4.48 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -5.2224331, -2.8014963, -5.2902765, -2.7659867, -2.3525500, 2.3772264
1: -10.0900106, -7.2931690, -10.1563911, -7.2499199, -2.8106780, 2.8273549
2: -5.5347357, -2.7622347, -5.5968347, -2.6902475, -2.8444881, 2.8346000
3: -12.1475792, -9.0272722, -12.1940699, -8.9918013, -3.0826421, 3.0583789
4: -8.7499352, -5.4736094, -8.8223429, -5.4270535, -3.2732940, 3.2938452
5: -0.9145565, 1.5375628, -0.9656922, 1.5818973, -2.4964538, 2.5032549
6: 5.1479845, 7.4512987, 5.0871520, 7.5102048, -2.3622203, 2.3641467
7: -18.8300114, -15.4321060, -18.8811684, -15.3954306, -2.9919958, 3.0100517
8: -1.5962601, 1.3372712, -1.6486995, 1.3943911, -2.9906511, 2.9859707
9: -8.8377466, -6.4417171, -8.9225187, -6.3767781, -2.4252911, 2.4182663

Time for backsubstitution: 12.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 6196
type: A, layer: 1, pos: 5817
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 4584

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6221

## Relational analysis of IS_A1_B2_A1_A1

### Relational analysis result of IS_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6729413, upper bound: 1.6806341
time: 4.49 seconds

## Relational analysis of IS_A1_B2_A1_A2

### Relational analysis result of IS_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6729413, upper bound: 1.6882992
time: 4.56 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -5.2482414, -2.7802112, -5.2933302, -2.7619328, -2.3819089, 2.3969107
1: -10.1428947, -7.2638259, -10.1672659, -7.2483907, -2.8369870, 2.8692446
2: -5.5605116, -2.7213585, -5.5987654, -2.6825676, -2.8779440, 2.8774068
3: -12.1751680, -9.0126381, -12.1994400, -8.9904881, -3.1124544, 3.1195016
4: -8.7701921, -5.4502707, -8.8240805, -5.4225054, -3.3287826, 3.3605666
5: -0.9283382, 1.5568302, -0.9681324, 1.5843486, -2.5126867, 2.5249624
6: 5.1297626, 7.4612293, 5.0850186, 7.5114970, -2.3817344, 2.3762107
7: -18.8547363, -15.4136333, -18.8848648, -15.3919125, -3.0232263, 3.0461006
8: -1.6276655, 1.3615065, -1.6544285, 1.3965702, -3.0242357, 3.0159349
9: -8.8711100, -6.3979945, -8.9248171, -6.3680735, -2.4614861, 2.4331985

Time for backsubstitution: 12.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6196
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 5817
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 4584
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6196

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6723721, upper bound: 1.6888919
time: 5.10 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6723719, upper bound: 1.6888900
time: 4.38 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -5.2902765, -2.7659867, -5.2261763, -2.7979712, -2.3783083, 2.3678868
1: -10.1563911, -7.2499199, -10.0850601, -7.2934723, -2.8291473, 2.8202682
2: -5.5968347, -2.6902475, -5.5393858, -2.7615266, -2.8353081, 2.8491383
3: -12.1940699, -8.9918013, -12.1501522, -9.0328121, -3.0534444, 3.0875707
4: -8.8223429, -5.4270535, -8.7278633, -5.4583225, -3.3120894, 3.2751217
5: -0.9656922, 1.5818973, -0.9134789, 1.5448258, -2.5105181, 2.4953761
6: 5.0871520, 7.5102048, 5.1299200, 7.4380894, -2.3509374, 2.3802848
7: -18.8811684, -15.3954306, -18.8095360, -15.4330311, -3.0122056, 2.9710631
8: -1.6486995, 1.3943911, -1.5942032, 1.3479271, -2.9966266, 2.9885943
9: -8.9225187, -6.3767781, -8.8239946, -6.4226036, -2.4392629, 2.4203589

Time for backsubstitution: 12.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 5817
type: B, layer: 1, pos: 5817
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 4584
type: A, layer: 1, pos: 4584
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6221

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6806243, upper bound: 1.6812062
time: 4.38 seconds

## Relational analysis of IS_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6806243, upper bound: 1.6888796
time: 4.76 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -5.2902727, -2.7659910, -5.2590303, -2.7726371, -2.4046369, 2.4221954
1: -10.1563854, -7.2499251, -10.1229868, -7.2392216, -2.8961921, 2.8594065
2: -5.5968275, -2.6902494, -5.5829296, -2.7290707, -2.8677568, 2.8926802
3: -12.1940660, -8.9918127, -12.1981802, -9.0026217, -3.0856128, 3.1400323
4: -8.8223305, -5.4270568, -8.7824383, -5.4036455, -3.3501799, 3.3328855
5: -0.9656897, 1.5818921, -0.9382769, 1.5657346, -2.5314243, 2.5201690
6: 5.0871572, 7.5101986, 5.0794468, 7.4618492, -2.3746920, 2.4307518
7: -18.8811512, -15.3954344, -18.8728199, -15.3286438, -3.0726361, 3.0393109
8: -1.6486969, 1.3943877, -1.6200600, 1.3686619, -3.0173588, 3.0144477
9: -8.9225111, -6.3767815, -8.8664351, -6.3743620, -2.4608150, 2.4642076

Time for backsubstitution: 12.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 5817
type: B, layer: 1, pos: 5817
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 4584

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6221

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6882985, upper bound: 1.6812060
time: 4.35 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6882985, upper bound: 1.6888795
time: 4.40 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -5.2677588, -2.7832687, -5.2582245, -2.7679403, -2.3963175, 2.3808217
1: -10.1138887, -7.2779808, -10.1559544, -7.2571578, -2.8296113, 2.8609858
2: -5.5728846, -2.7235374, -5.5816522, -2.7165978, -2.8562868, 2.8581147
3: -12.1716137, -9.0051632, -12.1876163, -8.9962482, -3.1025901, 3.0774684
4: -8.8036900, -5.4463215, -8.7789125, -5.4294071, -3.3232288, 3.3046079
5: -0.9540322, 1.5650427, -0.9330264, 1.5743580, -2.5283902, 2.4980693
6: 5.1048355, 7.5015616, 5.1026163, 7.4659147, -2.3610792, 2.3989453
7: -18.8598080, -15.4105301, -18.8783245, -15.4095373, -3.0163212, 3.0254302
8: -1.6223791, 1.3721428, -1.6320291, 1.3817148, -3.0040939, 3.0041718
9: -8.8914909, -6.4121346, -8.8765793, -6.3745332, -2.4468737, 2.4309790

Time for backsubstitution: 12.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: A, layer: 1, pos: 5817
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 4584

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6221

## Relational analysis of IS_A2_B2_A1_A1

### Relational analysis result of IS_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6882984, upper bound: 1.6806348
time: 4.65 seconds

## Relational analysis of IS_A2_B2_A1_A2

### Relational analysis result of IS_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6882984, upper bound: 1.6882997
time: 4.67 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -5.2933240, -2.7619441, -5.2582245, -2.7679403, -2.4111495, 2.3979170
1: -10.1672506, -7.2483940, -10.1559544, -7.2571578, -2.8548503, 2.8678803
2: -5.5987620, -2.6825778, -5.5816522, -2.7165978, -2.8821642, 2.8990743
3: -12.1994286, -8.9904928, -12.1876163, -8.9962482, -3.1311631, 3.1271029
4: -8.8240776, -5.4225173, -8.7789125, -5.4294071, -3.3841071, 3.3563952
5: -0.9681258, 1.5843440, -0.9330264, 1.5743580, -2.5424838, 2.5173705
6: 5.0850210, 7.5114956, 5.1026163, 7.4659147, -2.3808937, 2.4088793
7: -18.8848591, -15.3919230, -18.8783245, -15.4095373, -3.0542212, 3.0607347
8: -1.6544187, 1.3965669, -1.6320291, 1.3817148, -3.0361335, 3.0285959
9: -8.9248161, -6.3680997, -8.8765793, -6.3745332, -2.4578629, 2.4481094

Time for backsubstitution: 12.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 5817
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 4584
type: B, layer: 1, pos: 4584
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6221

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6806332, upper bound: 1.6888787
time: 10.06 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6882979, upper bound: 1.6888796
time: 4.28 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 27.14 seconds
IS_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 27.14
Output dim: 6, lower bound: -1.6729470, upper bound: 1.6647027
IS_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 27.14
Output dim: 6, lower bound: -1.6729470, upper bound: 1.6723669
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 27.14
Output dim: 6, lower bound: -1.6723778, upper bound: 1.6729596
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 27.14
Output dim: 6, lower bound: -1.6723777, upper bound: 1.6729576
IS_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 27.14
Output dim: 6, lower bound: -1.6729413, upper bound: 1.6806341
IS_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 27.14
Output dim: 6, lower bound: -1.6729413, upper bound: 1.6882992
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 27.14
Output dim: 6, lower bound: -1.6723721, upper bound: 1.6888919
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 27.14
Output dim: 6, lower bound: -1.6723719, upper bound: 1.6888900
IS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 27.14
Output dim: 6, lower bound: -1.6806243, upper bound: 1.6812062
IS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 27.14
Output dim: 6, lower bound: -1.6806243, upper bound: 1.6888796
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 27.14
Output dim: 6, lower bound: -1.6882985, upper bound: 1.6812060
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 27.14
Output dim: 6, lower bound: -1.6882985, upper bound: 1.6888795
IS_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 27.14
Output dim: 6, lower bound: -1.6882984, upper bound: 1.6806348
IS_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 27.14
Output dim: 6, lower bound: -1.6882984, upper bound: 1.6882997
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 27.14
Output dim: 6, lower bound: -1.6806332, upper bound: 1.6888787
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 27.14
Output dim: 6, lower bound: -1.6882979, upper bound: 1.6888796

## BFS IS instance: IS_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -5.2161036, -2.8101668, -5.2451324, -2.7841792, -2.3230109, 2.3203011
1: -10.0724096, -7.2998371, -10.1323891, -7.2653742, -2.7737389, 2.7962418
2: -5.5182486, -2.7661290, -5.5586176, -2.7290115, -2.7892370, 2.7924886
3: -12.1381540, -9.0490913, -12.1701813, -9.0139275, -3.0502110, 3.0118668
4: -8.7192230, -5.4787874, -8.7685585, -5.4547720, -3.2125168, 3.2370162
5: -0.9091848, 1.5272841, -0.9259325, 1.5544188, -2.4636035, 2.4532166
6: 5.1556816, 7.4334130, 5.1317983, 7.4599595, -2.3042779, 2.3016148
7: -18.7861271, -15.4369545, -18.8512249, -15.4171371, -2.9243031, 2.9720535
8: -1.5904655, 1.3277726, -1.6220074, 1.3594251, -2.9498906, 2.9497800
9: -8.8184881, -6.4457726, -8.8688116, -6.4066362, -2.3737388, 2.3824987

Time for backsubstitution: 12.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 6196
type: A, layer: 1, pos: 5817
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 4584

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6221

## Relational analysis of IS_A1_B1_A1_A1_B1

### Relational analysis result of IS_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6652736, upper bound: 1.6646939
time: 4.79 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2

### Relational analysis result of IS_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6652736, upper bound: 1.6646952
time: 7.79 seconds

## BFS IS instance: IS_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -5.2489829, -2.7848151, -5.2451286, -2.7841847, -2.3771229, 2.3465104
1: -10.1104212, -7.2455597, -10.1323843, -7.2653785, -2.8125257, 2.8632960
2: -5.5617514, -2.7336950, -5.5586100, -2.7290151, -2.8327363, 2.8249149
3: -12.1861467, -9.0191727, -12.1701765, -9.0139389, -3.1026564, 3.0437336
4: -8.7736626, -5.4242344, -8.7685452, -5.4547734, -3.2701359, 3.2958565
5: -0.9339491, 1.5481961, -0.9259303, 1.5544121, -2.4883614, 2.4741263
6: 5.1053333, 7.4570303, 5.1318035, 7.4599524, -2.3546190, 2.3252268
7: -18.8491917, -15.3325748, -18.8512096, -15.4171410, -2.9923000, 3.0324242
8: -1.6163304, 1.3484097, -1.6220040, 1.3594203, -2.9757507, 2.9704137
9: -8.8607645, -6.3976526, -8.8688049, -6.4066372, -2.4174204, 2.4214149

Time for backsubstitution: 12.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6221
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 6196
type: A, layer: 1, pos: 5817
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 4584

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6221

## Relational analysis of IS_A1_B1_A1_A2_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6652736, upper bound: 1.6723673
time: 4.97 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2

### Relational analysis result of IS_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6652736, upper bound: 1.6723677
time: 5.05 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -5.2482414, -2.7802112, -5.2224331, -2.8014963, -2.3361740, 2.3377922
1: -10.1428947, -7.2638259, -10.0900106, -7.2931690, -2.8146038, 2.7961922
2: -5.5605116, -2.7213585, -5.5347357, -2.7622347, -2.7982769, 2.8133771
3: -12.1751680, -9.0126381, -12.1475792, -9.0272722, -3.0392752, 3.0608253
4: -8.7701921, -5.4502707, -8.7499352, -5.4736094, -3.2423897, 3.2505748
5: -0.9283382, 1.5568302, -0.9145565, 1.5375628, -2.4659009, 2.4713867
6: 5.1297626, 7.4612293, 5.1479845, 7.4512987, -2.3215361, 2.3132448
7: -18.8547363, -15.4136333, -18.8300114, -15.4321060, -2.9788132, 2.9759126
8: -1.6276655, 1.3615065, -1.5962601, 1.3372712, -2.9649367, 2.9577665
9: -8.8711100, -6.3979945, -8.8377466, -6.4417171, -2.3873110, 2.4036036

Time for backsubstitution: 12.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6221
type: A, layer: 1, pos: 6221
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: B, layer: 1, pos: 5817
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 4584

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6221

## Relational analysis of IS_A1_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6647034, upper bound: 1.6729473
time: 7.12 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6723672, upper bound: 1.6729494
time: 5.41 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -5.2482414, -2.7802112, -5.2482414, -2.7802112, -2.3528895, 2.3528891
1: -10.1428947, -7.2638259, -10.1428947, -7.2638259, -2.8210468, 2.8210473
2: -5.5605116, -2.7213585, -5.5605116, -2.7213585, -2.8391531, 2.8391531
3: -12.1751680, -9.0126381, -12.1751680, -9.0126381, -3.0891495, 3.0891495
4: -8.7701921, -5.4502707, -8.7701921, -5.4502707, -3.3149986, 3.3149981
5: -0.9283382, 1.5568302, -0.9283382, 1.5568302, -2.4851685, 2.4851685
6: 5.1297626, 7.4612293, 5.1297626, 7.4612293, -2.3314667, 2.3314667
7: -18.8547363, -15.4136333, -18.8547363, -15.4136333, -3.0133057, 3.0133052
8: -1.6276655, 1.3615065, -1.6276655, 1.3615065, -2.9891720, 2.9891720
9: -8.8711100, -6.3979945, -8.8711100, -6.3979945, -2.4013414, 2.4013414

Time for backsubstitution: 12.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 5817
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 4584
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6221

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6723674, upper bound: 1.6652833
time: 4.94 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6723674, upper bound: 1.6729473
time: 4.47 seconds

## BFS IS instance: IS_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -5.2161036, -2.8101668, -5.2902765, -2.7659867, -2.3417263, 2.3642335
1: -10.0724096, -7.2998371, -10.1563911, -7.2499199, -2.7897429, 2.8190885
2: -5.5182486, -2.7661290, -5.5968347, -2.6902475, -2.8280010, 2.8307056
3: -12.1381540, -9.0490913, -12.1940699, -8.9918013, -3.0734887, 3.0361919
4: -8.7192230, -5.4787874, -8.8223429, -5.4270535, -3.2415714, 3.2888532
5: -0.9091848, 1.5272841, -0.9656922, 1.5818973, -2.4910822, 2.4929762
6: 5.1556816, 7.4334130, 5.0871520, 7.5102048, -2.3545232, 2.3462610
7: -18.7861271, -15.4369545, -18.8811684, -15.3954306, -2.9460888, 3.0043001
8: -1.5904655, 1.3277726, -1.6486995, 1.3943911, -2.9848566, 2.9764721
9: -8.8184881, -6.4457726, -8.9225187, -6.3767781, -2.4044695, 2.4148221

Time for backsubstitution: 12.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 6196
type: A, layer: 1, pos: 5817
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 4584

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6221

## Relational analysis of IS_A1_B2_A1_A1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6652679, upper bound: 1.6806252
time: 4.47 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2

### Relational analysis result of IS_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6652679, upper bound: 1.6806266
time: 8.19 seconds

## BFS IS instance: IS_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -5.2489829, -2.7848151, -5.2902727, -2.7659910, -2.3958387, 2.3904424
1: -10.1104212, -7.2455597, -10.1563854, -7.2499251, -2.8285294, 2.8861418
2: -5.5617514, -2.7336950, -5.5968275, -2.6902494, -2.8715019, 2.8631325
3: -12.1861467, -9.0191727, -12.1940660, -8.9918127, -3.1259341, 3.0680599
4: -8.7736626, -5.4242344, -8.8223305, -5.4270568, -3.2991905, 3.3269882
5: -0.9339491, 1.5481961, -0.9656897, 1.5818921, -2.5158412, 2.5138857
6: 5.1053333, 7.4570303, 5.0871572, 7.5101986, -2.4048653, 2.3698730
7: -18.8491917, -15.3325748, -18.8811512, -15.3954344, -3.0140867, 3.0646956
8: -1.6163304, 1.3484097, -1.6486969, 1.3943877, -3.0107181, 2.9971066
9: -8.8607645, -6.3976526, -8.9225111, -6.3767815, -2.4481516, 2.4363432

Time for backsubstitution: 12.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 6196
type: A, layer: 1, pos: 5817
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 4584

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6221

## Relational analysis of IS_A1_B2_A1_A2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6652679, upper bound: 1.6882996
time: 4.81 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6652679, upper bound: 1.6883000
time: 5.04 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -5.2482414, -2.7802112, -5.2677588, -2.7832687, -2.3548746, 2.3820744
1: -10.1428947, -7.2638259, -10.1138887, -7.2779808, -2.8301468, 2.8188772
2: -5.5605116, -2.7213585, -5.5728846, -2.7235374, -2.8369741, 2.8515260
3: -12.1751680, -9.0126381, -12.1716137, -9.0051632, -3.0625801, 3.0852408
4: -8.7701921, -5.4502707, -8.8036900, -5.4463215, -3.2709112, 3.2996745
5: -0.9283382, 1.5568302, -0.9540322, 1.5650427, -2.4933810, 2.5108624
6: 5.1297626, 7.4612293, 5.1048355, 7.5015616, -2.3717990, 2.3563938
7: -18.8547363, -15.4136333, -18.8598080, -15.4105301, -3.0004897, 3.0081820
8: -1.6276655, 1.3615065, -1.6223791, 1.3721428, -2.9998083, 2.9838855
9: -8.8711100, -6.3979945, -8.8914909, -6.4121346, -2.4177973, 2.4219897

Time for backsubstitution: 13.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6221
type: A, layer: 1, pos: 6221
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: B, layer: 1, pos: 5817
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 4584

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6221

## Relational analysis of IS_A1_B2_A2_B1_B1

### Relational analysis result of IS_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6646976, upper bound: 1.6888798
time: 6.84 seconds

## Relational analysis of IS_A1_B2_A2_B1_B2

### Relational analysis result of IS_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6723614, upper bound: 1.6888818
time: 6.03 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -5.2482414, -2.7802112, -5.2933240, -2.7619441, -2.3716912, 2.3969059
1: -10.1428947, -7.2638259, -10.1672506, -7.2483940, -2.8369837, 2.8441157
2: -5.5605116, -2.7213585, -5.5987620, -2.6825778, -2.8779337, 2.8774035
3: -12.1751680, -9.0126381, -12.1994286, -8.9904928, -3.1124511, 3.1138144
4: -8.7701921, -5.4502707, -8.8240776, -5.4225173, -3.3441467, 3.3605618
5: -0.9283382, 1.5568302, -0.9681258, 1.5843440, -2.5126822, 2.5249560
6: 5.1297626, 7.4612293, 5.0850210, 7.5114956, -2.3817329, 2.3762083
7: -18.8547363, -15.4136333, -18.8848591, -15.3919230, -3.0351734, 3.0460901
8: -1.6276655, 1.3615065, -1.6544187, 1.3965669, -3.0242324, 3.0159252
9: -8.8711100, -6.3979945, -8.9248161, -6.3680997, -2.4321127, 2.4331937

Time for backsubstitution: 13.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 5817
type: A, layer: 1, pos: 5817
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 4584
type: B, layer: 1, pos: 4584
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6221

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6723617, upper bound: 1.6812156
time: 5.00 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6723617, upper bound: 1.6888796
time: 4.99 seconds

## BFS IS instance: IS_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -5.2838025, -2.7746513, -5.2261763, -2.7979712, -2.3673449, 2.3548713
1: -10.1389055, -7.2571144, -10.0850601, -7.2934723, -2.8084869, 2.8111372
2: -5.5803504, -2.6943059, -5.5393858, -2.7615266, -2.8188238, 2.8450799
3: -12.1847191, -9.0138416, -12.1501522, -9.0328121, -3.0442376, 3.0651774
4: -8.7917614, -5.4326897, -8.7278633, -5.4583225, -3.2807336, 3.2696018
5: -0.9598529, 1.5716417, -0.9134789, 1.5448258, -2.5046787, 2.4851205
6: 5.0969625, 7.4924631, 5.1299200, 7.4380894, -2.3411269, 2.3625431
7: -18.8369637, -15.4005337, -18.8095360, -15.4330311, -2.9661002, 2.9650002
8: -1.6419315, 1.3848343, -1.5942032, 1.3479271, -2.9898586, 2.9790375
9: -8.9033880, -6.3811445, -8.8239946, -6.4226036, -2.4184237, 2.4165318

Time for backsubstitution: 13.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 5817
type: B, layer: 1, pos: 5817
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 4584

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 481

## Relational analysis of IS_A2_B1_B1_A1_B1

### Relational analysis result of IS_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6806334, upper bound: 1.6652685
time: 4.53 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2

### Relational analysis result of IS_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6806344, upper bound: 1.6662648
time: 4.53 seconds

## BFS IS instance: IS_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -5.3163848, -2.7494216, -5.2261763, -2.7979712, -2.4044166, 2.3812044
1: -10.1772718, -7.2020931, -10.0850601, -7.2934723, -2.8476343, 2.8725920
2: -5.6238508, -2.6617050, -5.5393858, -2.7615266, -2.8623242, 2.8776808
3: -12.2329388, -8.9836197, -12.1501522, -9.0328121, -3.0966840, 3.0973554
4: -8.8463001, -5.3771024, -8.7278633, -5.4583225, -3.3372912, 3.3255544
5: -0.9854007, 1.5926565, -0.9134789, 1.5448258, -2.5302265, 2.5061355
6: 5.0436788, 7.5161247, 5.1299200, 7.4380894, -2.3944106, 2.3862047
7: -18.9002495, -15.2958918, -18.8095360, -15.4330311, -3.0312610, 3.0150349
8: -1.6690850, 1.4056177, -1.5942032, 1.3479271, -3.0170121, 2.9998209
9: -8.9458342, -6.3323441, -8.8239946, -6.4226036, -2.4611363, 2.4476089

Time for backsubstitution: 12.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 6196
type: B, layer: 1, pos: 5817
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 4584

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 481

## Relational analysis of IS_A2_B1_B1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6806334, upper bound: 1.6729419
time: 4.39 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6806344, upper bound: 1.6738931
time: 4.63 seconds

## BFS IS instance: IS_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -5.2838025, -2.7746513, -5.2590303, -2.7726371, -2.3936791, 2.3928750
1: -10.1389055, -7.2571144, -10.1229868, -7.2392216, -2.8692522, 2.8502817
2: -5.5803504, -2.6943059, -5.5829296, -2.7290707, -2.8512797, 2.8886237
3: -12.1847191, -9.0138416, -12.1981802, -9.0026217, -3.0764084, 3.1176481
4: -8.7917614, -5.4326897, -8.7824383, -5.4036455, -3.3185389, 3.3261871
5: -0.9598529, 1.5716417, -0.9382769, 1.5657346, -2.5255876, 2.5099187
6: 5.0969625, 7.4924631, 5.0794468, 7.4618492, -2.3648868, 2.4130163
7: -18.8369637, -15.4005337, -18.8728199, -15.3286438, -3.0264144, 3.0305834
8: -1.6419315, 1.3848343, -1.6200600, 1.3686619, -3.0105934, 3.0048943
9: -8.9033880, -6.3811445, -8.8664351, -6.3743620, -2.4399800, 2.4590437

Time for backsubstitution: 12.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 6196
type: B, layer: 1, pos: 5817
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 4584

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 481

## Relational analysis of IS_A2_B1_B2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6806243, upper bound: 1.6652676
time: 4.74 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6806254, upper bound: 1.6662662
time: 4.45 seconds

## BFS IS instance: IS_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -5.3163848, -2.7494216, -5.2590303, -2.7726371, -2.4427586, 2.4398823
1: -10.1772718, -7.2020931, -10.1229868, -7.2392216, -2.9163737, 2.9154899
2: -5.6238508, -2.6617050, -5.5829296, -2.7290707, -2.8947802, 2.9212246
3: -12.2329388, -8.9836197, -12.1981802, -9.0026217, -3.1288123, 3.1497829
4: -8.8463001, -5.3771024, -8.7824383, -5.4036455, -3.3750365, 3.3770032
5: -0.9854007, 1.5926565, -0.9382769, 1.5657346, -2.5511353, 2.5309334
6: 5.0436788, 7.5161247, 5.0794468, 7.4618492, -2.4181705, 2.4366779
7: -18.9002495, -15.2958918, -18.8728199, -15.3286438, -3.0918536, 3.0808496
8: -1.6690850, 1.4056177, -1.6200600, 1.3686619, -3.0377469, 3.0256777
9: -8.9458342, -6.3323441, -8.8664351, -6.3743620, -2.4826922, 2.4927680

Time for backsubstitution: 12.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 6196
type: B, layer: 1, pos: 5817
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 4584

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 481

## Relational analysis of IS_A2_B1_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6806243, upper bound: 1.6652684
time: 4.63 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6806254, upper bound: 1.6662645
time: 4.49 seconds

## BFS IS instance: IS_A2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -5.2614064, -2.7919488, -5.2582245, -2.7679403, -2.3854017, 2.3678434
1: -10.0963478, -7.2846117, -10.1559544, -7.2571578, -2.8087382, 2.8527603
2: -5.5563784, -2.7274427, -5.5816522, -2.7165978, -2.8397806, 2.8542094
3: -12.1622868, -9.0272360, -12.1876163, -8.9962482, -3.0934615, 3.0550013
4: -8.7731619, -5.4514956, -8.7789125, -5.4294071, -3.2914248, 3.2995996
5: -0.9487022, 1.5547765, -0.9330264, 1.5743580, -2.5230603, 2.4878030
6: 5.1125007, 7.4838295, 5.1026163, 7.4659147, -2.3534141, 2.3812132
7: -18.8156128, -15.4153891, -18.8783245, -15.4095373, -2.9700823, 3.0196648
8: -1.6165061, 1.3625689, -1.6320291, 1.3817148, -2.9982209, 2.9945979
9: -8.8723564, -6.4162254, -8.8765793, -6.3745332, -2.4260354, 2.4275224

Time for backsubstitution: 12.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: A, layer: 1, pos: 5817
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 4584

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6221

## Relational analysis of IS_A2_B2_A1_A1_B1

### Relational analysis result of IS_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6812045, upper bound: 1.6806247
time: 4.62 seconds

## Relational analysis of IS_A2_B2_A1_A1_B2

### Relational analysis result of IS_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6812045, upper bound: 1.6806261
time: 8.26 seconds

## BFS IS instance: IS_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -5.2939329, -2.7666664, -5.2582221, -2.7679448, -2.4256363, 2.3941631
1: -10.1346483, -7.2303486, -10.1559467, -7.2571616, -2.8477764, 2.9021177
2: -5.5999808, -2.6950133, -5.5816460, -2.7165997, -2.8833811, 2.8866327
3: -12.2102776, -8.9970169, -12.1876116, -8.9962587, -3.1457100, 3.0871954
4: -8.8276167, -5.3966913, -8.7789001, -5.4294119, -3.3492203, 3.3534498
5: -0.9735241, 1.5757519, -0.9330233, 1.5743513, -2.5478754, 2.5087752
6: 5.0620317, 7.5074506, 5.1026216, 7.4659090, -2.4038773, 2.4048290
7: -18.8790855, -15.3110418, -18.8783073, -15.4095402, -3.0385318, 3.0664959
8: -1.6424463, 1.3834391, -1.6320262, 1.3817105, -3.0241568, 3.0154653
9: -8.9147110, -6.3678966, -8.8765736, -6.3745360, -2.4697967, 2.4519544

Time for backsubstitution: 12.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: A, layer: 1, pos: 5817
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 4584

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6221

## Relational analysis of IS_A2_B2_A1_A2_B1

### Relational analysis result of IS_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6812045, upper bound: 1.6882989
time: 4.58 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2

### Relational analysis result of IS_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6812045, upper bound: 1.6882997
time: 5.17 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -5.2933240, -2.7619441, -5.2517452, -2.7765951, -2.3981042, 2.3869078
1: -10.1672506, -7.2483940, -10.1384811, -7.2643704, -2.8456755, 2.8471527
2: -5.5987620, -2.6825778, -5.5651875, -2.7206552, -2.8781068, 2.8826096
3: -12.1994286, -8.9904928, -12.1782436, -9.0181599, -3.1089010, 3.1178560
4: -8.8240776, -5.4225173, -8.7481613, -5.4350219, -3.3786445, 3.3256440
5: -0.9681258, 1.5843440, -0.9271373, 1.5640986, -2.5322244, 2.5114813
6: 5.0850210, 7.5114956, 5.1124811, 7.4480004, -2.3629794, 2.3990145
7: -18.8848591, -15.3919230, -18.8343201, -15.4146347, -3.0481505, 3.0147152
8: -1.6544187, 1.3965669, -1.6253157, 1.3721581, -3.0265768, 3.0218825
9: -8.9248161, -6.3680997, -8.8573236, -6.3788881, -2.4541066, 2.4273634

Time for backsubstitution: 12.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: B, layer: 1, pos: 5817
type: A, layer: 1, pos: 4584
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 4584
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6221

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6806241, upper bound: 1.6812062
time: 4.51 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6806241, upper bound: 1.6888796
time: 4.70 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -5.2933202, -2.7619476, -5.2846375, -2.7513275, -2.4244323, 2.4411647
1: -10.1672440, -7.2483978, -10.1764736, -7.2093625, -2.9133878, 2.8862429
2: -5.5987554, -2.6825805, -5.6086130, -2.6880355, -2.9107199, 2.9260325
3: -12.1994247, -8.9905014, -12.2264919, -8.9879665, -3.1410418, 3.1703868
4: -8.8240662, -5.4225187, -8.8028259, -5.3795376, -3.4103260, 3.3803072
5: -0.9681230, 1.5843381, -0.9526756, 1.5850611, -2.5531840, 2.5370135
6: 5.0850263, 7.5114889, 5.0591855, 7.4718199, -2.3867936, 2.4523034
7: -18.8848419, -15.3919258, -18.8973732, -15.3099594, -3.1044817, 3.0830150
8: -1.6544139, 1.3965621, -1.6524110, 1.3928280, -3.0472419, 3.0489731
9: -8.9248085, -6.3681016, -8.8998652, -6.3301525, -2.4760182, 2.4715118

Time for backsubstitution: 12.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 5817
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 4584
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 481

## Relational analysis of IS_A2_B2_A2_B2_B1

### Relational analysis result of IS_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6882979, upper bound: 1.6729418
time: 4.35 seconds

## Relational analysis of IS_A2_B2_A2_B2_B2

### Relational analysis result of IS_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6882989, upper bound: 1.6738932
time: 4.33 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 21.59 seconds
IS_A1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 21.59
Output dim: 6, lower bound: -1.6652736, upper bound: 1.6646939
IS_A1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 21.59
Output dim: 6, lower bound: -1.6652736, upper bound: 1.6646952
IS_A1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 21.59
Output dim: 6, lower bound: -1.6652736, upper bound: 1.6723673
IS_A1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 21.59
Output dim: 6, lower bound: -1.6652736, upper bound: 1.6723677
IS_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 21.59
Output dim: 6, lower bound: -1.6647034, upper bound: 1.6729473
IS_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 21.59
Output dim: 6, lower bound: -1.6723672, upper bound: 1.6729494
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.59
Output dim: 6, lower bound: -1.6723674, upper bound: 1.6652833
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.59
Output dim: 6, lower bound: -1.6723674, upper bound: 1.6729473
IS_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 21.59
Output dim: 6, lower bound: -1.6652679, upper bound: 1.6806252
IS_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 21.59
Output dim: 6, lower bound: -1.6652679, upper bound: 1.6806266
IS_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 21.59
Output dim: 6, lower bound: -1.6652679, upper bound: 1.6882996
IS_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 21.59
Output dim: 6, lower bound: -1.6652679, upper bound: 1.6883000
IS_A1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 21.59
Output dim: 6, lower bound: -1.6646976, upper bound: 1.6888798
IS_A1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 21.59
Output dim: 6, lower bound: -1.6723614, upper bound: 1.6888818
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.59
Output dim: 6, lower bound: -1.6723617, upper bound: 1.6812156
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.59
Output dim: 6, lower bound: -1.6723617, upper bound: 1.6888796
IS_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 21.59
Output dim: 6, lower bound: -1.6806334, upper bound: 1.6652685
IS_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 21.59
Output dim: 6, lower bound: -1.6806344, upper bound: 1.6662648
IS_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 21.59
Output dim: 6, lower bound: -1.6806334, upper bound: 1.6729419
IS_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 21.59
Output dim: 6, lower bound: -1.6806344, upper bound: 1.6738931
IS_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 21.59
Output dim: 6, lower bound: -1.6806243, upper bound: 1.6652676
IS_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 21.59
Output dim: 6, lower bound: -1.6806254, upper bound: 1.6662662
IS_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 21.59
Output dim: 6, lower bound: -1.6806243, upper bound: 1.6652684
IS_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 21.59
Output dim: 6, lower bound: -1.6806254, upper bound: 1.6662645
IS_A2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 21.59
Output dim: 6, lower bound: -1.6812045, upper bound: 1.6806247
IS_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 21.59
Output dim: 6, lower bound: -1.6812045, upper bound: 1.6806261
IS_A2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 21.59
Output dim: 6, lower bound: -1.6812045, upper bound: 1.6882989
IS_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 21.59
Output dim: 6, lower bound: -1.6812045, upper bound: 1.6882997
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 21.59
Output dim: 6, lower bound: -1.6806241, upper bound: 1.6812062
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.59
Output dim: 6, lower bound: -1.6806241, upper bound: 1.6888796
IS_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 21.59
Output dim: 6, lower bound: -1.6882979, upper bound: 1.6729418
IS_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 21.59
Output dim: 6, lower bound: -1.6882989, upper bound: 1.6738932

## BFS IS instance: IS_A1_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -5.2161036, -2.8101668, -5.2387247, -2.7928352, -2.3100348, 2.3094020
1: -10.0724096, -7.2998371, -10.1148434, -7.2723866, -2.7648253, 2.7755208
2: -5.5182486, -2.7661290, -5.5421543, -2.7330022, -2.7852464, 2.7760253
3: -12.1381540, -9.0490913, -12.1607485, -9.0357170, -3.0280952, 3.0026691
4: -8.7192230, -5.4787874, -8.7377930, -5.4602222, -3.2071781, 3.2054563
5: -0.9091848, 1.5272841, -0.9202315, 1.5441492, -2.4533339, 2.4475155
6: 5.1556816, 7.4334130, 5.1408463, 7.4420891, -2.2864075, 2.2925668
7: -18.7861271, -15.4369545, -18.8073292, -15.4221373, -2.9183512, 2.9262352
8: -1.5904655, 1.3277726, -1.6156371, 1.3499217, -2.9403872, 2.9434097
9: -8.8184881, -6.4457726, -8.8495579, -6.4108772, -2.3700066, 2.3617711

Time for backsubstitution: 12.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 6196
type: A, layer: 1, pos: 5817
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 4584

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4558

## Relational analysis of IS_A1_B1_A1_A1_B1_B1

### Relational analysis result of IS_A1_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6606591, upper bound: 1.6646521
time: 4.69 seconds

## Relational analysis of IS_A1_B1_A1_A1_B1_B2

### Relational analysis result of IS_A1_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6652561, upper bound: 1.6646857
time: 4.36 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -5.2161036, -2.8101668, -5.2716031, -2.7675383, -2.3362608, 2.3468456
1: -10.0724096, -7.2998371, -10.1529331, -7.2175641, -2.8260822, 2.8144660
2: -5.5182486, -2.7661290, -5.5855556, -2.7004545, -2.8177941, 2.8194265
3: -12.1381540, -9.0490913, -12.2089539, -9.0057898, -3.0599504, 3.0551782
4: -8.7192230, -5.4787874, -8.7923183, -5.4050560, -3.2663860, 3.2620249
5: -0.9091848, 1.5272841, -0.9455535, 1.5651026, -2.4742875, 2.4728377
6: 5.1556816, 7.4334130, 5.0884647, 7.4657230, -2.3100414, 2.3449483
7: -18.7861271, -15.4369545, -18.8702106, -15.3175535, -2.9783978, 2.9909844
8: -1.5904655, 1.3277726, -1.6423893, 1.3704858, -2.9609513, 2.9701619
9: -8.8184881, -6.4457726, -8.8919277, -6.3623738, -2.4035916, 2.4043839

Time for backsubstitution: 12.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 5817
type: A, layer: 1, pos: 5817
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 4584

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4558

## Relational analysis of IS_A1_B1_A1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6606591, upper bound: 1.6646516
time: 6.61 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2_B2

### Relational analysis result of IS_A1_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6652561, upper bound: 1.6646885
time: 5.14 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -5.2489829, -2.7848151, -5.2387247, -2.7928352, -2.3478408, 2.3356164
1: -10.1104212, -7.2455597, -10.1148434, -7.2723866, -2.8036170, 2.8362956
2: -5.5617514, -2.7336950, -5.5421543, -2.7330022, -2.8287492, 2.8084593
3: -12.1861467, -9.0191727, -12.1607485, -9.0357170, -3.0805502, 3.0345378
4: -8.7736626, -5.4242344, -8.7377930, -5.4602222, -3.2636180, 3.2643137
5: -0.9339491, 1.5481961, -0.9202315, 1.5441492, -2.4780984, 2.4684277
6: 5.1053333, 7.4570303, 5.1408463, 7.4420891, -2.3367558, 2.3161840
7: -18.8491917, -15.3325748, -18.8073292, -15.4221373, -2.9836855, 2.9865019
8: -1.6163304, 1.3484097, -1.6156371, 1.3499217, -2.9662521, 2.9640467
9: -8.8607645, -6.3976526, -8.8495579, -6.4108772, -2.4123521, 2.4006033

Time for backsubstitution: 12.74 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=6, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=2.363316535949707
rel_dist={6: [-1.6889095697923429, 1.6889090371909665]}

## Binary search (step 1) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 481
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 6196
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 6221
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 4584
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 481

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5461592, upper bound: 1.5620941
time: 4.01 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5620923, upper bound: 1.5620939
time: 4.15 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 8.35 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 8.35
Output dim: 6, lower bound: -1.5461592, upper bound: 1.5620941
IS_A2, status: Status.UNKNOWN, split count: 1, time: 8.35
Output dim: 6, lower bound: -1.5620923, upper bound: 1.5620939

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -5.2482443, -2.7802007, -5.2582340, -2.7679250, -2.2534657, 2.2537079
1: -10.1429062, -7.2638254, -10.1559734, -7.2571516, -2.7071733, 2.7131982
2: -5.5605159, -2.7213488, -5.5816660, -2.7165861, -2.8439298, 2.8603172
3: -12.1751766, -9.0126343, -12.1876364, -8.9962378, -2.9707427, 2.9674430
4: -8.7701921, -5.4502583, -8.7789192, -5.4293895, -3.1633549, 3.1573048
5: -0.9283434, 1.5568354, -0.9330347, 1.5743690, -2.5027122, 2.4898701
6: 5.1297588, 7.4612336, 5.1026025, 7.4659190, -2.3185101, 2.3412559
7: -18.8547440, -15.4136238, -18.8783417, -15.4095268, -2.8228755, 2.8418846
8: -1.6276751, 1.3615103, -1.6320400, 1.3817253, -3.0094004, 2.9935503
9: -8.8711119, -6.3979683, -8.8765841, -6.3745003, -2.3304386, 2.3155546

Time for backsubstitution: 12.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6196
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 6221
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 4584

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6196

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5461592, upper bound: 1.5618107
time: 4.15 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5461589, upper bound: 1.5620938
time: 4.27 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -5.2933302, -2.7619328, -5.2582293, -2.7679298, -2.2974706, 2.2808132
1: -10.1672659, -7.2483907, -10.1559629, -7.2571559, -2.7302370, 2.7421918
2: -5.5987654, -2.6825676, -5.5816512, -2.7165890, -2.8821764, 2.8990836
3: -12.1994400, -8.9904881, -12.1876249, -8.9962482, -2.9954824, 2.9908409
4: -8.8240805, -5.4225054, -8.7789125, -5.4293985, -3.2124515, 3.1973310
5: -0.9681324, 1.5843486, -0.9330314, 1.5743585, -2.5424910, 2.5173800
6: 5.0850186, 7.5114970, 5.1026182, 7.4659157, -2.3652425, 2.3638031
7: -18.8848648, -15.3919125, -18.8783226, -15.4095297, -2.8558292, 2.8636780
8: -1.6544285, 1.3965702, -1.6320386, 1.3817148, -3.0361433, 3.0286088
9: -8.9248171, -6.3680735, -8.8765821, -6.3745136, -2.3614111, 2.3471947

Time for backsubstitution: 12.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6196
type: A, layer: 1, pos: 6196
type: B, layer: 1, pos: 6221
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6196

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5618089, upper bound: 1.5620931
time: 4.53 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5620920, upper bound: 1.5620937
time: 4.08 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 21.48 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 21.48
Output dim: 6, lower bound: -1.5461592, upper bound: 1.5618107
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 21.48
Output dim: 6, lower bound: -1.5461589, upper bound: 1.5620938
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 21.48
Output dim: 6, lower bound: -1.5618089, upper bound: 1.5620931
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 21.48
Output dim: 6, lower bound: -1.5620920, upper bound: 1.5620937

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -5.2224331, -2.8014963, -5.2536955, -2.7738168, -2.2222691, 2.2223947
1: -10.0900106, -7.2931690, -10.1402311, -7.2594242, -2.6547098, 2.6666036
2: -5.5347357, -2.7622347, -5.5788741, -2.7278290, -2.8069067, 2.8166394
3: -12.1475792, -9.0272722, -12.1798868, -8.9981575, -2.9348598, 2.9048476
4: -8.7499352, -5.4736094, -8.7764006, -5.4359994, -3.0980167, 3.0895762
5: -0.9145565, 1.5375628, -0.9294628, 1.5707810, -2.4853375, 2.4670258
6: 5.1479845, 7.4512987, 5.1056476, 7.4640212, -2.2981739, 2.3456511
7: -18.8300114, -15.4321060, -18.8730392, -15.4146700, -2.7848883, 2.8116360
8: -1.5962601, 1.3372712, -1.6236932, 1.3786340, -2.9748940, 2.9609644
9: -8.8377466, -6.4417171, -8.8732319, -6.3872190, -2.2876778, 2.2681363

Time for backsubstitution: 12.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 6196
type: A, layer: 1, pos: 4584
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 4584

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 481

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5461592, upper bound: 1.5458794
time: 4.40 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5461592, upper bound: 1.5618107
time: 4.02 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -5.2482414, -2.7802112, -5.2582359, -2.7679257, -2.2534599, 2.2421298
1: -10.1428947, -7.2638259, -10.1559744, -7.2571521, -2.6786947, 2.7131934
2: -5.5605116, -2.7213585, -5.5816655, -2.7165873, -2.8439243, 2.8603070
3: -12.1751680, -9.0126381, -12.1876345, -8.9962387, -2.9642229, 2.9674397
4: -8.7701921, -5.4502707, -8.7789192, -5.4293904, -3.1567607, 3.1629605
5: -0.9283382, 1.5568302, -0.9330350, 1.5743685, -2.5027065, 2.4898653
6: 5.1297626, 7.4612293, 5.1026030, 7.4659185, -2.3217921, 2.3392215
7: -18.8547363, -15.4136333, -18.8783417, -15.4095287, -2.8176155, 2.8465381
8: -1.6276655, 1.3615065, -1.6320410, 1.3817248, -3.0093904, 2.9935474
9: -8.8711100, -6.3979945, -8.8765841, -6.3745012, -2.3277779, 2.2801299

Time for backsubstitution: 12.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 4584
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 4584

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 481

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5461589, upper bound: 1.5461641
time: 4.27 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5461589, upper bound: 1.5620938
time: 4.41 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -5.2888408, -2.7678273, -5.2325306, -2.7892942, -2.2662358, 2.2496448
1: -10.1514978, -7.2506413, -10.1026011, -7.2868237, -2.6831365, 2.6892948
2: -5.5959358, -2.6937864, -5.5558739, -2.7576232, -2.8383126, 2.8620875
3: -12.1917906, -8.9924145, -12.1595306, -9.0108700, -2.9331198, 2.9541831
4: -8.8215637, -5.4291391, -8.7585564, -5.4531546, -3.1475368, 3.1319675
5: -0.9645835, 1.5807486, -0.9188528, 1.5550904, -2.5196738, 2.4996014
6: 5.0881305, 7.5096035, 5.1222115, 7.4559803, -2.3678498, 2.3421843
7: -18.8794956, -15.3970509, -18.8535328, -15.4281778, -2.8253922, 2.8254819
8: -1.6460779, 1.3934045, -1.6000094, 1.3574438, -3.0035217, 2.9934139
9: -8.9214840, -6.3808112, -8.8432522, -6.4185290, -2.3138661, 2.3041365

Time for backsubstitution: 12.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6221
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 6196
type: B, layer: 1, pos: 5817
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 4584

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6221

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5541047, upper bound: 1.5620823
time: 5.88 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5617974, upper bound: 1.5620819
time: 4.54 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -5.2933288, -2.7619345, -5.2582245, -2.7679408, -2.2858939, 2.2808082
1: -10.1672621, -7.2483907, -10.1559505, -7.2571592, -2.7302318, 2.7137113
2: -5.5987654, -2.6825697, -5.5816483, -2.7165999, -2.8821654, 2.8990786
3: -12.1994390, -8.9904890, -12.1876154, -8.9962502, -2.9954786, 2.9840865
4: -8.8240805, -5.4225063, -8.7789097, -5.4294105, -3.2113166, 3.1907382
5: -0.9681313, 1.5843482, -0.9330250, 1.5743546, -2.5424860, 2.5173731
6: 5.0850172, 7.5114956, 5.1026211, 7.4659138, -2.3632059, 2.3639653
7: -18.8848648, -15.3919125, -18.8783188, -15.4095383, -2.8604031, 2.8583493
8: -1.6544261, 1.3965702, -1.6320276, 1.3817120, -3.0361381, 3.0285978
9: -8.9248180, -6.3680782, -8.8765783, -6.3745384, -2.3256655, 2.3445420

Time for backsubstitution: 12.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6221
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 6196
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 5817
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 444

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6221

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5544199, upper bound: 1.5620820
time: 4.03 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5620805, upper bound: 1.5620826
time: 4.31 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 21.18 seconds
IS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 21.18
Output dim: 6, lower bound: -1.5461592, upper bound: 1.5458794
IS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 21.18
Output dim: 6, lower bound: -1.5461592, upper bound: 1.5618107
IS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 21.18
Output dim: 6, lower bound: -1.5461589, upper bound: 1.5461641
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 21.18
Output dim: 6, lower bound: -1.5461589, upper bound: 1.5620938
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 21.18
Output dim: 6, lower bound: -1.5541047, upper bound: 1.5620823
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 21.18
Output dim: 6, lower bound: -1.5617974, upper bound: 1.5620819
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 21.18
Output dim: 6, lower bound: -1.5544199, upper bound: 1.5620820
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 21.18
Output dim: 6, lower bound: -1.5620805, upper bound: 1.5620826

## BFS IS instance: IS_A1_A1_B1

### Backsubstitution after applying IS history:
0: -5.2224331, -2.8014963, -5.2436676, -2.7860188, -2.2081070, 2.2081721
1: -10.0900106, -7.2931690, -10.1275167, -7.2661057, -2.6439838, 2.6502190
2: -5.5347357, -2.7622347, -5.5577326, -2.7325571, -2.8021786, 2.7954979
3: -12.1475792, -9.0272722, -12.1678715, -9.0145378, -2.9175243, 2.8913131
4: -8.7499352, -5.4736094, -8.7677803, -5.4568429, -3.0746007, 3.0724478
5: -0.9145565, 1.5375628, -0.9248191, 1.5532694, -2.4678259, 2.4623818
6: 5.1479845, 7.4512987, 5.1327534, 7.4593544, -2.2934237, 2.3185453
7: -18.8300114, -15.4321060, -18.8495674, -15.4187622, -2.7784905, 2.7867460
8: -1.5962601, 1.3372712, -1.6193886, 1.3584561, -2.9547162, 2.9566598
9: -8.8377466, -6.4417171, -8.8677711, -6.4106560, -2.2626915, 2.2582417

Time for backsubstitution: 12.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 6196
type: A, layer: 1, pos: 5817
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 4584

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6221

## Relational analysis of IS_A1_A1_B1_A1

### Relational analysis result of IS_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5461481, upper bound: 1.5381754
time: 4.41 seconds

## Relational analysis of IS_A1_A1_B1_A2

### Relational analysis result of IS_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5461481, upper bound: 1.5458680
time: 4.37 seconds

## BFS IS instance: IS_A1_A1_B2

### Backsubstitution after applying IS history:
0: -5.2224331, -2.8014963, -5.2888408, -2.7678273, -2.2268219, 2.2521389
1: -10.0900106, -7.2931690, -10.1514978, -7.2506413, -2.6599970, 2.6730390
2: -5.5347357, -2.7622347, -5.5959358, -2.6937864, -2.8409493, 2.8337011
3: -12.1475792, -9.0272722, -12.1917906, -8.9924145, -2.9407992, 2.9156680
4: -8.7499352, -5.4736094, -8.8215637, -5.4291391, -3.1036482, 3.1242857
5: -0.9145565, 1.5375628, -0.9645835, 1.5807486, -2.4953051, 2.5021462
6: 5.1479845, 7.4512987, 5.0881305, 7.5096035, -2.3160548, 2.3631682
7: -18.8300114, -15.4321060, -18.8794956, -15.3970509, -2.8003359, 2.8189592
8: -1.5962601, 1.3372712, -1.6460779, 1.3934045, -2.9896646, 2.9833491
9: -8.8377466, -6.4417171, -8.9214840, -6.3808112, -2.2934127, 2.2894037

Time for backsubstitution: 12.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 6196
type: A, layer: 1, pos: 5817
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 4584

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6221

## Relational analysis of IS_A1_A1_B2_A1

### Relational analysis result of IS_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5461481, upper bound: 1.5541064
time: 4.32 seconds

## Relational analysis of IS_A1_A1_B2_A2

### Relational analysis result of IS_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5461481, upper bound: 1.5617991
time: 4.40 seconds

## BFS IS instance: IS_A1_A2_B1

### Backsubstitution after applying IS history:
0: -5.2482414, -2.7802112, -5.2482462, -2.7802019, -2.2392087, 2.2276368
1: -10.1428947, -7.2638259, -10.1429052, -7.2638235, -2.6679568, 2.6964312
2: -5.5605116, -2.7213585, -5.5605154, -2.7213490, -2.8391626, 2.8391569
3: -12.1751680, -9.0126381, -12.1751757, -9.0126343, -2.9468689, 2.9533825
4: -8.7701921, -5.4502707, -8.7701941, -5.4502583, -3.1331830, 3.1454430
5: -0.9283382, 1.5568302, -0.9283432, 1.5568340, -2.4851723, 2.4851732
6: 5.1297626, 7.4612293, 5.1297588, 7.4612312, -2.3170223, 2.3117089
7: -18.8547363, -15.4136333, -18.8547401, -15.4136276, -2.8111825, 2.8209634
8: -1.6276655, 1.3615065, -1.6276741, 1.3615093, -2.9891748, 2.9891806
9: -8.8711100, -6.3979945, -8.8711138, -6.3979712, -2.3030472, 2.2701833

Time for backsubstitution: 12.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 6196
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 5817
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 5817
type: A, layer: 1, pos: 4584
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 444

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6221

## Relational analysis of IS_A1_A2_B1_A1

### Relational analysis result of IS_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5461479, upper bound: 1.5384907
time: 4.76 seconds

## Relational analysis of IS_A1_A2_B1_A2

### Relational analysis result of IS_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5461479, upper bound: 1.5461527
time: 4.27 seconds

## BFS IS instance: IS_A1_A2_B2

### Backsubstitution after applying IS history:
0: -5.2482414, -2.7802112, -5.2933288, -2.7619345, -2.2580123, 2.2716537
1: -10.1428947, -7.2638259, -10.1672621, -7.2483907, -2.6838942, 2.7194996
2: -5.5605116, -2.7213585, -5.5987654, -2.6825697, -2.8779418, 2.8774068
3: -12.1751680, -9.0126381, -12.1994390, -8.9904890, -2.9701691, 2.9781327
4: -8.7701921, -5.4502707, -8.8240805, -5.4225063, -3.1623530, 3.1877718
5: -0.9283382, 1.5568302, -0.9681313, 1.5843482, -2.5126863, 2.5249615
6: 5.1297626, 7.4612293, 5.0850172, 7.5114956, -2.3364496, 2.3579326
7: -18.8547363, -15.4136333, -18.8848648, -15.3919125, -2.8329916, 2.8537493
8: -1.6276655, 1.3615065, -1.6544261, 1.3965702, -3.0242357, 3.0159326
9: -8.8711100, -6.3979945, -8.9248180, -6.3680782, -2.3338640, 2.3009970

Time for backsubstitution: 12.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 4584

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6221

## Relational analysis of IS_A1_A2_B2_A1

### Relational analysis result of IS_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5461479, upper bound: 1.5544218
time: 4.52 seconds

## Relational analysis of IS_A1_A2_B2_A2

### Relational analysis result of IS_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5461479, upper bound: 1.5620824
time: 4.18 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -5.2888408, -2.7678273, -5.2261748, -2.7979717, -2.2532182, 2.2387307
1: -10.1514978, -7.2506413, -10.0850582, -7.2934737, -2.6748295, 2.6685090
2: -5.5959358, -2.6937864, -5.5393820, -2.7615275, -2.8344083, 2.8455956
3: -12.1917906, -8.9924145, -12.1501522, -9.0328150, -2.9107308, 2.9449944
4: -8.8215637, -5.4291391, -8.7278614, -5.4583254, -3.1425257, 3.1001682
5: -0.9645835, 1.5807486, -0.9134784, 1.5448227, -2.5094061, 2.4942269
6: 5.0881305, 7.5096035, 5.1299238, 7.4380884, -2.3499579, 2.3345902
7: -18.8794956, -15.3970509, -18.8095284, -15.4330320, -2.8196363, 2.7793984
8: -1.6460779, 1.3934045, -1.5942025, 1.3479252, -2.9940031, 2.9876070
9: -8.9214840, -6.3808112, -8.8239956, -6.4226065, -2.3103991, 2.2832582

Time for backsubstitution: 12.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 6196
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 4584

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6221

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5540978, upper bound: 1.5544143
time: 4.95 seconds

## Relational analysis of IS_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5540978, upper bound: 1.5620823
time: 4.85 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -5.2888374, -2.7678323, -5.2590275, -2.7726381, -2.2795448, 2.2917843
1: -10.1514883, -7.2506461, -10.1229849, -7.2392240, -2.7389398, 2.7076449
2: -5.5959282, -2.6937885, -5.5829248, -2.7290716, -2.8668566, 2.8891363
3: -12.1917830, -8.9924278, -12.1981773, -9.0026245, -2.9428930, 2.9974508
4: -8.8215485, -5.4291430, -8.7824373, -5.4036484, -3.1774182, 3.1567783
5: -0.9645795, 1.5807396, -0.9382758, 1.5657318, -2.5303111, 2.5190153
6: 5.0881376, 7.5095949, 5.0794506, 7.4618497, -2.3737121, 2.3726163
7: -18.8794727, -15.3970537, -18.8728142, -15.3286428, -2.8766351, 2.8450212
8: -1.6460745, 1.3933992, -1.6200590, 1.3686581, -3.0147326, 3.0134583
9: -8.9214754, -6.3808141, -8.8664341, -6.3743649, -2.3319492, 2.3257923

Time for backsubstitution: 12.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 6196
type: B, layer: 1, pos: 5817
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 4584

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6221

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5617978, upper bound: 1.5544144
time: 5.49 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5617978, upper bound: 1.5620821
time: 4.64 seconds

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: -5.2933288, -2.7619345, -5.2517414, -2.7765985, -2.2728491, 2.2697976
1: -10.1672621, -7.2483907, -10.1384773, -7.2643700, -2.7210588, 2.6929827
2: -5.5987654, -2.6825697, -5.5651836, -2.7206559, -2.8781095, 2.8826139
3: -12.1994390, -8.9904890, -12.1782389, -9.0181637, -2.9732161, 2.9748397
4: -8.8240805, -5.4225063, -8.7481575, -5.4350233, -3.2058539, 3.1590500
5: -0.9681313, 1.5843482, -0.9271365, 1.5640960, -2.5322273, 2.5114846
6: 5.0850172, 7.5114956, 5.1124864, 7.4479995, -2.3443189, 2.3543768
7: -18.8848648, -15.3919125, -18.8343143, -15.4146347, -2.8543310, 2.8122520
8: -1.6544261, 1.3965702, -1.6253161, 1.3721547, -3.0265808, 3.0218863
9: -8.9248180, -6.3680782, -8.8573236, -6.3788934, -2.3219090, 2.3237119

Time for backsubstitution: 12.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 6196
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6221

## Relational analysis of IS_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5544130, upper bound: 1.5544141
time: 4.62 seconds

## Relational analysis of IS_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5544130, upper bound: 1.5620820
time: 4.04 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: -5.2933245, -2.7619419, -5.2846365, -2.7513292, -2.2991762, 2.3227983
1: -10.1672544, -7.2483964, -10.1764727, -7.2093635, -2.7770672, 2.7320709
2: -5.5987554, -2.6825728, -5.6086111, -2.6880355, -2.9107199, 2.9260383
3: -12.1994314, -8.9905043, -12.2264900, -8.9879684, -3.0053496, 3.0273652
4: -8.8240633, -5.4225092, -8.8028259, -5.3795390, -3.2375312, 3.2156715
5: -0.9681270, 1.5843402, -0.9526744, 1.5850587, -2.5531857, 2.5370145
6: 5.0850239, 7.5114899, 5.0591898, 7.4718199, -2.3691468, 2.3950560
7: -18.8848419, -15.3919182, -18.8973675, -15.3099632, -2.9071879, 2.8776369
8: -1.6544216, 1.3965650, -1.6524096, 1.3928242, -3.0472457, 3.0489745
9: -8.9248095, -6.3680801, -8.8998642, -6.3301563, -2.3438187, 2.3665705

Time for backsubstitution: 12.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 5817
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 481

## Relational analysis of IS_A2_B2_B2_B1

### Relational analysis result of IS_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5620805, upper bound: 1.5461485
time: 4.10 seconds

## Relational analysis of IS_A2_B2_B2_B2

### Relational analysis result of IS_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5620815, upper bound: 1.5470678
time: 4.14 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 21.06 seconds
IS_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 21.06
Output dim: 6, lower bound: -1.5461481, upper bound: 1.5381754
IS_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 21.06
Output dim: 6, lower bound: -1.5461481, upper bound: 1.5458680
IS_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 21.06
Output dim: 6, lower bound: -1.5461481, upper bound: 1.5541064
IS_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 21.06
Output dim: 6, lower bound: -1.5461481, upper bound: 1.5617991
IS_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 21.06
Output dim: 6, lower bound: -1.5461479, upper bound: 1.5384907
IS_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 21.06
Output dim: 6, lower bound: -1.5461479, upper bound: 1.5461527
IS_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 21.06
Output dim: 6, lower bound: -1.5461479, upper bound: 1.5544218
IS_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 21.06
Output dim: 6, lower bound: -1.5461479, upper bound: 1.5620824
IS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 21.06
Output dim: 6, lower bound: -1.5540978, upper bound: 1.5544143
IS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 21.06
Output dim: 6, lower bound: -1.5540978, upper bound: 1.5620823
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 21.06
Output dim: 6, lower bound: -1.5617978, upper bound: 1.5544144
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 21.06
Output dim: 6, lower bound: -1.5617978, upper bound: 1.5620821
IS_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 21.06
Output dim: 6, lower bound: -1.5544130, upper bound: 1.5544141
IS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 21.06
Output dim: 6, lower bound: -1.5544130, upper bound: 1.5620820
IS_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 21.06
Output dim: 6, lower bound: -1.5620805, upper bound: 1.5461485
IS_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 21.06
Output dim: 6, lower bound: -1.5620815, upper bound: 1.5470678

## BFS IS instance: IS_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -5.2161036, -2.8101668, -5.2436676, -2.7860188, -2.1972833, 2.1951790
1: -10.0724096, -7.2998371, -10.1275167, -7.2661057, -2.6230488, 2.6419516
2: -5.5182486, -2.7661290, -5.5577326, -2.7325571, -2.7856915, 2.7916036
3: -12.1381540, -9.0490913, -12.1678715, -9.0145378, -2.9083710, 2.8691258
4: -8.7192230, -5.4787874, -8.7677803, -5.4568429, -3.0428782, 3.0674558
5: -0.9091848, 1.5272841, -0.9248191, 1.5532694, -2.4624543, 2.4521031
6: 5.1556816, 7.4334130, 5.1327534, 7.4593544, -2.2858520, 2.3006597
7: -18.7861271, -15.4369545, -18.8495674, -15.4187622, -2.7325826, 2.7809939
8: -1.5904655, 1.3277726, -1.6193886, 1.3584561, -2.9489217, 2.9471612
9: -8.8184881, -6.4457726, -8.8677711, -6.4106560, -2.2418704, 2.2546048

Time for backsubstitution: 12.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 4584

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6221

## Relational analysis of IS_A1_A1_B1_A1_B1

### Relational analysis result of IS_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5384834, upper bound: 1.5381679
time: 4.44 seconds

## Relational analysis of IS_A1_A1_B1_A1_B2

### Relational analysis result of IS_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5384834, upper bound: 1.5381683
time: 4.98 seconds

## BFS IS instance: IS_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -5.2489829, -2.7848151, -5.2436619, -2.7860260, -2.2501402, 2.2213860
1: -10.1104212, -7.2455597, -10.1275082, -7.2661123, -2.6618328, 2.7058749
2: -5.5617514, -2.7336950, -5.5577230, -2.7325602, -2.8291912, 2.8240280
3: -12.1861467, -9.0191727, -12.1678648, -9.0145531, -2.9608140, 2.9009867
4: -8.7736626, -5.4242344, -8.7677641, -5.4568453, -3.0993428, 3.1262884
5: -0.9339491, 1.5481961, -0.9248147, 1.5532618, -2.4872108, 2.4730108
6: 5.1053333, 7.4570303, 5.1327600, 7.4593468, -2.3383770, 2.3242702
7: -18.8491917, -15.3325748, -18.8495464, -15.4187660, -2.7979579, 2.8379312
8: -1.6163304, 1.3484097, -1.6193814, 1.3584509, -2.9747813, 2.9677911
9: -8.8607645, -6.3976526, -8.8677616, -6.4106593, -2.2842379, 2.2925570

Time for backsubstitution: 12.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6221
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 4584

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6221

## Relational analysis of IS_A1_A1_B1_A2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5384834, upper bound: 1.5458679
time: 4.29 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2

### Relational analysis result of IS_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5384834, upper bound: 1.5458678
time: 4.71 seconds

## BFS IS instance: IS_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -5.2161036, -2.8101668, -5.2888408, -2.7678273, -2.2159982, 2.2391458
1: -10.0724096, -7.2998371, -10.1514978, -7.2506413, -2.6390610, 2.6647716
2: -5.5182486, -2.7661290, -5.5959358, -2.6937864, -2.8244622, 2.8298068
3: -12.1381540, -9.0490913, -12.1917906, -8.9924145, -2.9316459, 2.8934808
4: -8.7192230, -5.4787874, -8.8215637, -5.4291391, -3.0719256, 3.1192942
5: -0.9091848, 1.5272841, -0.9645835, 1.5807486, -2.4899335, 2.4918675
6: 5.1556816, 7.4334130, 5.0881305, 7.5096035, -2.3085480, 2.3452826
7: -18.7861271, -15.4369545, -18.8794956, -15.3970509, -2.7544279, 2.8132067
8: -1.5904655, 1.3277726, -1.6460779, 1.3934045, -2.9838700, 2.9738505
9: -8.8184881, -6.4457726, -8.9214840, -6.3808112, -2.2725906, 2.2859592

Time for backsubstitution: 12.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 4584

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6221

## Relational analysis of IS_A1_A1_B2_A1_B1

### Relational analysis result of IS_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5384787, upper bound: 1.5540980
time: 4.14 seconds

## Relational analysis of IS_A1_A1_B2_A1_B2

### Relational analysis result of IS_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5384787, upper bound: 1.5540984
time: 4.73 seconds

## BFS IS instance: IS_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -5.2489829, -2.7848151, -5.2888374, -2.7678323, -2.2688556, 2.2653522
1: -10.1104212, -7.2455597, -10.1514883, -7.2506461, -2.6778440, 2.7287774
2: -5.5617514, -2.7336950, -5.5959282, -2.6937885, -2.8679628, 2.8622332
3: -12.1861467, -9.0191727, -12.1917830, -8.9924278, -2.9840870, 2.9253416
4: -8.7736626, -5.4242344, -8.8215485, -5.4291430, -3.1283898, 3.1542275
5: -0.9339491, 1.5481961, -0.9645795, 1.5807396, -2.5146887, 2.5127754
6: 5.1053333, 7.4570303, 5.0881376, 7.5095949, -2.3465419, 2.3688927
7: -18.8491917, -15.3325748, -18.8794727, -15.3970537, -2.8198032, 2.8701785
8: -1.6163304, 1.3484097, -1.6460745, 1.3933992, -3.0097296, 2.9944842
9: -8.8607645, -6.3976526, -8.9214754, -6.3808141, -2.3149595, 2.3074791

Time for backsubstitution: 12.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 6196
type: A, layer: 1, pos: 5817
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 4584

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6221

## Relational analysis of IS_A1_A1_B2_A2_B1

### Relational analysis result of IS_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5384787, upper bound: 1.5617979
time: 4.36 seconds

## Relational analysis of IS_A1_A1_B2_A2_B2

### Relational analysis result of IS_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5384787, upper bound: 1.5617978
time: 4.58 seconds

## BFS IS instance: IS_A1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -5.2418299, -2.7888646, -5.2482462, -2.7802019, -2.2282600, 2.2146511
1: -10.1253624, -7.2708364, -10.1429052, -7.2638235, -2.6470771, 2.6875014
2: -5.5440521, -2.7253480, -5.5605154, -2.7213490, -2.8227031, 2.8351674
3: -12.1657391, -9.0344229, -12.1751757, -9.0126343, -2.9376917, 2.9312606
4: -8.7394238, -5.4557095, -8.7701941, -5.4502583, -3.1015730, 3.1401405
5: -0.9226303, 1.5465605, -0.9283432, 1.5568340, -2.4794643, 2.4749036
6: 5.1388330, 7.4433780, 5.1297588, 7.4612312, -2.3081431, 2.2928176
7: -18.8108482, -15.4186344, -18.8547401, -15.4136276, -2.7652745, 2.8150024
8: -1.6212859, 1.3519697, -1.6276741, 1.3615093, -2.9827952, 2.9796438
9: -8.8518553, -6.4022369, -8.8711138, -6.3979712, -2.2822819, 2.2663546

Time for backsubstitution: 12.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6221
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: B, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 4584
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6221

## Relational analysis of IS_A1_A2_B1_A1_B1

### Relational analysis result of IS_A1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5384832, upper bound: 1.5384831
time: 4.00 seconds

## Relational analysis of IS_A1_A2_B1_A1_B2

### Relational analysis result of IS_A1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5384832, upper bound: 1.5384839
time: 4.43 seconds

## BFS IS instance: IS_A1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -5.2746997, -2.7635746, -5.2482395, -2.7802072, -2.2811112, 2.2408783
1: -10.1634455, -7.2160187, -10.1428957, -7.2638297, -2.6860609, 2.7430754
2: -5.5874386, -2.6928062, -5.5605063, -2.7213519, -2.8660867, 2.8677001
3: -12.2139626, -9.0044994, -12.1751699, -9.0126495, -2.9900579, 2.9631076
4: -8.7939548, -5.4005175, -8.7701769, -5.4502640, -3.1580443, 3.1917005
5: -0.9479667, 1.5675259, -0.9283395, 1.5568259, -2.5047927, 2.4958653
6: 5.0864253, 7.4669991, 5.1297655, 7.4612250, -2.3594356, 2.3174963
7: -18.8736801, -15.3140564, -18.8547192, -15.4136276, -2.8304548, 2.8677082
8: -1.6480575, 1.3725429, -1.6276679, 1.3615036, -3.0095611, 3.0002108
9: -8.8942327, -6.3537078, -8.8711014, -6.3979745, -2.3249173, 2.3042107

Time for backsubstitution: 12.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 6221
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 4584
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 4584
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6196

## Relational analysis of IS_A1_A2_B1_A2_B1

### Relational analysis result of IS_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5458679, upper bound: 1.5461518
time: 6.07 seconds

## Relational analysis of IS_A1_A2_B1_A2_B2

### Relational analysis result of IS_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5458677, upper bound: 1.5461517
time: 4.29 seconds

## BFS IS instance: IS_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -5.2418299, -2.7888646, -5.2933288, -2.7619345, -2.2470627, 2.2586679
1: -10.1253624, -7.2708364, -10.1672621, -7.2483907, -2.6630144, 2.7105699
2: -5.5440521, -2.7253480, -5.5987654, -2.6825697, -2.8614824, 2.8734174
3: -12.1657391, -9.0344229, -12.1994390, -8.9904890, -2.9609909, 2.9560108
4: -8.7394238, -5.4557095, -8.8240805, -5.4225063, -3.1307430, 3.1825070
5: -0.9226303, 1.5465605, -0.9681313, 1.5843482, -2.5069785, 2.5146918
6: 5.1388330, 7.4433780, 5.0850172, 7.5114956, -2.3276443, 2.3390417
7: -18.8108482, -15.4186344, -18.8848648, -15.3919125, -2.7870846, 2.8477874
8: -1.6212859, 1.3519697, -1.6544261, 1.3965702, -3.0178561, 3.0063958
9: -8.8518553, -6.4022369, -8.9248180, -6.3680782, -2.3130987, 2.2973599

Time for backsubstitution: 12.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6221
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 5817
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 4584

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6221

## Relational analysis of IS_A1_A2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5384785, upper bound: 1.5544132
time: 4.69 seconds

## Relational analysis of IS_A1_A2_B2_A1_B2

### Relational analysis result of IS_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5384785, upper bound: 1.5544138
time: 4.38 seconds

## BFS IS instance: IS_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -5.2746997, -2.7635746, -5.2933245, -2.7619419, -2.2999134, 2.2848947
1: -10.1634455, -7.2160187, -10.1672544, -7.2483964, -2.7019978, 2.7662268
2: -5.5874386, -2.6928062, -5.5987554, -2.6825728, -2.9048657, 2.9059491
3: -12.2139626, -9.0044994, -12.1994314, -8.9905043, -3.0133581, 2.9878583
4: -8.7939548, -5.4005175, -8.8240633, -5.4225092, -3.1872149, 3.2140651
5: -0.9479667, 1.5675259, -0.9681270, 1.5843402, -2.5323069, 2.5356529
6: 5.0864253, 7.4669991, 5.0850239, 7.5114899, -2.3676271, 2.3637218
7: -18.8736801, -15.3140564, -18.8848419, -15.3919182, -2.8522644, 2.9005089
8: -1.6480575, 1.3725429, -1.6544216, 1.3965650, -3.0446224, 3.0269644
9: -8.8942327, -6.3537078, -8.9248095, -6.3680801, -2.3557346, 2.3191416

Time for backsubstitution: 12.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 6221
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6196

## Relational analysis of IS_A1_A2_B2_A2_B1

### Relational analysis result of IS_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5458633, upper bound: 1.5620810
time: 5.42 seconds

## Relational analysis of IS_A1_A2_B2_A2_B2

### Relational analysis result of IS_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5458631, upper bound: 1.5620805
time: 7.05 seconds

## BFS IS instance: IS_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -5.2823706, -2.7764890, -5.2261748, -2.7979717, -2.2422557, 2.2257183
1: -10.1340027, -7.2578382, -10.0850582, -7.2934737, -2.6541653, 2.6593881
2: -5.5794525, -2.6978426, -5.5393820, -2.7615275, -2.8179250, 2.8415394
3: -12.1824379, -9.0144539, -12.1501522, -9.0328150, -2.9015260, 2.9226031
4: -8.7909880, -5.4347801, -8.7278614, -5.4583254, -3.1111727, 3.0946522
5: -0.9587463, 1.5704901, -0.9134784, 1.5448227, -2.5035691, 2.4839685
6: 5.0979352, 7.4918532, 5.1299238, 7.4380884, -2.3401532, 2.3152826
7: -18.8352833, -15.4021587, -18.8095284, -15.4330320, -2.7735310, 2.7733212
8: -1.6393118, 1.3838472, -1.5942025, 1.3479252, -2.9872370, 2.9780498
9: -8.9023581, -6.3851776, -8.8239956, -6.4226065, -2.2895603, 2.2794340

Time for backsubstitution: 12.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 6196
type: B, layer: 1, pos: 5817
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 4584

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 481

## Relational analysis of IS_A2_B1_B1_A1_B1

### Relational analysis result of IS_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5541047, upper bound: 1.5384788
time: 4.80 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2

### Relational analysis result of IS_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5541057, upper bound: 1.5394075
time: 4.94 seconds

## BFS IS instance: IS_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -5.3149548, -2.7512562, -5.2261748, -2.7979717, -2.2793326, 2.2520497
1: -10.1723766, -7.2028117, -10.0850582, -7.2934737, -2.6932812, 2.7188768
2: -5.6229596, -2.6652415, -5.5393820, -2.7615275, -2.8614321, 2.8741405
3: -12.2306499, -8.9842291, -12.1501522, -9.0328150, -2.9539704, 2.9547811
4: -8.8455210, -5.3792024, -8.7278614, -5.4583254, -3.1677265, 3.1473641
5: -0.9842891, 1.5915017, -0.9134784, 1.5448227, -2.5291119, 2.5049801
6: 5.0446591, 7.5155215, 5.1299238, 7.4380884, -2.3934293, 2.3401783
7: -18.8985939, -15.2975092, -18.8095284, -15.4330320, -2.8387203, 2.8200989
8: -1.6664591, 1.4046412, -1.5942025, 1.3479252, -3.0143843, 2.9988437
9: -8.9447937, -6.3363886, -8.8239956, -6.4226065, -2.3322592, 2.3111820

Time for backsubstitution: 12.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 6196
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 4584

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 481

## Relational analysis of IS_A2_B1_B1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5541047, upper bound: 1.5461481
time: 5.07 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5541057, upper bound: 1.5470671
time: 5.16 seconds

## BFS IS instance: IS_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -5.2823706, -2.7764890, -5.2590275, -2.7726381, -2.2685890, 2.2637205
1: -10.1340027, -7.2578382, -10.1229849, -7.2392240, -2.7149296, 2.6985321
2: -5.5794525, -2.6978426, -5.5829248, -2.7290716, -2.8503809, 2.8850822
3: -12.1824379, -9.0144539, -12.1981773, -9.0026245, -2.9336958, 2.9750719
4: -8.7909880, -5.4347801, -8.7824373, -5.4036484, -3.1457796, 3.1512380
5: -0.9587463, 1.5704901, -0.9382758, 1.5657318, -2.5244780, 2.5087659
6: 5.0979352, 7.4918532, 5.0794506, 7.4618497, -2.3639145, 2.3530467
7: -18.8352833, -15.4021587, -18.8728142, -15.3286428, -2.8304176, 2.8389025
8: -1.6393118, 1.3838472, -1.6200590, 1.3686581, -3.0079699, 3.0039062
9: -8.9023581, -6.3851776, -8.8664341, -6.3743649, -2.3111162, 2.3219457

Time for backsubstitution: 12.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 4584

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 481

## Relational analysis of IS_A2_B1_B2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5540978, upper bound: 1.5384793
time: 4.47 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5540989, upper bound: 1.5394080
time: 4.56 seconds

## BFS IS instance: IS_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -5.3149548, -2.7512562, -5.2590275, -2.7726381, -2.3150687, 2.3094733
1: -10.1723766, -7.2028117, -10.1229849, -7.2392240, -2.7591982, 2.7612281
2: -5.6229596, -2.6652415, -5.5829248, -2.7290716, -2.8938880, 2.9176834
3: -12.2306499, -8.9842291, -12.1981773, -9.0026245, -2.9860926, 3.0072017
4: -8.8455210, -5.3792024, -8.7824373, -5.4036484, -3.2022743, 3.2008972
5: -0.9842891, 1.5915017, -0.9382758, 1.5657318, -2.5500207, 2.5297775
6: 5.0446591, 7.5155215, 5.0794506, 7.4618497, -2.4171906, 2.3784697
7: -18.8985939, -15.2975092, -18.8728142, -15.3286428, -2.8958750, 2.8859131
8: -1.6664591, 1.4046412, -1.6200590, 1.3686581, -3.0351171, 3.0247002
9: -8.9447937, -6.3363886, -8.8664341, -6.3743649, -2.3538141, 2.3537006

Time for backsubstitution: 12.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 6196
type: B, layer: 1, pos: 5817
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 4584

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 481

## Relational analysis of IS_A2_B1_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5540978, upper bound: 1.5461479
time: 4.54 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5540989, upper bound: 1.5470676
time: 5.03 seconds

## BFS IS instance: IS_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -5.2868514, -2.7705956, -5.2517414, -2.7765985, -2.2618284, 2.2567866
1: -10.1497889, -7.2555819, -10.1384773, -7.2643700, -2.7002401, 2.6838841
2: -5.5822859, -2.6866260, -5.5651836, -2.7206559, -2.8616300, 2.8785577
3: -12.1900921, -9.0125275, -12.1782389, -9.0181637, -2.9640121, 2.9524403
4: -8.7934952, -5.4281259, -8.7481575, -5.4350233, -3.1742492, 3.1535921
5: -0.9622877, 1.5740923, -0.9271365, 1.5640960, -2.5263836, 2.5012288
6: 5.0948458, 7.4937716, 5.1124864, 7.4479995, -2.3347387, 2.3350835
7: -18.8406696, -15.3970203, -18.8343143, -15.4146347, -2.8081732, 2.8061876
8: -1.6476529, 1.3869662, -1.6253161, 1.3721547, -3.0198076, 3.0122824
9: -8.9056873, -6.3724427, -8.8573236, -6.3788934, -2.3011379, 2.3199754

Time for backsubstitution: 12.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 6196
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 5817
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 444

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 481

## Relational analysis of IS_A2_B2_B1_A1_B1

### Relational analysis result of IS_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5544199, upper bound: 1.5384787
time: 4.10 seconds

## Relational analysis of IS_A2_B2_B1_A1_B2

### Relational analysis result of IS_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5544209, upper bound: 1.5394075
time: 4.35 seconds

## BFS IS instance: IS_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -5.3194294, -2.7453742, -5.2517414, -2.7765985, -2.2992392, 2.2831168
1: -10.1881466, -7.2005644, -10.1384773, -7.2643700, -2.7394505, 2.7435334
2: -5.6257682, -2.6540318, -5.5651836, -2.7206559, -2.9051123, 2.9111519
3: -12.2383308, -8.9823084, -12.1782389, -9.0181637, -3.0164623, 2.9846196
4: -8.8480406, -5.3725176, -8.7481575, -5.4350233, -3.2306747, 3.1982799
5: -0.9878460, 1.5951183, -0.9271365, 1.5640960, -2.5519419, 2.5222549
6: 5.0415435, 7.5174217, 5.1124864, 7.4479995, -2.3896451, 2.3599880
7: -18.9039059, -15.2923870, -18.8343143, -15.4146347, -2.8739948, 2.8496079
8: -1.6748233, 1.4077740, -1.6253161, 1.3721547, -3.0469780, 3.0330901
9: -8.9481421, -6.3236198, -8.8573236, -6.3788934, -2.3438258, 2.3421402

Time for backsubstitution: 12.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 6196
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 4584
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 444

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 481

## Relational analysis of IS_A2_B2_B1_A2_B1

### Relational analysis result of IS_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5544199, upper bound: 1.5461479
time: 4.07 seconds

## Relational analysis of IS_A2_B2_B1_A2_B2

### Relational analysis result of IS_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5544209, upper bound: 1.5470672
time: 4.48 seconds

## BFS IS instance: IS_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -5.2933245, -2.7619419, -5.2746997, -2.7635746, -2.2848949, 2.2999136
1: -10.1672544, -7.2483964, -10.1634455, -7.2160187, -2.7662268, 2.7019982
2: -5.5987554, -2.6825728, -5.5874386, -2.6928062, -2.9059491, 2.9048657
3: -12.1994314, -8.9905043, -12.2139626, -9.0044994, -2.9878588, 3.0133586
4: -8.8240633, -5.4225092, -8.7939548, -5.4005175, -3.2140651, 3.1872153
5: -0.9681270, 1.5843402, -0.9479667, 1.5675259, -2.5356529, 2.5323069
6: 5.0850239, 7.5114899, 5.0864253, 7.4669991, -2.3637214, 2.3676271
7: -18.8848419, -15.3919182, -18.8736801, -15.3140564, -2.9005094, 2.8522639
8: -1.6544216, 1.3965650, -1.6480575, 1.3725429, -3.0269644, 3.0446224
9: -8.9248095, -6.3680801, -8.8942327, -6.3537078, -2.3191414, 2.3557343

Time for backsubstitution: 12.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6196

## Relational analysis of IS_A2_B2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5617973, upper bound: 1.5458628
time: 4.56 seconds

## Relational analysis of IS_A2_B2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5617971, upper bound: 1.5461485
time: 4.29 seconds

## BFS IS instance: IS_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -5.2933245, -2.7619419, -5.3194246, -2.7453840, -2.2912836, 2.3316035
1: -10.1672544, -7.2483964, -10.1881361, -7.2005663, -2.7871270, 2.7420063
2: -5.5987554, -2.6825728, -5.6257658, -2.6540394, -2.9447160, 2.9431930
3: -12.1994314, -8.9905043, -12.2383204, -8.9823093, -3.0129576, 3.0399847
4: -8.8240633, -5.4225092, -8.8480387, -5.3725290, -3.2523394, 3.2264347
5: -0.9681270, 1.5843402, -0.9878407, 1.5951147, -2.5632417, 2.5721807
6: 5.0850239, 7.5114899, 5.0415483, 7.5174193, -2.3861017, 2.4144127
7: -18.8848419, -15.3919182, -18.9039021, -15.2923937, -2.9132771, 2.8762131
8: -1.6544216, 1.3965650, -1.6748140, 1.4077725, -3.0621941, 3.0713789
9: -8.9248095, -6.3680801, -8.9481373, -6.3236427, -2.3530064, 2.3736234

Time for backsubstitution: 12.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4584
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6196

## Relational analysis of IS_A2_B2_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5617983, upper bound: 1.5467895
time: 4.59 seconds

## Relational analysis of IS_A2_B2_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5617981, upper bound: 1.5470677
time: 4.34 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 21.68 seconds
IS_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 21.68
Output dim: 6, lower bound: -1.5384834, upper bound: 1.5381679
IS_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 21.68
Output dim: 6, lower bound: -1.5384834, upper bound: 1.5381683
IS_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 21.68
Output dim: 6, lower bound: -1.5384834, upper bound: 1.5458679
IS_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 21.68
Output dim: 6, lower bound: -1.5384834, upper bound: 1.5458678
IS_A1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 21.68
Output dim: 6, lower bound: -1.5384787, upper bound: 1.5540980
IS_A1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 21.68
Output dim: 6, lower bound: -1.5384787, upper bound: 1.5540984
IS_A1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 21.68
Output dim: 6, lower bound: -1.5384787, upper bound: 1.5617979
IS_A1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 21.68
Output dim: 6, lower bound: -1.5384787, upper bound: 1.5617978
IS_A1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 21.68
Output dim: 6, lower bound: -1.5384832, upper bound: 1.5384831
IS_A1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 21.68
Output dim: 6, lower bound: -1.5384832, upper bound: 1.5384839
IS_A1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 21.68
Output dim: 6, lower bound: -1.5458679, upper bound: 1.5461518
IS_A1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 21.68
Output dim: 6, lower bound: -1.5458677, upper bound: 1.5461517
IS_A1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 21.68
Output dim: 6, lower bound: -1.5384785, upper bound: 1.5544132
IS_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 21.68
Output dim: 6, lower bound: -1.5384785, upper bound: 1.5544138
IS_A1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 21.68
Output dim: 6, lower bound: -1.5458633, upper bound: 1.5620810
IS_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 21.68
Output dim: 6, lower bound: -1.5458631, upper bound: 1.5620805
IS_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 21.68
Output dim: 6, lower bound: -1.5541047, upper bound: 1.5384788
IS_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 21.68
Output dim: 6, lower bound: -1.5541057, upper bound: 1.5394075
IS_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 21.68
Output dim: 6, lower bound: -1.5541047, upper bound: 1.5461481
IS_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 21.68
Output dim: 6, lower bound: -1.5541057, upper bound: 1.5470671
IS_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 21.68
Output dim: 6, lower bound: -1.5540978, upper bound: 1.5384793
IS_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 21.68
Output dim: 6, lower bound: -1.5540989, upper bound: 1.5394080
IS_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 21.68
Output dim: 6, lower bound: -1.5540978, upper bound: 1.5461479
IS_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 21.68
Output dim: 6, lower bound: -1.5540989, upper bound: 1.5470676
IS_A2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 21.68
Output dim: 6, lower bound: -1.5544199, upper bound: 1.5384787
IS_A2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 21.68
Output dim: 6, lower bound: -1.5544209, upper bound: 1.5394075
IS_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 21.68
Output dim: 6, lower bound: -1.5544199, upper bound: 1.5461479
IS_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 21.68
Output dim: 6, lower bound: -1.5544209, upper bound: 1.5470672
IS_A2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 21.68
Output dim: 6, lower bound: -1.5617973, upper bound: 1.5458628
IS_A2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.68
Output dim: 6, lower bound: -1.5617971, upper bound: 1.5461485
IS_A2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.68
Output dim: 6, lower bound: -1.5617983, upper bound: 1.5467895
IS_A2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.68
Output dim: 6, lower bound: -1.5617981, upper bound: 1.5470677

## BFS IS instance: IS_A1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -5.2161036, -2.8101668, -5.2372613, -2.7946754, -2.1843100, 2.1842816
1: -10.0724096, -7.2998371, -10.1099634, -7.2731204, -2.6141486, 2.6212292
2: -5.5182486, -2.7661290, -5.5412679, -2.7365475, -2.7817011, 2.7751389
3: -12.1381540, -9.0490913, -12.1584396, -9.0363264, -2.8862581, 2.8599286
4: -8.7192230, -5.4787874, -8.7370167, -5.4623003, -3.0375433, 3.0358968
5: -0.9091848, 1.5272841, -0.9191207, 1.5430002, -2.4521852, 2.4464049
6: 5.1556816, 7.4334130, 5.1417923, 7.4414749, -2.2669849, 2.2916207
7: -18.7861271, -15.4369545, -18.8056679, -15.4237614, -2.7266226, 2.7351670
8: -1.5904655, 1.3277726, -1.6130207, 1.3489528, -2.9394183, 2.9407933
9: -8.8184881, -6.4457726, -8.8485212, -6.4148970, -2.2381411, 2.2338750

Time for backsubstitution: 12.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 6196
type: A, layer: 1, pos: 5817
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 4584

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4558

## Relational analysis of IS_A1_A1_B1_A1_B1_B1

### Relational analysis result of IS_A1_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5349074, upper bound: 1.5381334
time: 4.12 seconds

## Relational analysis of IS_A1_A1_B1_A1_B1_B2

### Relational analysis result of IS_A1_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5384697, upper bound: 1.5381614
time: 4.23 seconds

## BFS IS instance: IS_A1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -5.2161036, -2.8101668, -5.2701435, -2.7693744, -2.2105317, 2.2217309
1: -10.0724096, -7.2998371, -10.1480589, -7.2182961, -2.6754208, 2.6601372
2: -5.5182486, -2.7661290, -5.5846777, -2.7039986, -2.8142500, 2.8185487
3: -12.1381540, -9.0490913, -12.2066364, -9.0063992, -2.9181132, 2.9124367
4: -8.7192230, -5.4787874, -8.7915373, -5.4071431, -3.0967512, 3.0924640
5: -0.9091848, 1.5272841, -0.9444358, 1.5639503, -2.4731350, 2.4717200
6: 5.1556816, 7.4334130, 5.0894203, 7.4651146, -2.2916498, 2.3439927
7: -18.7861271, -15.4369545, -18.8685703, -15.3191719, -2.7834249, 2.7999425
8: -1.5904655, 1.3277726, -1.6397624, 1.3695250, -2.9599905, 2.9675350
9: -8.8184881, -6.4457726, -8.8908815, -6.3664055, -2.2697692, 2.2764730

Time for backsubstitution: 12.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 4584
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 4584

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4558

## Relational analysis of IS_A1_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5349074, upper bound: 1.5381339
time: 4.43 seconds

## Relational analysis of IS_A1_A1_B1_A1_B2_B2

### Relational analysis result of IS_A1_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5384697, upper bound: 1.5381637
time: 5.52 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -5.2489829, -2.7848151, -5.2372613, -2.7946754, -2.2221165, 2.2104959
1: -10.1104212, -7.2455597, -10.1099634, -7.2731204, -2.6529422, 2.6820030
2: -5.5617514, -2.7336950, -5.5412679, -2.7365475, -2.8252039, 2.8075728
3: -12.1861467, -9.0191727, -12.1584396, -9.0363264, -2.9387140, 2.8917975
4: -8.7736626, -5.4242344, -8.7370167, -5.4623003, -3.0939841, 3.0947542
5: -0.9339491, 1.5481961, -0.9191207, 1.5430002, -2.4769492, 2.4673166
6: 5.1053333, 7.4570303, 5.1417923, 7.4414749, -2.3188939, 2.3152380
7: -18.8491917, -15.3325748, -18.8056679, -15.4237614, -2.7919569, 2.7920139
8: -1.6163304, 1.3484097, -1.6130207, 1.3489528, -2.9652832, 2.9614303
9: -8.8607645, -6.3976526, -8.8485212, -6.4148970, -2.2804871, 2.2717438

Time for backsubstitution: 12.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 6196
type: A, layer: 1, pos: 5817
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 4584

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4558

## Relational analysis of IS_A1_A1_B1_A2_B1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5349074, upper bound: 1.5458146
time: 5.68 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_B2

### Relational analysis result of IS_A1_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5384697, upper bound: 1.5458539
time: 4.22 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -5.2489829, -2.7848151, -5.2701435, -2.7693744, -2.2677593, 2.2673671
1: -10.1104212, -7.2455597, -10.1480589, -7.2182961, -2.7217045, 2.7259269
2: -5.5617514, -2.7336950, -5.5846777, -2.7039986, -2.8577528, 2.8509827
3: -12.1861467, -9.0191727, -12.2066364, -9.0063992, -2.9705205, 2.9442577
4: -8.7736626, -5.4242344, -8.7915373, -5.4071431, -3.1433988, 3.1415272
5: -0.9339491, 1.5481961, -0.9444358, 1.5639503, -2.4978995, 2.4926319
6: 5.1053333, 7.4570303, 5.0894203, 7.4651146, -2.3441620, 2.3676100
7: -18.8491917, -15.3325748, -18.8685703, -15.3191719, -2.8489614, 2.8570030
8: -1.6163304, 1.3484097, -1.6397624, 1.3695250, -2.9858553, 2.9881721
9: -8.8607645, -6.3976526, -8.8908815, -6.3664055, -2.3121285, 2.3143449

Time for backsubstitution: 12.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 6196
type: A, layer: 1, pos: 5817
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 4584

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4558

## Relational analysis of IS_A1_A1_B1_A2_B2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5349074, upper bound: 1.5458150
time: 4.64 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B2

### Relational analysis result of IS_A1_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5384697, upper bound: 1.5458542
time: 4.91 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 22.25 seconds
IS_A1_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 22.25
Output dim: 6, lower bound: -1.5349074, upper bound: 1.5381334
IS_A1_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 22.25
Output dim: 6, lower bound: -1.5384697, upper bound: 1.5381614
IS_A1_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 22.25
Output dim: 6, lower bound: -1.5349074, upper bound: 1.5381339
IS_A1_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 22.25
Output dim: 6, lower bound: -1.5384697, upper bound: 1.5381637
IS_A1_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 22.25
Output dim: 6, lower bound: -1.5349074, upper bound: 1.5458146
IS_A1_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 22.25
Output dim: 6, lower bound: -1.5384697, upper bound: 1.5458539
IS_A1_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 22.25
Output dim: 6, lower bound: -1.5349074, upper bound: 1.5458150
IS_A1_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 22.25
Output dim: 6, lower bound: -1.5384697, upper bound: 1.5458542
IS_A1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 22.25
Output dim: 6, lower bound: -1.5384787, upper bound: 1.5540980
IS_A1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 22.25
Output dim: 6, lower bound: -1.5384787, upper bound: 1.5540984
IS_A1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 22.25
Output dim: 6, lower bound: -1.5384787, upper bound: 1.5617979
IS_A1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 22.25
Output dim: 6, lower bound: -1.5384787, upper bound: 1.5617978
IS_A1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 22.25
Output dim: 6, lower bound: -1.5384832, upper bound: 1.5384831
IS_A1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 22.25
Output dim: 6, lower bound: -1.5384832, upper bound: 1.5384839
IS_A1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 22.25
Output dim: 6, lower bound: -1.5458679, upper bound: 1.5461518
IS_A1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 22.25
Output dim: 6, lower bound: -1.5458677, upper bound: 1.5461517
IS_A1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 22.25
Output dim: 6, lower bound: -1.5384785, upper bound: 1.5544132
IS_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 22.25
Output dim: 6, lower bound: -1.5384785, upper bound: 1.5544138
IS_A1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 22.25
Output dim: 6, lower bound: -1.5458633, upper bound: 1.5620810
IS_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 22.25
Output dim: 6, lower bound: -1.5458631, upper bound: 1.5620805
IS_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 22.25
Output dim: 6, lower bound: -1.5541047, upper bound: 1.5384788
IS_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 22.25
Output dim: 6, lower bound: -1.5541057, upper bound: 1.5394075
IS_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 22.25
Output dim: 6, lower bound: -1.5541047, upper bound: 1.5461481
IS_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 22.25
Output dim: 6, lower bound: -1.5541057, upper bound: 1.5470671
IS_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 22.25
Output dim: 6, lower bound: -1.5540978, upper bound: 1.5384793
IS_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 22.25
Output dim: 6, lower bound: -1.5540989, upper bound: 1.5394080
IS_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 22.25
Output dim: 6, lower bound: -1.5540978, upper bound: 1.5461479
IS_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 22.25
Output dim: 6, lower bound: -1.5540989, upper bound: 1.5470676
IS_A2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 22.25
Output dim: 6, lower bound: -1.5544199, upper bound: 1.5384787
IS_A2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 22.25
Output dim: 6, lower bound: -1.5544209, upper bound: 1.5394075
IS_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 22.25
Output dim: 6, lower bound: -1.5544199, upper bound: 1.5461479
IS_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 22.25
Output dim: 6, lower bound: -1.5544209, upper bound: 1.5470672
IS_A2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.25
Output dim: 6, lower bound: -1.5617973, upper bound: 1.5458628
IS_A2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.25
Output dim: 6, lower bound: -1.5617971, upper bound: 1.5461485
IS_A2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.25
Output dim: 6, lower bound: -1.5617983, upper bound: 1.5467895
IS_A2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.25
Output dim: 6, lower bound: -1.5617981, upper bound: 1.5470677
Binary search (step 1): status=Status.UNKNOWN, k_low=6, k_high=8, k_mid=7, eps_mid=0.0273438, abs_max=2.3460235595703125
rel_dist={6: [-1.5621134080175434, 1.562113956458468]}

## Binary search (step 2) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 481
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 6196
type: B, layer: 1, pos: 6196
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 6221
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: A, layer: 1, pos: 5817
type: B, layer: 1, pos: 4584
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 481

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4800157, upper bound: 1.4943217
time: 4.23 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4944183, upper bound: 1.4944194
time: 4.00 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 8.40 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 8.40
Output dim: 6, lower bound: -1.4800157, upper bound: 1.4943217
IS_A2, status: Status.UNKNOWN, split count: 1, time: 8.40
Output dim: 6, lower bound: -1.4944183, upper bound: 1.4944194

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -5.2482443, -2.7802007, -5.2574754, -2.7689109, -2.1903715, 2.1906478
1: -10.1429062, -7.2638254, -10.1547470, -7.2576556, -2.6314898, 2.6368985
2: -5.5605159, -2.7213488, -5.5800514, -2.7169671, -2.8435488, 2.8587027
3: -12.1751766, -9.0126343, -12.1863689, -8.9974995, -2.8987246, 2.8953099
4: -8.7701921, -5.4502583, -8.7781849, -5.4309831, -3.0782967, 3.0726166
5: -0.9283434, 1.5568354, -0.9326491, 1.5730131, -2.5013566, 2.4894845
6: 5.1297588, 7.4612336, 5.1047010, 7.4655485, -2.2817621, 2.3027349
7: -18.8547440, -15.4136238, -18.8764381, -15.4098434, -2.7272487, 2.7447047
8: -1.6276751, 1.3615103, -1.6316652, 1.3801584, -3.0078335, 2.9931755
9: -8.8711119, -6.3979683, -8.8761616, -6.3762927, -2.2652435, 2.2514846

Time for backsubstitution: 12.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6196
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 6221
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 4584

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6196

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4800157, upper bound: 1.4941241
time: 4.10 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4800154, upper bound: 1.4943215
time: 4.19 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -5.2933302, -2.7619328, -5.2582283, -2.7679303, -2.2355208, 2.2171488
1: -10.1672659, -7.2483907, -10.1559610, -7.2571564, -2.6553640, 2.6667795
2: -5.5987654, -2.6825676, -5.5816493, -2.7165899, -2.8821754, 2.8990817
3: -12.1994400, -8.9904881, -12.1876230, -8.9962492, -2.9247980, 2.9197898
4: -8.8240805, -5.4225054, -8.7789135, -5.4293995, -3.1276286, 3.1114531
5: -0.9681324, 1.5843486, -0.9330316, 1.5743568, -2.5424891, 2.5173802
6: 5.0850186, 7.5114970, 5.1026196, 7.4659157, -2.3272910, 2.3269198
7: -18.8848648, -15.3919125, -18.8783188, -15.4095287, -2.7599735, 2.7685595
8: -1.6544285, 1.3965702, -1.6320372, 1.3817124, -3.0361409, 3.0286074
9: -8.9248171, -6.3680735, -8.8765821, -6.3745155, -2.2975986, 2.2812848

Time for backsubstitution: 12.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6196
type: A, layer: 1, pos: 6196
type: B, layer: 1, pos: 6221
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6196

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4942202, upper bound: 1.4944195
time: 4.63 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4944180, upper bound: 1.4944191
time: 4.32 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 21.76 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 21.76
Output dim: 6, lower bound: -1.4800157, upper bound: 1.4941241
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 21.76
Output dim: 6, lower bound: -1.4800154, upper bound: 1.4943215
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 21.76
Output dim: 6, lower bound: -1.4942202, upper bound: 1.4944195
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 21.76
Output dim: 6, lower bound: -1.4944180, upper bound: 1.4944191

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -5.2224331, -2.8014963, -5.2520556, -2.7758567, -2.1581407, 2.1586034
1: -10.0900106, -7.2931690, -10.1363020, -7.2603703, -2.5784755, 2.5876846
2: -5.5347357, -2.7622347, -5.5767260, -2.7303267, -2.8044090, 2.8144913
3: -12.1475792, -9.0272722, -12.1775866, -8.9997768, -2.8625407, 2.8321719
4: -8.7499352, -5.4736094, -8.7752762, -5.4388347, -3.0113354, 3.0033631
5: -0.9145565, 1.5375628, -0.9284375, 1.5687466, -2.4833031, 2.4660003
6: 5.1479845, 7.4512987, 5.1082945, 7.4633012, -2.2610497, 2.3230894
7: -18.8300114, -15.4321060, -18.8702412, -15.4159622, -2.6887259, 2.7140083
8: -1.5962601, 1.3372712, -1.6217828, 1.3765016, -2.9727616, 2.9590540
9: -8.8377466, -6.4417171, -8.8721924, -6.3914423, -2.2193933, 2.2032900

Time for backsubstitution: 12.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 6196
type: A, layer: 1, pos: 4584
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 4584

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 481

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4800157, upper bound: 1.4798190
time: 4.28 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4800157, upper bound: 1.4941241
time: 4.25 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -5.2482414, -2.7802112, -5.2574754, -2.7689133, -2.1903653, 2.1783895
1: -10.1428947, -7.2638259, -10.1547451, -7.2576575, -2.6013360, 2.6368914
2: -5.5605116, -2.7213585, -5.5800505, -2.7169695, -2.8435421, 2.8586919
3: -12.1751680, -9.0126381, -12.1863680, -8.9974995, -2.8917465, 2.8953061
4: -8.7701921, -5.4502707, -8.7781858, -5.4309850, -3.0716987, 3.0767016
5: -0.9283382, 1.5568302, -0.9326477, 1.5730140, -2.5013523, 2.4894779
6: 5.1297626, 7.4612293, 5.1047029, 7.4655471, -2.2848082, 2.3005199
7: -18.8547363, -15.4136333, -18.8764343, -15.4098434, -2.7219887, 2.7482848
8: -1.6276655, 1.3615065, -1.6316633, 1.3801589, -3.0078244, 2.9931698
9: -8.8711100, -6.3979945, -8.8761616, -6.3762980, -2.2615852, 2.2137761

Time for backsubstitution: 12.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 6221
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 5817
type: A, layer: 1, pos: 4584
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 4584

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 481

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4800154, upper bound: 1.4800196
time: 4.02 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4800154, upper bound: 1.4943216
time: 4.06 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -5.2879682, -2.7689369, -5.2325292, -2.7892954, -2.2035470, 2.1848769
1: -10.1485367, -7.2510824, -10.1025982, -7.2868247, -2.6055093, 2.6133361
2: -5.5953932, -2.6959255, -5.5558701, -2.7576246, -2.8377686, 2.8599446
3: -12.1904135, -8.9927883, -12.1595306, -9.0108719, -2.8614960, 2.8828158
4: -8.8210897, -5.4303923, -8.7585545, -5.4531569, -3.0626268, 3.0444026
5: -0.9639138, 1.5800430, -0.9188522, 1.5550880, -2.5190020, 2.4988952
6: 5.0887265, 7.5092325, 5.1222157, 7.4559779, -2.3475657, 2.3048959
7: -18.8784714, -15.3980284, -18.8535271, -15.4281788, -2.7289133, 2.7298069
8: -1.6444960, 1.3928032, -1.6000097, 1.3574419, -3.0019379, 2.9928129
9: -8.9208574, -6.3832536, -8.8432522, -6.4185314, -2.2492959, 2.2351351

Time for backsubstitution: 12.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6221
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 6196
type: B, layer: 1, pos: 5817
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 4584

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6221

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4871384, upper bound: 1.4941342
time: 4.39 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4942112, upper bound: 1.4944112
time: 4.83 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -5.2933292, -2.7619357, -5.2582235, -2.7679420, -2.2232642, 2.2171426
1: -10.1672611, -7.2483912, -10.1559477, -7.2571597, -2.6553564, 2.6366239
2: -5.5987644, -2.6825709, -5.5816445, -2.7165992, -2.8821652, 2.8990736
3: -12.1994362, -8.9904919, -12.1876125, -8.9962521, -2.9247942, 2.9125781
4: -8.8240805, -5.4225082, -8.7789097, -5.4294128, -3.1249185, 3.1048579
5: -0.9681301, 1.5843471, -0.9330238, 1.5743525, -2.5424826, 2.5173709
6: 5.0850186, 7.5114956, 5.1026235, 7.4659128, -2.3250728, 2.3268461
7: -18.8848648, -15.3919163, -18.8783169, -15.4095345, -2.7634864, 2.7632289
8: -1.6544251, 1.3965693, -1.6320271, 1.3817101, -3.0361352, 3.0285964
9: -8.9248180, -6.3680792, -8.8765774, -6.3745418, -2.2595649, 2.2781150

Time for backsubstitution: 12.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6221
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 6196
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 4584
type: B, layer: 1, pos: 4584
type: A, layer: 1, pos: 5817
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 444

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6221

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4873818, upper bound: 1.4941342
time: 4.33 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4944093, upper bound: 1.4944108
time: 4.30 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 21.37 seconds
IS_A1_A1_B1, status: Status.VERIFIED, split count: 3, time: 21.37
Output dim: 6, lower bound: -1.4800157, upper bound: 1.4798190
IS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 21.37
Output dim: 6, lower bound: -1.4800157, upper bound: 1.4941241
IS_A1_A2_B1, status: Status.VERIFIED, split count: 3, time: 21.37
Output dim: 6, lower bound: -1.4800154, upper bound: 1.4800196
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 21.37
Output dim: 6, lower bound: -1.4800154, upper bound: 1.4943216
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 21.37
Output dim: 6, lower bound: -1.4871384, upper bound: 1.4941342
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 21.37
Output dim: 6, lower bound: -1.4942112, upper bound: 1.4944112
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 21.37
Output dim: 6, lower bound: -1.4873818, upper bound: 1.4941342
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 21.37
Output dim: 6, lower bound: -1.4944093, upper bound: 1.4944108

## BFS IS instance: IS_A1_A1_B2

### Backsubstitution after applying IS history:
0: -5.2224331, -2.8014963, -5.2879682, -2.7689369, -2.1637683, 2.1894517
1: -10.0900106, -7.2931690, -10.1485367, -7.2510824, -2.5845799, 2.5954127
2: -5.5347357, -2.7622347, -5.5953932, -2.6959255, -2.8388102, 2.8331585
3: -12.1475792, -9.0272722, -12.1904135, -8.9927883, -2.8697982, 2.8440466
4: -8.7499352, -5.4736094, -8.8210897, -5.4303923, -3.0187335, 3.0393782
5: -0.9145565, 1.5375628, -0.9639138, 1.5800430, -2.4945993, 2.5014768
6: 5.1479845, 7.4512987, 5.0887265, 7.5092325, -2.2787681, 2.3437898
7: -18.8300114, -15.4321060, -18.8784714, -15.3980284, -2.7046628, 2.7232199
8: -1.5962601, 1.3372712, -1.6444960, 1.3928032, -2.9890633, 2.9817672
9: -8.8377466, -6.4417171, -8.9208574, -6.3832536, -2.2270226, 2.2248337

Time for backsubstitution: 12.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 6196
type: A, layer: 1, pos: 4584
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 5817
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 4584

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6221

## Relational analysis of IS_A1_A1_B2_A1

### Relational analysis result of IS_A1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4797304, upper bound: 1.4870425
time: 4.17 seconds

## Relational analysis of IS_A1_A1_B2_A2

### Relational analysis result of IS_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4800074, upper bound: 1.4941152
time: 4.25 seconds

## BFS IS instance: IS_A1_A2_B2

### Backsubstitution after applying IS history:
0: -5.2482414, -2.7802112, -5.2933292, -2.7619357, -2.1960635, 2.2090256
1: -10.1428947, -7.2638259, -10.1672611, -7.2483912, -2.6073475, 2.6446257
2: -5.5605116, -2.7213585, -5.5987644, -2.6825709, -2.8779407, 2.8774059
3: -12.1751680, -9.0126381, -12.1994362, -8.9904919, -2.8990269, 2.9074488
4: -8.7701921, -5.4502707, -8.8240805, -5.4225082, -3.0791383, 3.1013746
5: -0.9283382, 1.5568302, -0.9681301, 1.5843471, -2.5126853, 2.5249603
6: 5.1297626, 7.4612293, 5.0850186, 7.5114956, -2.2993321, 2.3213739
7: -18.8547363, -15.4136333, -18.8848648, -15.3919163, -2.7378745, 2.7575731
8: -1.6276655, 1.3615065, -1.6544251, 1.3965693, -3.0242348, 3.0159316
9: -8.8711100, -6.3979945, -8.9248180, -6.3680792, -2.2681689, 2.2348967

Time for backsubstitution: 12.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 5817
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 4584
type: A, layer: 1, pos: 444

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6221

## Relational analysis of IS_A1_A2_B2_A1

### Relational analysis result of IS_A1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4797302, upper bound: 1.4872872
time: 5.10 seconds

## Relational analysis of IS_A1_A2_B2_A2

### Relational analysis result of IS_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4800071, upper bound: 1.4943127
time: 4.34 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -5.2873125, -2.7698274, -5.2261739, -2.7979715, -2.1894174, 2.1726253
1: -10.1467457, -7.2518096, -10.0850554, -7.2934742, -2.5950837, 2.5916162
2: -5.5937037, -2.6963367, -5.5393786, -2.7615285, -2.8321753, 2.8430419
3: -12.1894398, -8.9950533, -12.1501503, -9.0328178, -2.8381777, 2.8713236
4: -8.8179483, -5.4309807, -8.7278595, -5.4583254, -3.0543976, 3.0120463
5: -0.9633098, 1.5789913, -0.9134777, 1.5448213, -2.5081310, 2.4924688
6: 5.0897198, 7.5074081, 5.1299262, 7.4380884, -2.3276362, 2.2948825
7: -18.8739395, -15.3985443, -18.8095245, -15.4330320, -2.7184315, 2.6831069
8: -1.6437981, 1.3918214, -1.5942013, 1.3479233, -2.9917214, 2.9860227
9: -8.9188938, -6.3836975, -8.8239937, -6.4226103, -2.2432804, 2.2138700

Time for backsubstitution: 12.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 6196
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 4584

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6221

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4871325, upper bound: 1.4873774
time: 4.62 seconds

## Relational analysis of IS_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4871325, upper bound: 1.4941342
time: 4.40 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -5.2879620, -2.7689452, -5.2590284, -2.7726388, -2.2168541, 2.2263885
1: -10.1485243, -7.2510891, -10.1229839, -7.2392240, -2.6596513, 2.6316862
2: -5.5953832, -2.6959286, -5.5829225, -2.7290716, -2.8663116, 2.8869939
3: -12.1904030, -8.9928055, -12.1981754, -9.0026264, -2.8712640, 2.9260812
4: -8.8210697, -5.4303980, -8.7824364, -5.4036508, -3.0909095, 3.0686359
5: -0.9639095, 1.5800333, -0.9382752, 1.5657303, -2.5296397, 2.5183086
6: 5.0887327, 7.5092220, 5.0794530, 7.4618492, -2.3534203, 2.3352904
7: -18.8784447, -15.3980331, -18.8728123, -15.3286438, -2.7784424, 2.7480354
8: -1.6444888, 1.3927960, -1.6200588, 1.3686562, -3.0131450, 3.0128548
9: -8.9208479, -6.3832569, -8.8664322, -6.3743672, -2.2673779, 2.2561336

Time for backsubstitution: 12.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 6196
type: B, layer: 1, pos: 5817
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 4584

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6221

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4939106, upper bound: 1.4873777
time: 4.60 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4939106, upper bound: 1.4873769
time: 5.62 seconds

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: -5.2926702, -2.7628260, -5.2517400, -2.7765985, -2.2090998, 2.2047989
1: -10.1654711, -7.2491188, -10.1384754, -7.2643714, -2.6440525, 2.6149678
2: -5.5970745, -2.6829817, -5.5651798, -2.7206554, -2.8764191, 2.8821981
3: -12.1984653, -8.9927559, -12.1782379, -9.0181637, -2.9016008, 2.9010253
4: -8.8209381, -5.4230943, -8.7481565, -5.4350252, -3.1156020, 3.0726180
5: -0.9675263, 1.5832965, -0.9271361, 1.5640943, -2.5316205, 2.5104327
6: 5.0860167, 7.5096693, 5.1124907, 7.4479985, -2.3052135, 2.3148384
7: -18.8803329, -15.3924332, -18.8343124, -15.4146366, -2.7526798, 2.7165174
8: -1.6537259, 1.3955851, -1.6253152, 1.3721528, -3.0258787, 3.0209002
9: -8.9228535, -6.3685222, -8.8573236, -6.3788958, -2.2532659, 2.2569070

Time for backsubstitution: 12.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 6196
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6221

## Relational analysis of IS_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4873759, upper bound: 1.4873773
time: 4.56 seconds

## Relational analysis of IS_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4873759, upper bound: 1.4941340
time: 4.46 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: -5.2933235, -2.7619438, -5.2846346, -2.7513292, -2.2365432, 2.2585049
1: -10.1672506, -7.2483978, -10.1764708, -7.2093635, -2.7006865, 2.6549821
2: -5.5987539, -2.6825731, -5.6086087, -2.6880374, -2.9107165, 2.9260356
3: -12.1994295, -8.9905071, -12.2264891, -8.9879684, -2.9346595, 2.9558535
4: -8.8240595, -5.4225121, -8.8028240, -5.3795395, -3.1511307, 3.1292129
5: -0.9681263, 1.5843382, -0.9526744, 1.5850575, -2.5531838, 2.5370126
6: 5.0850258, 7.5114861, 5.0591922, 7.4718180, -2.3310127, 2.3578978
7: -18.8848362, -15.3919201, -18.8973618, -15.3099651, -2.8085361, 2.7812052
8: -1.6544197, 1.3965635, -1.6524105, 1.3928237, -3.0472434, 3.0489740
9: -8.9248066, -6.3680830, -8.8998642, -6.3301587, -2.2777154, 2.2994847

Time for backsubstitution: 12.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 6196
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6221
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 5817
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 481

## Relational analysis of IS_A2_B2_B2_B1

### Relational analysis result of IS_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4943115, upper bound: 1.4800072
time: 4.42 seconds

## Relational analysis of IS_A2_B2_B2_B2

### Relational analysis result of IS_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4943126, upper bound: 1.4809507
time: 4.67 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 21.94 seconds
IS_A1_A1_B2_A1, status: Status.VERIFIED, split count: 4, time: 21.94
Output dim: 6, lower bound: -1.4797304, upper bound: 1.4870425
IS_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 21.94
Output dim: 6, lower bound: -1.4800074, upper bound: 1.4941152
IS_A1_A2_B2_A1, status: Status.VERIFIED, split count: 4, time: 21.94
Output dim: 6, lower bound: -1.4797302, upper bound: 1.4872872
IS_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 21.94
Output dim: 6, lower bound: -1.4800071, upper bound: 1.4943127
IS_A2_B1_B1_A1, status: Status.VERIFIED, split count: 4, time: 21.94
Output dim: 6, lower bound: -1.4871325, upper bound: 1.4873774
IS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 21.94
Output dim: 6, lower bound: -1.4871325, upper bound: 1.4941342
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 21.94
Output dim: 6, lower bound: -1.4939106, upper bound: 1.4873777
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 21.94
Output dim: 6, lower bound: -1.4939106, upper bound: 1.4873769
IS_A2_B2_B1_A1, status: Status.VERIFIED, split count: 4, time: 21.94
Output dim: 6, lower bound: -1.4873759, upper bound: 1.4873773
IS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 21.94
Output dim: 6, lower bound: -1.4873759, upper bound: 1.4941340
IS_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 21.94
Output dim: 6, lower bound: -1.4943115, upper bound: 1.4800072
IS_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 21.94
Output dim: 6, lower bound: -1.4943126, upper bound: 1.4809507

## BFS IS instance: IS_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -5.2489829, -2.7848151, -5.2879620, -2.7689452, -2.2051735, 2.2026641
1: -10.1104212, -7.2455597, -10.1485243, -7.2510891, -2.6024265, 2.6490479
2: -5.5617514, -2.7336950, -5.5953832, -2.6959286, -2.8658228, 2.8616881
3: -12.1861467, -9.0191727, -12.1904030, -8.9928055, -2.9130850, 2.8537145
4: -8.7736626, -5.4242344, -8.8210697, -5.4303980, -3.0428987, 3.0677195
5: -0.9339491, 1.5481961, -0.9639095, 1.5800333, -2.5139823, 2.5121055
6: 5.1053333, 7.4570303, 5.0887327, 7.5092220, -2.3092160, 2.3494914
7: -18.8491917, -15.3325748, -18.8784447, -15.3980331, -2.7228203, 2.7717762
8: -1.6163304, 1.3484097, -1.6444888, 1.3927960, -3.0091264, 2.9928985
9: -8.8607645, -6.3976526, -8.9208479, -6.3832569, -2.2479115, 2.2429082

Time for backsubstitution: 12.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 4584
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 4584

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6221

## Relational analysis of IS_A1_A1_B2_A2_B1

### Relational analysis result of IS_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4729700, upper bound: 1.4938138
time: 4.28 seconds

## Relational analysis of IS_A1_A1_B2_A2_B2

### Relational analysis result of IS_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4729700, upper bound: 1.4941161
time: 4.79 seconds

## BFS IS instance: IS_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -5.2746997, -2.7635746, -5.2933235, -2.7619438, -2.2373381, 2.2222648
1: -10.1634455, -7.2160187, -10.1672506, -7.2483978, -2.6254492, 2.6894011
2: -5.5874386, -2.6928062, -5.5987539, -2.6825731, -2.9048655, 2.9059477
3: -12.2139626, -9.0044994, -12.1994295, -8.9905071, -2.9422150, 2.9171700
4: -8.7939548, -5.4005175, -8.8240595, -5.4225121, -3.1034222, 3.1276646
5: -0.9479667, 1.5675259, -0.9681263, 1.5843382, -2.5323048, 2.5356522
6: 5.0864253, 7.4669991, 5.0850258, 7.5114861, -2.3304701, 2.3271623
7: -18.8736801, -15.3140564, -18.8848362, -15.3919201, -2.7558365, 2.8016436
8: -1.6480575, 1.3725429, -1.6544197, 1.3965635, -3.0446210, 3.0269625
9: -8.8942327, -6.3537078, -8.9248066, -6.3680830, -2.2893801, 2.2530394

Time for backsubstitution: 12.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6196
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 4584

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6196

## Relational analysis of IS_A1_A2_B2_A2_B1

### Relational analysis result of IS_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4798064, upper bound: 1.4943111
time: 5.10 seconds

## Relational analysis of IS_A1_A2_B2_A2_B2

### Relational analysis result of IS_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4798062, upper bound: 1.4943119
time: 4.94 seconds

## BFS IS instance: IS_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -5.3140841, -2.7523673, -5.2261739, -2.7979715, -2.2160540, 2.1872797
1: -10.1694145, -7.2032523, -10.0850554, -7.2934742, -2.6156316, 2.6416664
2: -5.6224208, -2.6673784, -5.5393786, -2.7615285, -2.8608923, 2.8720002
3: -12.2292652, -8.9846039, -12.1501503, -9.0328178, -2.8823462, 2.8834147
4: -8.8450422, -5.3804679, -8.7278595, -5.4583254, -3.0822859, 3.0581803
5: -0.9836173, 1.5907933, -0.9134777, 1.5448213, -2.5284386, 2.5042710
6: 5.0452557, 7.5151501, 5.1299262, 7.4380884, -2.3720703, 2.3012843
7: -18.8975811, -15.2984867, -18.8095245, -15.4330320, -2.7422581, 2.7226939
8: -1.6648712, 1.4040427, -1.5942013, 1.3479233, -3.0127945, 2.9982440
9: -8.9441595, -6.3388381, -8.8239937, -6.4226103, -2.2661662, 2.2415855

Time for backsubstitution: 12.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 6196
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 4584

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 481

## Relational analysis of IS_A2_B1_B1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4870414, upper bound: 1.4797306
time: 4.58 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4870424, upper bound: 1.4806630
time: 4.31 seconds

## BFS IS instance: IS_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -5.2814970, -2.7776022, -5.2590284, -2.7726388, -2.2059011, 2.1989548
1: -10.1310387, -7.2582793, -10.1229839, -7.2392240, -2.6372986, 2.6225810
2: -5.5789070, -2.6999831, -5.5829225, -2.7290716, -2.8498354, 2.8829393
3: -12.1810589, -9.0148296, -12.1981754, -9.0026264, -2.8620720, 2.9037061
4: -8.7905121, -5.4360418, -8.7824364, -5.4036508, -3.0592756, 3.0636730
5: -0.9580765, 1.5697855, -0.9382752, 1.5657303, -2.5238068, 2.5080607
6: 5.0985250, 7.4914799, 5.0794530, 7.4618492, -2.3437486, 2.3157613
7: -18.8342571, -15.4031363, -18.8728123, -15.3286438, -2.7322249, 2.7432289
8: -1.6377285, 1.3832436, -1.6200588, 1.3686562, -3.0063846, 3.0033023
9: -8.9017315, -6.3876200, -8.8664322, -6.3743672, -2.2465458, 2.2529469

Time for backsubstitution: 12.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 4584

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 481

## Relational analysis of IS_A2_B1_B2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4870355, upper bound: 1.4729698
time: 8.69 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4870365, upper bound: 1.4738983
time: 4.56 seconds

## BFS IS instance: IS_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -5.3140841, -2.7523673, -5.2590284, -2.7726388, -2.2504759, 2.2440779
1: -10.1694145, -7.2032523, -10.1229839, -7.2392240, -2.6799088, 2.6837854
2: -5.6224208, -2.6673784, -5.5829225, -2.7290716, -2.8933492, 2.9155440
3: -12.2292652, -8.9846039, -12.1981754, -9.0026264, -2.9144654, 2.9358339
4: -8.8450422, -5.3804679, -8.7824364, -5.4036508, -3.1152120, 3.1127543
5: -0.9836173, 1.5907933, -0.9382752, 1.5657303, -2.5493476, 2.5290685
6: 5.0452557, 7.5151501, 5.0794530, 7.4618492, -2.3974295, 2.3411453
7: -18.8975811, -15.2984867, -18.8728123, -15.3286438, -2.7964325, 2.7872460
8: -1.6648712, 1.4040427, -1.6200588, 1.3686562, -3.0335274, 3.0241015
9: -8.9441595, -6.3388381, -8.8664322, -6.3743672, -2.2886028, 2.2834725

Time for backsubstitution: 12.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 6196
type: B, layer: 1, pos: 5817
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 4584

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 481

## Relational analysis of IS_A2_B1_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4870355, upper bound: 1.4800068
time: 5.82 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4870365, upper bound: 1.4809511
time: 4.87 seconds

## BFS IS instance: IS_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -5.3194294, -2.7453763, -5.2517400, -2.7765985, -2.2359905, 2.2194533
1: -10.1881447, -7.2005653, -10.1384754, -7.2643714, -2.6645765, 2.6646986
2: -5.6257677, -2.6540332, -5.5651798, -2.7206554, -2.9051123, 2.9111466
3: -12.2383270, -8.9823074, -12.1782379, -9.0181637, -2.9457760, 2.9131098
4: -8.8480406, -5.3725176, -8.7481565, -5.4350252, -3.1419964, 3.1107917
5: -0.9878467, 1.5951174, -0.9271361, 1.5640943, -2.5519409, 2.5222535
6: 5.0415444, 7.5174203, 5.1124907, 7.4479985, -2.3507996, 2.3212631
7: -18.9039078, -15.2923851, -18.8343124, -15.4146366, -2.7770786, 2.7527630
8: -1.6748221, 1.4077744, -1.6253152, 1.3721528, -3.0469749, 3.0330896
9: -8.9481401, -6.3236246, -8.8573236, -6.3788958, -2.2762103, 2.2757139

Time for backsubstitution: 12.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 6196
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 5817
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 444

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 481

## Relational analysis of IS_A2_B2_B1_A2_B1

### Relational analysis result of IS_A2_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4872859, upper bound: 1.4797304
time: 4.43 seconds

## Relational analysis of IS_A2_B2_B1_A2_B2

### Relational analysis result of IS_A2_B2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4872869, upper bound: 1.4806633
time: 4.82 seconds

## BFS IS instance: IS_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -5.2933235, -2.7619438, -5.2746997, -2.7635746, -2.2222648, 2.2373376
1: -10.1672506, -7.2483978, -10.1634455, -7.2160187, -2.6894011, 2.6254492
2: -5.5987539, -2.6825731, -5.5874386, -2.6928062, -2.9059477, 2.9048655
3: -12.1994295, -8.9905071, -12.2139626, -9.0044994, -2.9171696, 2.9422145
4: -8.8240595, -5.4225121, -8.7939548, -5.4005175, -3.1276650, 3.1034217
5: -0.9681263, 1.5843382, -0.9479667, 1.5675259, -2.5356522, 2.5323048
6: 5.0850258, 7.5114861, 5.0864253, 7.4669991, -2.3271623, 2.3304701
7: -18.8848362, -15.3919201, -18.8736801, -15.3140564, -2.8016434, 2.7558365
8: -1.6544197, 1.3965635, -1.6480575, 1.3725429, -3.0269625, 3.0446210
9: -8.9248066, -6.3680830, -8.8942327, -6.3537078, -2.2530394, 2.2893801

Time for backsubstitution: 12.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6196
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6221
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 4584

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6196

## Relational analysis of IS_A2_B2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4941141, upper bound: 1.4798063
time: 10.96 seconds

## Relational analysis of IS_A2_B2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4941139, upper bound: 1.4800074
time: 4.85 seconds

## BFS IS instance: IS_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -5.2933235, -2.7619438, -5.3194246, -2.7453840, -2.2269392, 2.2673120
1: -10.1672506, -7.2483978, -10.1881361, -7.2005663, -2.7103791, 2.6649194
2: -5.5987539, -2.6825731, -5.6257658, -2.6540394, -2.9447145, 2.9431927
3: -12.1994295, -8.9905071, -12.2383204, -8.9823093, -2.9419031, 2.9684749
4: -8.8240595, -5.4225121, -8.8480387, -5.3725290, -3.1641169, 3.1399789
5: -0.9681263, 1.5843382, -0.9878407, 1.5951147, -2.5632410, 2.5721788
6: 5.0850258, 7.5114861, 5.0415483, 7.5174193, -2.3477511, 2.3761780
7: -18.8848362, -15.3919201, -18.9039021, -15.2923937, -2.8148603, 2.7790456
8: -1.6544197, 1.3965635, -1.6748140, 1.4077725, -3.0621922, 3.0713775
9: -8.9248066, -6.3680830, -8.9481373, -6.3236427, -2.2851157, 2.3073654

Time for backsubstitution: 12.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6196
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6221
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: B, layer: 1, pos: 4584
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 5817
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6196

## Relational analysis of IS_A2_B2_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4941151, upper bound: 1.4798059
time: 7.49 seconds

## Relational analysis of IS_A2_B2_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4941149, upper bound: 1.4809509
time: 4.42 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 24.72 seconds
IS_A1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 24.72
Output dim: 6, lower bound: -1.4729700, upper bound: 1.4938138
IS_A1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 24.72
Output dim: 6, lower bound: -1.4729700, upper bound: 1.4941161
IS_A1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 24.72
Output dim: 6, lower bound: -1.4798064, upper bound: 1.4943111
IS_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 24.72
Output dim: 6, lower bound: -1.4798062, upper bound: 1.4943119
IS_A2_B1_B1_A2_B1, status: Status.VERIFIED, split count: 5, time: 24.72
Output dim: 6, lower bound: -1.4870414, upper bound: 1.4797306
IS_A2_B1_B1_A2_B2, status: Status.VERIFIED, split count: 5, time: 24.72
Output dim: 6, lower bound: -1.4870424, upper bound: 1.4806630
IS_A2_B1_B2_A1_B1, status: Status.VERIFIED, split count: 5, time: 24.72
Output dim: 6, lower bound: -1.4870355, upper bound: 1.4729698
IS_A2_B1_B2_A1_B2, status: Status.VERIFIED, split count: 5, time: 24.72
Output dim: 6, lower bound: -1.4870365, upper bound: 1.4738983
IS_A2_B1_B2_A2_B1, status: Status.VERIFIED, split count: 5, time: 24.72
Output dim: 6, lower bound: -1.4870355, upper bound: 1.4800068
IS_A2_B1_B2_A2_B2, status: Status.VERIFIED, split count: 5, time: 24.72
Output dim: 6, lower bound: -1.4870365, upper bound: 1.4809511
IS_A2_B2_B1_A2_B1, status: Status.VERIFIED, split count: 5, time: 24.72
Output dim: 6, lower bound: -1.4872859, upper bound: 1.4797304
IS_A2_B2_B1_A2_B2, status: Status.VERIFIED, split count: 5, time: 24.72
Output dim: 6, lower bound: -1.4872869, upper bound: 1.4806633
IS_A2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.72
Output dim: 6, lower bound: -1.4941141, upper bound: 1.4798063
IS_A2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.72
Output dim: 6, lower bound: -1.4941139, upper bound: 1.4800074
IS_A2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.72
Output dim: 6, lower bound: -1.4941151, upper bound: 1.4798059
IS_A2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.72
Output dim: 6, lower bound: -1.4941149, upper bound: 1.4809509

## BFS IS instance: IS_A1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -5.2489829, -2.7848151, -5.2814970, -2.7776022, -2.1776972, 2.1917105
1: -10.1104212, -7.2455597, -10.1310387, -7.2582793, -2.5933223, 2.6268687
2: -5.5617514, -2.7336950, -5.5789070, -2.6999831, -2.8617682, 2.8452120
3: -12.1861467, -9.0191727, -12.1810589, -9.0148296, -2.8907084, 2.8445215
4: -8.7736626, -5.4242344, -8.7905121, -5.4360418, -3.0379167, 3.0360854
5: -0.9339491, 1.5481961, -0.9580765, 1.5697855, -2.5037346, 2.5062726
6: 5.1053333, 7.4570303, 5.0985250, 7.4914799, -2.2896872, 2.3398199
7: -18.8491917, -15.3325748, -18.8342571, -15.4031363, -2.7180138, 2.7255774
8: -1.6163304, 1.3484097, -1.6377285, 1.3832436, -2.9995739, 2.9861381
9: -8.8607645, -6.3976526, -8.9017315, -6.3876200, -2.2447252, 2.2220769

Time for backsubstitution: 12.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4584
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 4584

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4558

## Relational analysis of IS_A1_A1_B2_A2_B1_B1

### Relational analysis result of IS_A1_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4699824, upper bound: 1.4938025
time: 4.31 seconds

## Relational analysis of IS_A1_A1_B2_A2_B1_B2

### Relational analysis result of IS_A1_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4729579, upper bound: 1.4938012
time: 4.33 seconds

## BFS IS instance: IS_A1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -5.2489829, -2.7848151, -5.3140841, -2.7523673, -2.2227411, 2.2363205
1: -10.1104212, -7.2455597, -10.1694145, -7.2032523, -2.6620421, 2.6692863
2: -5.5617514, -2.7336950, -5.6224208, -2.6673784, -2.8943729, 2.8887258
3: -12.1861467, -9.0191727, -12.2292652, -8.9846039, -2.9228377, 2.8969164
4: -8.7736626, -5.4242344, -8.8450422, -5.3804679, -3.0872416, 3.0920217
5: -0.9339491, 1.5481961, -0.9836173, 1.5907933, -2.5247424, 2.5318134
6: 5.1053333, 7.4570303, 5.0452557, 7.5151501, -2.3150711, 2.3917537
7: -18.8491917, -15.3325748, -18.8975811, -15.2984867, -2.7619920, 2.7897637
8: -1.6163304, 1.3484097, -1.6648712, 1.4040427, -3.0203731, 3.0132809
9: -8.8607645, -6.3976526, -8.9441595, -6.3388381, -2.2734661, 2.2641337

Time for backsubstitution: 12.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 6196
type: A, layer: 1, pos: 4584
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 4584

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4558

## Relational analysis of IS_A1_A1_B2_A2_B2_B1

### Relational analysis result of IS_A1_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4699824, upper bound: 1.4941037
time: 7.84 seconds

## Relational analysis of IS_A1_A1_B2_A2_B2_B2

### Relational analysis result of IS_A1_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4729579, upper bound: 1.4941031
time: 4.63 seconds

## BFS IS instance: IS_A1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -5.2746997, -2.7635746, -5.2677522, -2.7832770, -2.2101698, 2.2069128
1: -10.1634455, -7.2160187, -10.1138763, -7.2779884, -2.6239443, 2.6379988
2: -5.5874386, -2.6928062, -5.5728731, -2.7235405, -2.8638980, 2.8800669
3: -12.2139626, -9.0044994, -12.1716070, -9.0051813, -2.8954601, 2.8832450
4: -8.7939548, -5.4005175, -8.8036690, -5.4463263, -3.0427427, 3.0655723
5: -0.9479667, 1.5675259, -0.9540278, 1.5650328, -2.5129995, 2.5215535
6: 5.0864253, 7.4669991, 5.1048431, 7.5015497, -2.3490944, 2.3081348
7: -18.8736801, -15.3140564, -18.8597832, -15.4105330, -2.7339520, 2.7657499
8: -1.6480575, 1.3725429, -1.6223748, 1.3721352, -3.0201926, 2.9949176
9: -8.8942327, -6.3537078, -8.8914795, -6.4121380, -2.2456965, 2.2406237

Time for backsubstitution: 12.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6221
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 4584
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 4584

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6221

## Relational analysis of IS_A1_A2_B2_A2_B1_B1

### Relational analysis result of IS_A1_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4727271, upper bound: 1.4940342
time: 6.66 seconds

## Relational analysis of IS_A1_A2_B2_A2_B1_B2

### Relational analysis result of IS_A1_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4727271, upper bound: 1.4943107
time: 5.72 seconds

## BFS IS instance: IS_A1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -5.2746997, -2.7635746, -5.2933183, -2.7619526, -2.2250800, 2.2222614
1: -10.1634455, -7.2160187, -10.1672392, -7.2483997, -2.6254463, 2.6723413
2: -5.5874386, -2.6928062, -5.5987515, -2.6825814, -2.9048572, 2.9059453
3: -12.2139626, -9.0044994, -12.1994209, -8.9905109, -2.9422121, 2.9101100
4: -8.7939548, -5.4005175, -8.8240576, -5.4225216, -3.1140332, 3.1276617
5: -0.9479667, 1.5675259, -0.9681207, 1.5843351, -2.5323019, 2.5356464
6: 5.0864253, 7.4669991, 5.0850296, 7.5114851, -2.3304672, 2.3324432
7: -18.8736801, -15.3140564, -18.8848305, -15.3919277, -2.7649860, 2.8016391
8: -1.6480575, 1.3725429, -1.6544116, 1.3965607, -3.0446181, 3.0269544
9: -8.8942327, -6.3537078, -8.9248037, -6.3681040, -2.2566533, 2.2530360

Time for backsubstitution: 12.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 4584
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4558

## Relational analysis of IS_A1_A2_B2_A2_B2_B1

### Relational analysis result of IS_A1_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4768327, upper bound: 1.4942989
time: 4.51 seconds

## Relational analysis of IS_A1_A2_B2_A2_B2_B2

### Relational analysis result of IS_A1_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4797942, upper bound: 1.4942998
time: 4.49 seconds

## BFS IS instance: IS_A2_B2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -5.2677522, -2.7832770, -5.2746997, -2.7635746, -2.2069125, 2.2101696
1: -10.1138763, -7.2779884, -10.1634455, -7.2160187, -2.6379991, 2.6239438
2: -5.5728731, -2.7235405, -5.5874386, -2.6928062, -2.8800669, 2.8638980
3: -12.1716070, -9.0051813, -12.2139626, -9.0044994, -2.8832450, 2.8954597
4: -8.8036690, -5.4463263, -8.7939548, -5.4005175, -3.0655725, 3.0427423
5: -0.9540278, 1.5650328, -0.9479667, 1.5675259, -2.5215535, 2.5129995
6: 5.1048431, 7.5015497, 5.0864253, 7.4669991, -2.3081346, 2.3490944
7: -18.8597832, -15.4105330, -18.8736801, -15.3140564, -2.7657499, 2.7339516
8: -1.6223748, 1.3721352, -1.6480575, 1.3725429, -2.9949176, 3.0201926
9: -8.8914795, -6.4121380, -8.8942327, -6.3537078, -2.2406240, 2.2456963

Time for backsubstitution: 12.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 4584
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 4584

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6221

## Relational analysis of IS_A2_B2_B2_B1_A1_A1

### Relational analysis result of IS_A2_B2_B2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4870354, upper bound: 1.4727272
time: 5.03 seconds

## Relational analysis of IS_A2_B2_B2_B1_A1_A2

### Relational analysis result of IS_A2_B2_B2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4870354, upper bound: 1.4798061
time: 4.56 seconds

## BFS IS instance: IS_A2_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -5.2933183, -2.7619526, -5.2746997, -2.7635746, -2.2222614, 2.2250800
1: -10.1672392, -7.2483997, -10.1634455, -7.2160187, -2.6723423, 2.6254468
2: -5.5987515, -2.6825814, -5.5874386, -2.6928062, -2.9059453, 2.9048572
3: -12.1994209, -8.9905109, -12.2139626, -9.0044994, -2.9101095, 2.9422121
4: -8.8240576, -5.4225216, -8.7939548, -5.4005175, -3.1276617, 3.1140332
5: -0.9681207, 1.5843351, -0.9479667, 1.5675259, -2.5356464, 2.5323019
6: 5.0850296, 7.5114851, 5.0864253, 7.4669991, -2.3324428, 2.3304672
7: -18.8848305, -15.3919277, -18.8736801, -15.3140564, -2.8016391, 2.7649860
8: -1.6544116, 1.3965607, -1.6480575, 1.3725429, -3.0269544, 3.0446181
9: -8.9248037, -6.3681040, -8.8942327, -6.3537078, -2.2530360, 2.2566535

Time for backsubstitution: 12.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 6221
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 4584
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4558

## Relational analysis of IS_A2_B2_B2_B1_A2_A1

### Relational analysis result of IS_A2_B2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4941021, upper bound: 1.4770409
time: 4.48 seconds

## Relational analysis of IS_A2_B2_B2_B1_A2_A2

### Relational analysis result of IS_A2_B2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4941020, upper bound: 1.4799956
time: 5.02 seconds

## BFS IS instance: IS_A2_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -5.2677522, -2.7832770, -5.3194246, -2.7453840, -2.2136993, 2.2400162
1: -10.1138763, -7.2779884, -10.1881361, -7.2005663, -2.6589684, 2.6633997
2: -5.5728731, -2.7235405, -5.6257658, -2.6540394, -2.9188337, 2.9022253
3: -12.1716070, -9.0051813, -12.2383204, -8.9823093, -2.9074087, 2.9217954
4: -8.8036690, -5.4463263, -8.8480387, -5.3725290, -3.1020145, 3.0797138
5: -0.9540278, 1.5650328, -0.9878407, 1.5951147, -2.5491424, 2.5528736
6: 5.1048431, 7.5015497, 5.0415483, 7.5174193, -2.3292713, 2.3947659
7: -18.8597832, -15.4105330, -18.9039021, -15.2923937, -2.7789359, 2.7568693
8: -1.6223748, 1.3721352, -1.6748140, 1.4077725, -3.0301473, 3.0469491
9: -8.8914795, -6.4121380, -8.9481373, -6.3236427, -2.2725835, 2.2636759

Time for backsubstitution: 12.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: A, layer: 1, pos: 4584
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 4584

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6221

## Relational analysis of IS_A2_B2_B2_B2_A1_A1

### Relational analysis result of IS_A2_B2_B2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4871325, upper bound: 1.4727266
time: 6.44 seconds

## Relational analysis of IS_A2_B2_B2_B2_A1_A2

### Relational analysis result of IS_A2_B2_B2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4871325, upper bound: 1.4807568
time: 4.57 seconds

## BFS IS instance: IS_A2_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -5.2933183, -2.7619526, -5.3194246, -2.7453840, -2.2269354, 2.2550564
1: -10.1672392, -7.2483997, -10.1881361, -7.2005663, -2.6939044, 2.6649175
2: -5.5987515, -2.6825814, -5.6257658, -2.6540394, -2.9447122, 2.9431844
3: -12.1994209, -8.9905109, -12.2383204, -8.9823093, -2.9348431, 2.9684725
4: -8.8240576, -5.4225216, -8.8480387, -5.3725290, -3.1641145, 3.1504712
5: -0.9681207, 1.5843351, -0.9878407, 1.5951147, -2.5632353, 2.5721757
6: 5.0850296, 7.5114851, 5.0415483, 7.5174193, -2.3504009, 2.3761749
7: -18.8848305, -15.3919277, -18.9039021, -15.2923937, -2.8148561, 2.7882738
8: -1.6544116, 1.3965607, -1.6748140, 1.4077725, -3.0621841, 3.0713747
9: -8.9248037, -6.3681040, -8.9481373, -6.3236427, -2.2851124, 2.2727671

Time for backsubstitution: 12.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6221
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 4584
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4558

## Relational analysis of IS_A2_B2_B2_B2_A2_B1

### Relational analysis result of IS_A2_B2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4912373, upper bound: 1.4809395
time: 4.33 seconds

## Relational analysis of IS_A2_B2_B2_B2_A2_B2

### Relational analysis result of IS_A2_B2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4941990, upper bound: 1.4809394
time: 4.31 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 21.45 seconds
IS_A1_A1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 21.45
Output dim: 6, lower bound: -1.4699824, upper bound: 1.4938025
IS_A1_A1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 21.45
Output dim: 6, lower bound: -1.4729579, upper bound: 1.4938012
IS_A1_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 21.45
Output dim: 6, lower bound: -1.4699824, upper bound: 1.4941037
IS_A1_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 21.45
Output dim: 6, lower bound: -1.4729579, upper bound: 1.4941031
IS_A1_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 21.45
Output dim: 6, lower bound: -1.4727271, upper bound: 1.4940342
IS_A1_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 21.45
Output dim: 6, lower bound: -1.4727271, upper bound: 1.4943107
IS_A1_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 21.45
Output dim: 6, lower bound: -1.4768327, upper bound: 1.4942989
IS_A1_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 21.45
Output dim: 6, lower bound: -1.4797942, upper bound: 1.4942998
IS_A2_B2_B2_B1_A1_A1, status: Status.VERIFIED, split count: 6, time: 21.45
Output dim: 6, lower bound: -1.4870354, upper bound: 1.4727272
IS_A2_B2_B2_B1_A1_A2, status: Status.VERIFIED, split count: 6, time: 21.45
Output dim: 6, lower bound: -1.4870354, upper bound: 1.4798061
IS_A2_B2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 21.45
Output dim: 6, lower bound: -1.4941021, upper bound: 1.4770409
IS_A2_B2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 21.45
Output dim: 6, lower bound: -1.4941020, upper bound: 1.4799956
IS_A2_B2_B2_B2_A1_A1, status: Status.VERIFIED, split count: 6, time: 21.45
Output dim: 6, lower bound: -1.4871325, upper bound: 1.4727266
IS_A2_B2_B2_B2_A1_A2, status: Status.VERIFIED, split count: 6, time: 21.45
Output dim: 6, lower bound: -1.4871325, upper bound: 1.4807568
IS_A2_B2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 21.45
Output dim: 6, lower bound: -1.4912373, upper bound: 1.4809395
IS_A2_B2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 21.45
Output dim: 6, lower bound: -1.4941990, upper bound: 1.4809394

## BFS IS instance: IS_A1_A1_B2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -5.2477002, -2.7871976, -5.2742743, -2.7887924, -2.1651101, 2.1816015
1: -10.1093674, -7.2474694, -10.1255455, -7.2658758, -2.5829105, 2.6167390
2: -5.5583549, -2.7348199, -5.5664506, -2.7147577, -2.8435972, 2.8316307
3: -12.1837444, -9.0207567, -12.1707325, -9.0228138, -2.8780127, 2.8321319
4: -8.7669344, -5.4253836, -8.7681227, -5.4514971, -3.0160265, 3.0105751
5: -0.9321623, 1.5452507, -0.9474792, 1.5597303, -2.4918926, 2.4927299
6: 5.1069756, 7.4521103, 5.1150503, 7.4770751, -2.2727170, 2.3165593
7: -18.8449059, -15.3339882, -18.8202000, -15.4218578, -2.6953478, 2.7092962
8: -1.6149139, 1.3464689, -1.6276555, 1.3759875, -2.9909015, 2.9741244
9: -8.8590107, -6.3987885, -8.8946218, -6.4034939, -2.2262611, 2.2125742

Time for backsubstitution: 12.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4584
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 4584

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4558

## Relational analysis of IS_A1_A1_B2_A2_B1_B1_A1

### Relational analysis result of IS_A1_A1_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4699883, upper bound: 1.4908409
time: 4.11 seconds

## Relational analysis of IS_A1_A1_B2_A2_B1_B1_A2

### Relational analysis result of IS_A1_A1_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4699883, upper bound: 1.4938015
time: 4.40 seconds

## BFS IS instance: IS_A1_A1_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -5.2489839, -2.7848151, -5.2814937, -2.7776053, -2.1757765, 2.1917055
1: -10.1104202, -7.2455602, -10.1310349, -7.2582855, -2.5932007, 2.6265831
2: -5.5617504, -2.7336965, -5.5788994, -2.6999862, -2.8617642, 2.8452029
3: -12.1861477, -9.0191727, -12.1810522, -9.0148325, -2.8937650, 2.8440366
4: -8.7736607, -5.4242339, -8.7904930, -5.4360447, -3.0372233, 3.0239015
5: -0.9339489, 1.5481956, -0.9580728, 1.5697793, -2.5037282, 2.5062685
6: 5.1053348, 7.4570284, 5.0985298, 7.4914637, -2.2879162, 2.3398154
7: -18.8491898, -15.3325739, -18.8342419, -15.4031382, -2.7180099, 2.7204287
8: -1.6163306, 1.3484101, -1.6377246, 1.3832393, -2.9995699, 2.9861348
9: -8.8607645, -6.3976550, -8.9017267, -6.3876209, -2.2446451, 2.2211475

Time for backsubstitution: 12.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 5817
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 4584

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4558

## Relational analysis of IS_A1_A1_B2_A2_B1_B2_A1

### Relational analysis result of IS_A1_A1_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4729639, upper bound: 1.4908396
time: 4.30 seconds

## Relational analysis of IS_A1_A1_B2_A2_B1_B2_A2

### Relational analysis result of IS_A1_A1_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4729639, upper bound: 1.4938013
time: 4.51 seconds

## BFS IS instance: IS_A1_A1_B2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -5.2477002, -2.7871976, -5.3068581, -2.7635276, -2.2101207, 2.2249186
1: -10.1093674, -7.2474694, -10.1638727, -7.2109795, -2.6514416, 2.6591754
2: -5.5583549, -2.7348199, -5.6100302, -2.6822224, -2.8761325, 2.8752103
3: -12.1837444, -9.0207567, -12.2187557, -8.9926691, -2.9100504, 2.8844719
4: -8.7669344, -5.4253836, -8.8226013, -5.3960686, -3.0650635, 3.0664220
5: -0.9321623, 1.5452507, -0.9728829, 1.5807289, -2.5128913, 2.5181336
6: 5.1069756, 7.4521103, 5.0622859, 7.5006871, -2.2980371, 2.3647654
7: -18.8449059, -15.3339882, -18.8835297, -15.3172722, -2.7372541, 2.7734625
8: -1.6149139, 1.3464689, -1.6545544, 1.3968058, -3.0117197, 3.0010233
9: -8.8590107, -6.3987885, -8.9369688, -6.3548183, -2.2549744, 2.2544899

Time for backsubstitution: 12.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 6196
type: A, layer: 1, pos: 4584
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 4584

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4558

## Relational analysis of IS_A1_A1_B2_A2_B2_B1_A1

### Relational analysis result of IS_A1_A1_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4721283, upper bound: 1.4911413
time: 4.90 seconds

## Relational analysis of IS_A1_A1_B2_A2_B2_B1_A2

### Relational analysis result of IS_A1_A1_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4721283, upper bound: 1.4941027
time: 5.18 seconds

## BFS IS instance: IS_A1_A1_B2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -5.2489839, -2.7848151, -5.3140817, -2.7523713, -2.2208190, 2.2350740
1: -10.1104202, -7.2455602, -10.1694126, -7.2032576, -2.6619215, 2.6690001
2: -5.5617504, -2.7336965, -5.6224127, -2.6673813, -2.8943691, 2.8887162
3: -12.1861477, -9.0191727, -12.2292576, -8.9846058, -2.9258938, 2.8964386
4: -8.7736607, -5.4242339, -8.8450241, -5.3804703, -3.0865560, 3.0798473
5: -0.9339489, 1.5481956, -0.9836127, 1.5907873, -2.5247362, 2.5318084
6: 5.1053348, 7.4570284, 5.0452576, 7.5151343, -2.3133001, 2.3880115
7: -18.8491898, -15.3325739, -18.8975697, -15.2984896, -2.7590275, 2.7846158
8: -1.6163306, 1.3484101, -1.6648669, 1.4040389, -3.0203695, 3.0132771
9: -8.8607645, -6.3976550, -8.9441557, -6.3388400, -2.2725637, 2.2631936

Time for backsubstitution: 12.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 5817
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 4584

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4558

## Relational analysis of IS_A1_A1_B2_A2_B2_B2_A1

### Relational analysis result of IS_A1_A1_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4750844, upper bound: 1.4911410
time: 4.39 seconds

## Relational analysis of IS_A1_A1_B2_A2_B2_B2_A2

### Relational analysis result of IS_A1_A1_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4750844, upper bound: 1.4941024
time: 4.28 seconds

## BFS IS instance: IS_A1_A2_B2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -5.2746997, -2.7635746, -5.2614064, -2.7919488, -2.1827264, 2.1955380
1: -10.1634455, -7.2160187, -10.0963478, -7.2846117, -2.6157255, 2.6156085
2: -5.5874386, -2.6928062, -5.5563784, -2.7274427, -2.8599958, 2.8635721
3: -12.2139626, -9.0044994, -12.1622868, -9.0272360, -2.8730068, 2.8741293
4: -8.7939548, -5.4005175, -8.7731619, -5.4514956, -3.0382729, 3.0337787
5: -0.9479667, 1.5675259, -0.9487022, 1.5547765, -2.5027432, 2.5162282
6: 5.0864253, 7.4669991, 5.1125007, 7.4838295, -2.3294935, 2.3006196
7: -18.8736801, -15.3140564, -18.8156128, -15.4153891, -2.7294531, 2.7194710
8: -1.6480575, 1.3725429, -1.6165061, 1.3625689, -3.0106263, 2.9890490
9: -8.8942327, -6.3537078, -8.8723564, -6.4162254, -2.2413630, 2.2197931

Time for backsubstitution: 12.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 4584
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 4584

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4558

## Relational analysis of IS_A1_A2_B2_A2_B1_B1_A1

### Relational analysis result of IS_A1_A2_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4727152, upper bound: 1.4910696
time: 4.34 seconds

## Relational analysis of IS_A1_A2_B2_A2_B1_B1_A2

### Relational analysis result of IS_A1_A2_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4727152, upper bound: 1.4940221
time: 5.50 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 22.65 seconds
IS_A1_A1_B2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 22.65
Output dim: 6, lower bound: -1.4699883, upper bound: 1.4908409
IS_A1_A1_B2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 22.65
Output dim: 6, lower bound: -1.4699883, upper bound: 1.4938015
IS_A1_A1_B2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 22.65
Output dim: 6, lower bound: -1.4729639, upper bound: 1.4908396
IS_A1_A1_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 22.65
Output dim: 6, lower bound: -1.4729639, upper bound: 1.4938013
IS_A1_A1_B2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 22.65
Output dim: 6, lower bound: -1.4721283, upper bound: 1.4911413
IS_A1_A1_B2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 22.65
Output dim: 6, lower bound: -1.4721283, upper bound: 1.4941027
IS_A1_A1_B2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 22.65
Output dim: 6, lower bound: -1.4750844, upper bound: 1.4911410
IS_A1_A1_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 22.65
Output dim: 6, lower bound: -1.4750844, upper bound: 1.4941024
IS_A1_A2_B2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 22.65
Output dim: 6, lower bound: -1.4727152, upper bound: 1.4910696
IS_A1_A2_B2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 22.65
Output dim: 6, lower bound: -1.4727152, upper bound: 1.4940221
IS_A1_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 22.65
Output dim: 6, lower bound: -1.4727271, upper bound: 1.4943107
IS_A1_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 22.65
Output dim: 6, lower bound: -1.4768327, upper bound: 1.4942989
IS_A1_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 22.65
Output dim: 6, lower bound: -1.4797942, upper bound: 1.4942998
IS_A2_B2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 22.65
Output dim: 6, lower bound: -1.4941021, upper bound: 1.4770409
IS_A2_B2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 22.65
Output dim: 6, lower bound: -1.4941020, upper bound: 1.4799956
IS_A2_B2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 22.65
Output dim: 6, lower bound: -1.4912373, upper bound: 1.4809395
IS_A2_B2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 22.65
Output dim: 6, lower bound: -1.4941990, upper bound: 1.4809394
Binary search (step 2): status=Status.UNKNOWN, k_low=6, k_high=6, k_mid=6, eps_mid=0.0234375, abs_max=2.309645652770996
rel_dist={6: [-1.4944306407619283, 1.4944305738519086]}

## Binary Search with IS_dual Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01953125
execution time: 2407.72 seconds
