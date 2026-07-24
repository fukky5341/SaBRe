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
execution time: IAR + LP analysis = 14.04 + 34.05 = 48.09 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -1.8584840, upper bound: 1.8584796


# Binary Search by BASE starts (time budget: 3551.91 seconds, max iter: 100)

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
Binary search time: 195.68 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.01953125


# Individual Split (IS_dual_ind) starts
Time budget: 3356.23 seconds

## Binary search (step 0) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 6221
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 481

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6729517, upper bound: 1.6888901
time: 4.17 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6888884, upper bound: 1.6888886
time: 4.25 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 8.57 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 8.57
Output dim: 6, lower bound: -1.6729517, upper bound: 1.6888901
IS_A2, status: Status.UNKNOWN, split count: 1, time: 8.57
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

Time for backsubstitution: 12.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 481

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6729517, upper bound: 1.6729512
time: 4.41 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6729517, upper bound: 1.6888887
time: 4.08 seconds

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

Time for backsubstitution: 12.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 481

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6888894, upper bound: 1.6729512
time: 4.92 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6888894, upper bound: 1.6888888
time: 4.33 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 22.05 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 22.05
Output dim: 6, lower bound: -1.6729517, upper bound: 1.6729512
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 22.05
Output dim: 6, lower bound: -1.6729517, upper bound: 1.6888887
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 22.05
Output dim: 6, lower bound: -1.6888894, upper bound: 1.6729512
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 22.05
Output dim: 6, lower bound: -1.6888894, upper bound: 1.6888888

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

Time for backsubstitution: 12.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6221
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6221

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6729415, upper bound: 1.6652836
time: 4.31 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6729415, upper bound: 1.6729476
time: 4.29 seconds

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

Time for backsubstitution: 12.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6221
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6221

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6729415, upper bound: 1.6812172
time: 4.57 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6729415, upper bound: 1.6888809
time: 4.66 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -5.2933302, -2.7619328, -5.2482443, -2.7802007, -2.4071255, 2.3819132
1: -10.1672659, -7.2483907, -10.1429062, -7.2638254, -2.8692474, 2.8621154
2: -5.5987654, -2.6825676, -5.5605159, -2.7213488, -2.8774166, 2.8779483
3: -12.1994400, -8.9904881, -12.1751766, -9.0126343, -3.1195040, 3.1180553
4: -8.8240805, -5.4225054, -8.7701921, -5.4502583, -3.3579946, 3.3353758
5: -0.9681324, 1.5843486, -0.9283434, 1.5568354, -2.5249677, 2.5126920
6: 5.0850186, 7.5114970, 5.1297588, 7.4612336, -2.3762150, 2.3817382
7: -18.8848648, -15.3919125, -18.8547440, -15.4136238, -3.0394015, 3.0284843
8: -1.6544285, 1.3965702, -1.6276751, 1.3615103, -3.0159388, 3.0242453
9: -8.9248171, -6.3680735, -8.8711119, -6.3979683, -2.4643061, 2.4630537

Time for backsubstitution: 12.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6221
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6221

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6888782, upper bound: 1.6652766
time: 4.93 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6888782, upper bound: 1.6729406
time: 4.91 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -5.2933302, -2.7619328, -5.2933302, -2.7619328, -2.4168739, 2.4168742
1: -10.1672659, -7.2483907, -10.1672659, -7.2483907, -2.9029198, 2.9029193
2: -5.5987654, -2.6825676, -5.5987654, -2.6825676, -2.9161978, 2.9161978
3: -12.1994400, -8.9904881, -12.1994400, -8.9904881, -3.1452880, 3.1452880
4: -8.8240805, -5.4225054, -8.8240805, -5.4225054, -3.3797636, 3.3797636
5: -0.9681324, 1.5843486, -0.9681324, 1.5843486, -2.5524809, 2.5524809
6: 5.0850186, 7.5114970, 5.0850186, 7.5114970, -2.4264784, 2.4264784
7: -18.8848648, -15.3919125, -18.8848648, -15.3919125, -3.0538716, 3.0538721
8: -1.6544285, 1.3965702, -1.6544285, 1.3965702, -3.0509987, 3.0509987
9: -8.9248171, -6.3680735, -8.9248171, -6.3680735, -2.4869719, 2.4869719

Time for backsubstitution: 12.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6221
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6221

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6888792, upper bound: 1.6662721
time: 4.76 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6888792, upper bound: 1.6738916
time: 4.57 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 22.20 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 22.20
Output dim: 6, lower bound: -1.6729415, upper bound: 1.6652836
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 22.20
Output dim: 6, lower bound: -1.6729415, upper bound: 1.6729476
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 22.20
Output dim: 6, lower bound: -1.6729415, upper bound: 1.6812172
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 22.20
Output dim: 6, lower bound: -1.6729415, upper bound: 1.6888809
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 22.20
Output dim: 6, lower bound: -1.6888782, upper bound: 1.6652766
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 22.20
Output dim: 6, lower bound: -1.6888782, upper bound: 1.6729406
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 22.20
Output dim: 6, lower bound: -1.6888792, upper bound: 1.6662721
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 22.20
Output dim: 6, lower bound: -1.6888792, upper bound: 1.6738916

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -5.2418346, -2.7888536, -5.2482443, -2.7802007, -2.3521619, 2.3501263
1: -10.1253729, -7.2708321, -10.1429062, -7.2638254, -2.8253002, 2.8372512
2: -5.5440569, -2.7253385, -5.5605159, -2.7213488, -2.8227081, 2.8351774
3: -12.1657486, -9.0344219, -12.1751766, -9.0126343, -3.0855570, 3.0726328
4: -8.7394257, -5.4556971, -8.7701921, -5.4502583, -3.2746220, 3.3009143
5: -0.9226367, 1.5465655, -0.9283434, 1.5568354, -2.4794722, 2.4749088
6: 5.1388283, 7.4433804, 5.1297588, 7.4612336, -2.3224053, 2.3136215
7: -18.8108521, -15.4186230, -18.8547440, -15.4136238, -2.9607949, 3.0007219
8: -1.6212955, 1.3519731, -1.6276751, 1.3615103, -2.9828057, 2.9796481
9: -8.8518600, -6.4022117, -8.8711119, -6.3979683, -2.4115181, 2.4284213

Time for backsubstitution: 12.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6221

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6652738, upper bound: 1.6652747
time: 4.42 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6652738, upper bound: 1.6652765
time: 5.12 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -5.2747049, -2.7635653, -5.2482419, -2.7802041, -2.4062686, 2.3763559
1: -10.1634569, -7.2160163, -10.1428995, -7.2638264, -2.8642869, 2.9047694
2: -5.5874424, -2.6927962, -5.5605097, -2.7213502, -2.8660922, 2.8677135
3: -12.2139702, -9.0044994, -12.1751728, -9.0126457, -3.1380587, 3.1044846
4: -8.7939568, -5.4005060, -8.7701817, -5.4502611, -3.3322015, 3.3602495
5: -0.9479733, 1.5675302, -0.9283415, 1.5568280, -2.5048013, 2.4958715
6: 5.0864220, 7.4670000, 5.1297641, 7.4612269, -2.3748050, 2.3372359
7: -18.8736877, -15.3140488, -18.8547249, -15.4136267, -3.0286989, 3.0610371
8: -1.6480672, 1.3725476, -1.6276722, 1.3615041, -3.0095713, 3.0002198
9: -8.8942366, -6.3536835, -8.8711071, -6.3979721, -2.4555006, 2.4676285

Time for backsubstitution: 12.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6221

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6652738, upper bound: 1.6729481
time: 4.59 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6652738, upper bound: 1.6729499
time: 4.91 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -5.2418346, -2.7888536, -5.2933302, -2.7619328, -2.3709641, 2.3941410
1: -10.1253729, -7.2708321, -10.1672659, -7.2483907, -2.8412366, 2.8603201
2: -5.5440569, -2.7253385, -5.5987654, -2.6825676, -2.8614893, 2.8734269
3: -12.1657486, -9.0344219, -12.1994400, -8.9904881, -3.1088572, 3.0973825
4: -8.7394257, -5.4556971, -8.8240805, -5.4225054, -3.3037920, 3.3527036
5: -0.9226367, 1.5465655, -0.9681324, 1.5843486, -2.5069852, 2.5146980
6: 5.1388283, 7.4433804, 5.0850186, 7.5114970, -2.3726687, 2.3583617
7: -18.8108521, -15.4186230, -18.8848648, -15.3919125, -2.9826035, 3.0334477
8: -1.6212955, 1.3519731, -1.6544285, 1.3965702, -3.0178657, 3.0064015
9: -8.8518600, -6.4022117, -8.9248171, -6.3680735, -2.4423323, 2.4606752

Time for backsubstitution: 12.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 76

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6221

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6652681, upper bound: 1.6812069
time: 4.45 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6652681, upper bound: 1.6812066
time: 4.58 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -5.2747049, -2.7635653, -5.2933254, -2.7619393, -2.4250712, 2.4203699
1: -10.1634569, -7.2160163, -10.1672564, -7.2483954, -2.8802233, 2.9278388
2: -5.5874424, -2.6927962, -5.5987597, -2.6825695, -2.9048729, 2.9059634
3: -12.2139702, -9.0044994, -12.1994343, -8.9904985, -3.1613617, 3.1292357
4: -8.7939568, -5.4005060, -8.8240681, -5.4225087, -3.3613725, 3.3848710
5: -0.9479733, 1.5675302, -0.9681293, 1.5843428, -2.5323162, 2.5356593
6: 5.0864220, 7.4670000, 5.0850220, 7.5114903, -2.4250684, 2.3819780
7: -18.8736877, -15.3140488, -18.8848495, -15.3919163, -3.0505066, 3.0937791
8: -1.6480672, 1.3725476, -1.6544237, 1.3965654, -3.0446327, 3.0269713
9: -8.8942366, -6.3536835, -8.9248114, -6.3680773, -2.4863133, 2.4825668

Time for backsubstitution: 12.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 76

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6221

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6652681, upper bound: 1.6888805
time: 4.16 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6652681, upper bound: 1.6888796
time: 4.77 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -5.2868509, -2.7705951, -5.2482443, -2.7802007, -2.3961053, 2.3688612
1: -10.1497879, -7.2555809, -10.1429062, -7.2638254, -2.8484306, 2.8530192
2: -5.5822864, -2.6866245, -5.5605159, -2.7213488, -2.8609376, 2.8738914
3: -12.1900940, -9.0125284, -12.1751766, -9.0126343, -3.1103010, 3.0956559
4: -8.7934961, -5.4281244, -8.7701921, -5.4502583, -3.3266134, 3.3298998
5: -0.9622895, 1.5740933, -0.9283434, 1.5568354, -2.5191250, 2.5024366
6: 5.0948477, 7.4937735, 5.1297588, 7.4612336, -2.3663859, 2.3640146
7: -18.8406715, -15.3970165, -18.8547440, -15.4136238, -2.9931941, 3.0224199
8: -1.6476550, 1.3869667, -1.6276751, 1.3615103, -3.0091653, 3.0146418
9: -8.9056864, -6.3724403, -8.8711119, -6.3979683, -2.4435093, 2.4591312

Time for backsubstitution: 12.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 76

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6221

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6812062, upper bound: 1.6652676
time: 4.47 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6812062, upper bound: 1.6652684
time: 6.87 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -5.3194294, -2.7453735, -5.2482419, -2.7802041, -2.4422884, 2.3951054
1: -10.1881475, -7.2005653, -10.1428995, -7.2638264, -2.8876348, 2.9207349
2: -5.6257677, -2.6540318, -5.5605097, -2.7213502, -2.9044175, 2.9064779
3: -12.2383289, -8.9823074, -12.1751728, -9.0126457, -3.1627407, 3.1278343
4: -8.8480415, -5.3725166, -8.7701817, -5.4502611, -3.3841877, 3.3897104
5: -0.9878474, 1.5951185, -0.9283415, 1.5568280, -2.5446754, 2.5234599
6: 5.0415444, 7.5174222, 5.1297641, 7.4612269, -2.4196825, 2.3876581
7: -18.9039097, -15.2923832, -18.8547249, -15.4136267, -3.0614872, 3.0727103
8: -1.6748238, 1.4077749, -1.6276722, 1.3615041, -3.0363278, 3.0354471
9: -8.9481421, -6.3236170, -8.8711071, -6.3979721, -2.4875693, 2.4986091

Time for backsubstitution: 12.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6221

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6812062, upper bound: 1.6729412
time: 5.22 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6812062, upper bound: 1.6729415
time: 4.51 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -5.2868509, -2.7705951, -5.2933302, -2.7619328, -2.4058409, 2.4038632
1: -10.1497879, -7.2555809, -10.1672659, -7.2483907, -2.8821011, 2.8938217
2: -5.5822864, -2.6866245, -5.5987654, -2.6825676, -2.8997188, 2.9121408
3: -12.1900940, -9.0125284, -12.1994400, -8.9904881, -3.1359792, 3.1228890
4: -8.7934961, -5.4281244, -8.8240805, -5.4225054, -3.3480196, 3.3743067
5: -0.9622895, 1.5740933, -0.9681324, 1.5843486, -2.5466380, 2.5422258
6: 5.0948477, 7.4937735, 5.0850186, 7.5114970, -2.4166493, 2.4087548
7: -18.8406715, -15.3970165, -18.8848648, -15.3919125, -3.0076642, 3.0478196
8: -1.6476550, 1.3869667, -1.6544285, 1.3965702, -3.0442252, 3.0413952
9: -8.9056864, -6.3724403, -8.9248171, -6.3680735, -2.4661512, 2.4830501

Time for backsubstitution: 12.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6221

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6812048, upper bound: 1.6662634
time: 4.89 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6812048, upper bound: 1.6652685
time: 6.58 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -5.3194294, -2.7453735, -5.2933254, -2.7619393, -2.4601903, 2.4301882
1: -10.1881475, -7.2005653, -10.1672564, -7.2483954, -2.9213076, 2.9575138
2: -5.6257677, -2.6540318, -5.5987597, -2.6825695, -2.9431982, 2.9447279
3: -12.2383289, -8.9823074, -12.1994343, -8.9904985, -3.1887946, 3.1550670
4: -8.8480415, -5.3725166, -8.8240681, -5.4225087, -3.4058790, 3.4231572
5: -0.9878474, 1.5951185, -0.9681293, 1.5843428, -2.5721903, 2.5632477
6: 5.0415444, 7.5174222, 5.0850220, 7.5114903, -2.4699459, 2.4324002
7: -18.9039097, -15.2923832, -18.8848495, -15.3919163, -3.0759592, 3.1065085
8: -1.6748238, 1.4077749, -1.6544237, 1.3965654, -3.0713892, 3.0621986
9: -8.9481421, -6.3236170, -8.9248114, -6.3680773, -2.5102673, 2.5164537

Time for backsubstitution: 12.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6221

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6812048, upper bound: 1.6738917
time: 5.31 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6812048, upper bound: 1.6738925
time: 4.27 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 22.43 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.43
Output dim: 6, lower bound: -1.6652738, upper bound: 1.6652747
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.43
Output dim: 6, lower bound: -1.6652738, upper bound: 1.6652765
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.43
Output dim: 6, lower bound: -1.6652738, upper bound: 1.6729481
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.43
Output dim: 6, lower bound: -1.6652738, upper bound: 1.6729499
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.43
Output dim: 6, lower bound: -1.6652681, upper bound: 1.6812069
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.43
Output dim: 6, lower bound: -1.6652681, upper bound: 1.6812066
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.43
Output dim: 6, lower bound: -1.6652681, upper bound: 1.6888805
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.43
Output dim: 6, lower bound: -1.6652681, upper bound: 1.6888796
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.43
Output dim: 6, lower bound: -1.6812062, upper bound: 1.6652676
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.43
Output dim: 6, lower bound: -1.6812062, upper bound: 1.6652684
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.43
Output dim: 6, lower bound: -1.6812062, upper bound: 1.6729412
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.43
Output dim: 6, lower bound: -1.6812062, upper bound: 1.6729415
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.43
Output dim: 6, lower bound: -1.6812048, upper bound: 1.6662634
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.43
Output dim: 6, lower bound: -1.6812048, upper bound: 1.6652685
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.43
Output dim: 6, lower bound: -1.6812048, upper bound: 1.6738917
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.43
Output dim: 6, lower bound: -1.6812048, upper bound: 1.6738925

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -5.2418346, -2.7888536, -5.2418346, -2.7888536, -2.3391771, 2.3391771
1: -10.1253729, -7.2708321, -10.1253729, -7.2708321, -2.8163729, 2.8163729
2: -5.5440569, -2.7253385, -5.5440569, -2.7253385, -2.8187184, 2.8187184
3: -12.1657486, -9.0344219, -12.1657486, -9.0344219, -3.0634351, 3.0634356
4: -8.7394257, -5.4556971, -8.7394257, -5.4556971, -3.2693300, 3.2693305
5: -0.9226367, 1.5465655, -0.9226367, 1.5465655, -2.4692023, 2.4692023
6: 5.1388283, 7.4433804, 5.1388283, 7.4433804, -2.3045521, 2.3045521
7: -18.8108521, -15.4186230, -18.8108521, -15.4186230, -2.9548411, 2.9548411
8: -1.6212955, 1.3519731, -1.6212955, 1.3519731, -2.9732685, 2.9732685
9: -8.8518600, -6.4022117, -8.8518600, -6.4022117, -2.4076991, 2.4076993

Time for backsubstitution: 12.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4558

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6652142, upper bound: 1.6606678
time: 4.41 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6652566, upper bound: 1.6652648
time: 4.37 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -5.2418346, -2.7888536, -5.2747049, -2.7635653, -2.3654122, 2.3769779
1: -10.1253729, -7.2708321, -10.1634569, -7.2160163, -2.8776116, 2.8553638
2: -5.5440569, -2.7253385, -5.5874424, -2.6927962, -2.8512607, 2.8621039
3: -12.1657486, -9.0344219, -12.2139702, -9.0044994, -3.0952902, 3.1159463
4: -8.7394257, -5.4556971, -8.7939568, -5.4005060, -3.3286839, 3.3257337
5: -0.9226367, 1.5465655, -0.9479733, 1.5675302, -2.4901669, 2.4945388
6: 5.1388283, 7.4433804, 5.0864220, 7.4670000, -2.3281717, 2.3569584
7: -18.8108521, -15.4186230, -18.8736877, -15.3140488, -3.0151377, 3.0200815
8: -1.6212955, 1.3519731, -1.6480672, 1.3725476, -2.9938431, 3.0000403
9: -8.8518600, -6.4022117, -8.8942366, -6.3536835, -2.4468653, 2.4503441

Time for backsubstitution: 12.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4558

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6652142, upper bound: 1.6606686
time: 4.59 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6652566, upper bound: 1.6652662
time: 5.15 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -5.2747049, -2.7635653, -5.2418346, -2.7888536, -2.3769774, 2.3654122
1: -10.1634569, -7.2160163, -10.1253729, -7.2708321, -2.8553638, 2.8776112
2: -5.5874424, -2.6927962, -5.5440569, -2.7253385, -2.8621039, 2.8512607
3: -12.2139702, -9.0044994, -12.1657486, -9.0344219, -3.1159463, 3.0952902
4: -8.7939568, -5.4005060, -8.7394257, -5.4556971, -3.3257341, 3.3286839
5: -0.9479733, 1.5675302, -0.9226367, 1.5465655, -2.4945388, 2.4901669
6: 5.0864220, 7.4670000, 5.1388283, 7.4433804, -2.3569584, 2.3281717
7: -18.8736877, -15.3140488, -18.8108521, -15.4186230, -3.0200820, 3.0151377
8: -1.6480672, 1.3725476, -1.6212955, 1.3519731, -3.0000403, 2.9938431
9: -8.8942366, -6.3536835, -8.8518600, -6.4022117, -2.4503441, 2.4468651

Time for backsubstitution: 12.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4558

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6652142, upper bound: 1.6683984
time: 4.65 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6652566, upper bound: 1.6729288
time: 4.27 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -5.2747049, -2.7635653, -5.2747049, -2.7635653, -2.4238858, 2.4238858
1: -10.1634569, -7.2160163, -10.1634569, -7.2160163, -2.9245777, 2.9245777
2: -5.5874424, -2.6927962, -5.5874424, -2.6927962, -2.8946462, 2.8946462
3: -12.2139702, -9.0044994, -12.2139702, -9.0044994, -3.1477575, 3.1477575
4: -8.7939568, -5.4005060, -8.7939568, -5.4005060, -3.3764467, 3.3764462
5: -0.9479733, 1.5675302, -0.9479733, 1.5675302, -2.5155034, 2.5155034
6: 5.0864220, 7.4670000, 5.0864220, 7.4670000, -2.3805780, 2.3805780
7: -18.8736877, -15.3140488, -18.8736877, -15.3140488, -3.0805960, 3.0805964
8: -1.6480672, 1.3725476, -1.6480672, 1.3725476, -3.0206149, 3.0206149
9: -8.8942366, -6.3536835, -8.8942366, -6.3536835, -2.4894783, 2.4894781

Time for backsubstitution: 12.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4558

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6652142, upper bound: 1.6683994
time: 4.29 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6652566, upper bound: 1.6729324
time: 5.81 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -5.2418346, -2.7888536, -5.2868509, -2.7705951, -2.3579121, 2.3831205
1: -10.1253729, -7.2708321, -10.1497879, -7.2555809, -2.8321400, 2.8395014
2: -5.5440569, -2.7253385, -5.5822864, -2.6866245, -2.8574324, 2.8569479
3: -12.1657486, -9.0344219, -12.1900940, -9.0125284, -3.0864592, 3.0881791
4: -8.7394257, -5.4556971, -8.7934961, -5.4281244, -3.2983160, 3.3213234
5: -0.9226367, 1.5465655, -0.9622895, 1.5740933, -2.4967301, 2.5088549
6: 5.1388283, 7.4433804, 5.0948477, 7.4937735, -2.3549452, 2.3485327
7: -18.8108521, -15.4186230, -18.8406715, -15.3970165, -2.9765387, 2.9872403
8: -1.6212955, 1.3519731, -1.6476550, 1.3869667, -3.0082622, 2.9996281
9: -8.8518600, -6.4022117, -8.9056864, -6.3724403, -2.4384089, 2.4398787

Time for backsubstitution: 12.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4558

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6652084, upper bound: 1.6765978
time: 4.61 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6652509, upper bound: 1.6811973
time: 5.06 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -5.2418346, -2.7888536, -5.3194294, -2.7453735, -2.3841615, 2.4205337
1: -10.1253729, -7.2708321, -10.1881475, -7.2005653, -2.8935771, 2.8787122
2: -5.5440569, -2.7253385, -5.6257677, -2.6540318, -2.8900251, 2.9004292
3: -12.1657486, -9.0344219, -12.2383289, -8.9823074, -3.1186380, 3.1406288
4: -8.7394257, -5.4556971, -8.8480415, -5.3725166, -3.3581438, 3.3777194
5: -0.9226367, 1.5465655, -0.9878474, 1.5951185, -2.5177553, 2.5344129
6: 5.1388283, 7.4433804, 5.0415444, 7.5174222, -2.3785939, 2.4018359
7: -18.8108521, -15.4186230, -18.9039097, -15.2923832, -3.0268111, 3.0528722
8: -1.6212955, 1.3519731, -1.6748238, 1.4077749, -3.0290704, 3.0267968
9: -8.8518600, -6.4022117, -8.9481421, -6.3236170, -2.4778457, 2.4825974

Time for backsubstitution: 12.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4558

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6652084, upper bound: 1.6765988
time: 4.69 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6652509, upper bound: 1.6812003
time: 7.90 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -5.2747049, -2.7635653, -5.2868509, -2.7705951, -2.3957124, 2.4093554
1: -10.1634569, -7.2160163, -10.1497879, -7.2555809, -2.8711319, 2.9007406
2: -5.5874424, -2.6927962, -5.5822864, -2.6866245, -2.9008179, 2.8894901
3: -12.2139702, -9.0044994, -12.1900940, -9.0125284, -3.1389694, 3.1200337
4: -8.7939568, -5.4005060, -8.7934961, -5.4281244, -3.3547192, 3.3532248
5: -0.9479733, 1.5675302, -0.9622895, 1.5740933, -2.5220666, 2.5298195
6: 5.0864220, 7.4670000, 5.0948477, 7.4937735, -2.4073515, 2.3721523
7: -18.8736877, -15.3140488, -18.8406715, -15.3970165, -3.0417795, 3.0475395
8: -1.6480672, 1.3725476, -1.6476550, 1.3869667, -3.0350339, 3.0202026
9: -8.8942366, -6.3536835, -8.9056864, -6.3724403, -2.4810543, 2.4617741

Time for backsubstitution: 12.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4558

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6652084, upper bound: 1.6843259
time: 4.60 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6652509, upper bound: 1.6888611
time: 4.55 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -5.2747049, -2.7635653, -5.3194294, -2.7453735, -2.4426351, 2.4585254
1: -10.1634569, -7.2160163, -10.1881475, -7.2005653, -2.9405432, 2.9479246
2: -5.5874424, -2.6927962, -5.6257677, -2.6540318, -2.9334106, 2.9329715
3: -12.2139702, -9.0044994, -12.2383289, -8.9823074, -3.1711073, 3.1724396
4: -8.7939568, -5.4005060, -8.8480415, -5.3725166, -3.4059057, 3.4097309
5: -0.9479733, 1.5675302, -0.9878474, 1.5951185, -2.5430918, 2.5553775
6: 5.0864220, 7.4670000, 5.0415444, 7.5174222, -2.4310002, 2.4254556
7: -18.8736877, -15.3140488, -18.9039097, -15.2923832, -3.0922694, 3.1134448
8: -1.6480672, 1.3725476, -1.6748238, 1.4077749, -3.0558422, 3.0473714
9: -8.8942366, -6.3536835, -8.9481421, -6.3236170, -2.5204582, 2.5044930

Time for backsubstitution: 12.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4558

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6652084, upper bound: 1.6843271
time: 4.43 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6652509, upper bound: 1.6888630
time: 5.33 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -5.2868509, -2.7705951, -5.2418346, -2.7888536, -2.3831205, 2.3579121
1: -10.1497879, -7.2555809, -10.1253729, -7.2708321, -2.8395014, 2.8321409
2: -5.5822864, -2.6866245, -5.5440569, -2.7253385, -2.8569479, 2.8574324
3: -12.1900940, -9.0125284, -12.1657486, -9.0344219, -3.0881791, 3.0864587
4: -8.7934961, -5.4281244, -8.7394257, -5.4556971, -3.3213234, 3.2983160
5: -0.9622895, 1.5740933, -0.9226367, 1.5465655, -2.5088549, 2.4967301
6: 5.0948477, 7.4937735, 5.1388283, 7.4433804, -2.3485327, 2.3549452
7: -18.8406715, -15.3970165, -18.8108521, -15.4186230, -2.9872403, 2.9765401
8: -1.6476550, 1.3869667, -1.6212955, 1.3519731, -2.9996281, 3.0082622
9: -8.9056864, -6.3724403, -8.8518600, -6.4022117, -2.4398787, 2.4384093

Time for backsubstitution: 12.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4558

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6811466, upper bound: 1.6606619
time: 4.34 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6811890, upper bound: 1.6652593
time: 4.43 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -5.2868509, -2.7705951, -5.2747049, -2.7635653, -2.4093556, 2.3957126
1: -10.1497879, -7.2555809, -10.1634569, -7.2160163, -2.9007401, 2.8711319
2: -5.5822864, -2.6866245, -5.5874424, -2.6927962, -2.8894901, 2.9008179
3: -12.1900940, -9.0125284, -12.2139702, -9.0044994, -3.1200342, 3.1389699
4: -8.7934961, -5.4281244, -8.7939568, -5.4005060, -3.3532245, 3.3547196
5: -0.9622895, 1.5740933, -0.9479733, 1.5675302, -2.5298195, 2.5220666
6: 5.0948477, 7.4937735, 5.0864220, 7.4670000, -2.3721523, 2.4073515
7: -18.8406715, -15.3970165, -18.8736877, -15.3140488, -3.0475388, 3.0417795
8: -1.6476550, 1.3869667, -1.6480672, 1.3725476, -3.0202026, 3.0350339
9: -8.9056864, -6.3724403, -8.8942366, -6.3536835, -2.4617746, 2.4810543

Time for backsubstitution: 12.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4558

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6811466, upper bound: 1.6606627
time: 8.19 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6811890, upper bound: 1.6652601
time: 4.69 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -5.3194294, -2.7453735, -5.2418346, -2.7888536, -2.4205337, 2.3841615
1: -10.1881475, -7.2005653, -10.1253729, -7.2708321, -2.8787127, 2.8935771
2: -5.6257677, -2.6540318, -5.5440569, -2.7253385, -2.9004292, 2.8900251
3: -12.2383289, -8.9823074, -12.1657486, -9.0344219, -3.1406293, 3.1186385
4: -8.8480415, -5.3725166, -8.7394257, -5.4556971, -3.3777189, 3.3581448
5: -0.9878474, 1.5951185, -0.9226367, 1.5465655, -2.5344129, 2.5177553
6: 5.0415444, 7.5174222, 5.1388283, 7.4433804, -2.4018359, 2.3785939
7: -18.9039097, -15.2923832, -18.8108521, -15.4186230, -3.0528722, 3.0268109
8: -1.6748238, 1.4077749, -1.6212955, 1.3519731, -3.0267968, 3.0290704
9: -8.9481421, -6.3236170, -8.8518600, -6.4022117, -2.4825976, 2.4778457

Time for backsubstitution: 12.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4558

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6811466, upper bound: 1.6683925
time: 4.34 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6811890, upper bound: 1.6729233
time: 4.74 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -5.3194294, -2.7453735, -5.2747049, -2.7635653, -2.4585252, 2.4426355
1: -10.1881475, -7.2005653, -10.1634569, -7.2160163, -2.9479246, 2.9405432
2: -5.6257677, -2.6540318, -5.5874424, -2.6927962, -2.9329715, 2.9334106
3: -12.2383289, -8.9823074, -12.2139702, -9.0044994, -3.1724396, 3.1711073
4: -8.8480415, -5.3725166, -8.7939568, -5.4005060, -3.4097316, 3.4059062
5: -0.9878474, 1.5951185, -0.9479733, 1.5675302, -2.5553775, 2.5430918
6: 5.0415444, 7.5174222, 5.0864220, 7.4670000, -2.4254556, 2.4310002
7: -18.9039097, -15.2923832, -18.8736877, -15.3140488, -3.1134453, 3.0922697
8: -1.6748238, 1.4077749, -1.6480672, 1.3725476, -3.0473714, 3.0558422
9: -8.9481421, -6.3236170, -8.8942366, -6.3536835, -2.5044930, 2.5204585

Time for backsubstitution: 12.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4558

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6811466, upper bound: 1.6683937
time: 4.26 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6652566, upper bound: 1.6729262
time: 6.18 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -5.2868509, -2.7705951, -5.2868509, -2.7705951, -2.3928294, 2.3928301
1: -10.1497879, -7.2555809, -10.1497879, -7.2555809, -2.8730040, 2.8730030
2: -5.5822864, -2.6866245, -5.5822864, -2.6866245, -2.8956618, 2.8956618
3: -12.1900940, -9.0125284, -12.1900940, -9.0125284, -3.1135807, 3.1135807
4: -8.7934961, -5.4281244, -8.7934961, -5.4281244, -3.3425627, 3.3425627
5: -0.9622895, 1.5740933, -0.9622895, 1.5740933, -2.5363827, 2.5363827
6: 5.0948477, 7.4937735, 5.0948477, 7.4937735, -2.3989258, 2.3989258
7: -18.8406715, -15.3970165, -18.8406715, -15.3970165, -3.0016127, 3.0016122
8: -1.6476550, 1.3869667, -1.6476550, 1.3869667, -3.0346217, 3.0346217
9: -8.9056864, -6.3724403, -8.9056864, -6.3724403, -2.4622293, 2.4622295

Time for backsubstitution: 12.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4558

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6811451, upper bound: 1.6615701
time: 4.31 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6811875, upper bound: 1.6662549
time: 4.82 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -5.2868509, -2.7705951, -5.3194294, -2.7453735, -2.4191608, 2.4308732
1: -10.1497879, -7.2555809, -10.1881475, -7.2005653, -2.9344406, 2.9122148
2: -5.5822864, -2.6866245, -5.6257677, -2.6540318, -2.9282546, 2.9391432
3: -12.1900940, -9.0125284, -12.2383289, -8.9823074, -3.1457605, 3.1664047
4: -8.7934961, -5.4281244, -8.8480415, -5.3725166, -3.3914094, 3.3992438
5: -0.9622895, 1.5740933, -0.9878474, 1.5951185, -2.5574079, 2.5619407
6: 5.0948477, 7.4937735, 5.0415444, 7.5174222, -2.4225745, 2.4522290
7: -18.8406715, -15.3970165, -18.9039097, -15.2923832, -3.0602689, 3.0672445
8: -1.6476550, 1.3869667, -1.6748238, 1.4077749, -3.0554299, 3.0617905
9: -8.9056864, -6.3724403, -8.9481421, -6.3236170, -2.4955907, 2.5050087

Time for backsubstitution: 12.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4558

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6811451, upper bound: 1.6615710
time: 4.35 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6811875, upper bound: 1.6662558
time: 4.41 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -5.3194294, -2.7453735, -5.2868509, -2.7705951, -2.4308729, 2.4191606
1: -10.1881475, -7.2005653, -10.1497879, -7.2555809, -2.9122152, 2.9344406
2: -5.6257677, -2.6540318, -5.5822864, -2.6866245, -2.9391432, 2.9282546
3: -12.2383289, -8.9823074, -12.1900940, -9.0125284, -3.1664047, 3.1457601
4: -8.8480415, -5.3725166, -8.7934961, -5.4281244, -3.3992443, 3.3914099
5: -0.9878474, 1.5951185, -0.9622895, 1.5740933, -2.5619407, 2.5574079
6: 5.0415444, 7.5174222, 5.0948477, 7.4937735, -2.4522290, 2.4225745
7: -18.9039097, -15.2923832, -18.8406715, -15.3970165, -3.0672445, 3.0602689
8: -1.6748238, 1.4077749, -1.6476550, 1.3869667, -3.0617905, 3.0554299
9: -8.9481421, -6.3236170, -8.9056864, -6.3724403, -2.5050082, 2.4955907

Time for backsubstitution: 12.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4558

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6811451, upper bound: 1.6692485
time: 5.33 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6811875, upper bound: 1.6738739
time: 4.52 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -5.3194294, -2.7453735, -5.3194294, -2.7453735, -2.4778776, 2.4778776
1: -10.1881475, -7.2005653, -10.1881475, -7.2005653, -2.9772115, 2.9772117
2: -5.6257677, -2.6540318, -5.6257677, -2.6540318, -2.9717360, 2.9717360
3: -12.2383289, -8.9823074, -12.2383289, -8.9823074, -3.1985407, 3.1985412
4: -8.8480415, -5.3725166, -8.8480415, -5.3725166, -3.4479957, 3.4479961
5: -0.9878474, 1.5951185, -0.9878474, 1.5951185, -2.5829659, 2.5829659
6: 5.0415444, 7.5174222, 5.0415444, 7.5174222, -2.4758778, 2.4758778
7: -18.9039097, -15.2923832, -18.9039097, -15.2923832, -3.1261744, 3.1261744
8: -1.6748238, 1.4077749, -1.6748238, 1.4077749, -3.0825987, 3.0825987
9: -8.9481421, -6.3236170, -8.9481421, -6.3236170, -2.5383472, 2.5383472

Time for backsubstitution: 12.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4558

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6811451, upper bound: 1.6692497
time: 4.20 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6811875, upper bound: 1.6738756
time: 4.41 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 21.46 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 21.46
Output dim: 6, lower bound: -1.6652142, upper bound: 1.6606678
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.46
Output dim: 6, lower bound: -1.6652566, upper bound: 1.6652648
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.46
Output dim: 6, lower bound: -1.6652142, upper bound: 1.6606686
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.46
Output dim: 6, lower bound: -1.6652566, upper bound: 1.6652662
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 21.46
Output dim: 6, lower bound: -1.6652142, upper bound: 1.6683984
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.46
Output dim: 6, lower bound: -1.6652566, upper bound: 1.6729288
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.46
Output dim: 6, lower bound: -1.6652142, upper bound: 1.6683994
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.46
Output dim: 6, lower bound: -1.6652566, upper bound: 1.6729324
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 21.46
Output dim: 6, lower bound: -1.6652084, upper bound: 1.6765978
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.46
Output dim: 6, lower bound: -1.6652509, upper bound: 1.6811973
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.46
Output dim: 6, lower bound: -1.6652084, upper bound: 1.6765988
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.46
Output dim: 6, lower bound: -1.6652509, upper bound: 1.6812003
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 21.46
Output dim: 6, lower bound: -1.6652084, upper bound: 1.6843259
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.46
Output dim: 6, lower bound: -1.6652509, upper bound: 1.6888611
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.46
Output dim: 6, lower bound: -1.6652084, upper bound: 1.6843271
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.46
Output dim: 6, lower bound: -1.6652509, upper bound: 1.6888630
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 21.46
Output dim: 6, lower bound: -1.6811466, upper bound: 1.6606619
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.46
Output dim: 6, lower bound: -1.6811890, upper bound: 1.6652593
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.46
Output dim: 6, lower bound: -1.6811466, upper bound: 1.6606627
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.46
Output dim: 6, lower bound: -1.6811890, upper bound: 1.6652601
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 21.46
Output dim: 6, lower bound: -1.6811466, upper bound: 1.6683925
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.46
Output dim: 6, lower bound: -1.6811890, upper bound: 1.6729233
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.46
Output dim: 6, lower bound: -1.6811466, upper bound: 1.6683937
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.46
Output dim: 6, lower bound: -1.6652566, upper bound: 1.6729262
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 21.46
Output dim: 6, lower bound: -1.6811451, upper bound: 1.6615701
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.46
Output dim: 6, lower bound: -1.6811875, upper bound: 1.6662549
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.46
Output dim: 6, lower bound: -1.6811451, upper bound: 1.6615710
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.46
Output dim: 6, lower bound: -1.6811875, upper bound: 1.6662558
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 21.46
Output dim: 6, lower bound: -1.6811451, upper bound: 1.6692485
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.46
Output dim: 6, lower bound: -1.6811875, upper bound: 1.6738739
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.46
Output dim: 6, lower bound: -1.6811451, upper bound: 1.6692497
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.46
Output dim: 6, lower bound: -1.6811875, upper bound: 1.6738756

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -5.2346039, -2.8000364, -5.2410846, -2.7902617, -2.3299541, 2.3272204
1: -10.1198444, -7.2782874, -10.1247587, -7.2719607, -2.8078580, 2.8055105
2: -5.5315886, -2.7399569, -5.5420475, -2.7259994, -2.8055892, 2.8020906
3: -12.1553612, -9.0423527, -12.1643314, -9.0353384, -3.0516925, 3.0517530
4: -8.7170601, -5.4710073, -8.7354650, -5.4563775, -3.2445564, 3.2506216
5: -0.9121457, 1.5365093, -0.9215785, 1.5448284, -2.4569740, 2.4580879
6: 5.1548109, 7.4289107, 5.1398025, 7.4404793, -2.2856684, 2.2891083
7: -18.7968864, -15.4372711, -18.8083363, -15.4194593, -2.9394875, 2.9340191
8: -1.6115003, 1.3447394, -1.6204586, 1.3508272, -2.9623275, 2.9651980
9: -8.8446941, -6.4180069, -8.8508396, -6.4028788, -2.3986349, 2.3899627

Time for backsubstitution: 12.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4558

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6606685, upper bound: 1.6606690
time: 4.36 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6606685, upper bound: 1.6606689
time: 4.80 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -5.2418308, -2.7888572, -5.2418346, -2.7888536, -2.3391728, 2.3375764
1: -10.1253710, -7.2708378, -10.1253729, -7.2708321, -2.8161001, 2.8164034
2: -5.5440469, -2.7253413, -5.5440569, -2.7253385, -2.8187084, 2.8187156
3: -12.1657419, -9.0344257, -12.1657486, -9.0344219, -3.0630631, 3.0669575
4: -8.7394075, -5.4556999, -8.7394257, -5.4556971, -3.2606554, 3.2686429
5: -0.9226316, 1.5465596, -0.9226367, 1.5465655, -2.4691973, 2.4691963
6: 5.1388321, 7.4433641, 5.1388283, 7.4433804, -2.3045483, 2.3045359
7: -18.8108368, -15.4186287, -18.8108521, -15.4186230, -2.9506702, 2.9548373
8: -1.6212916, 1.3519673, -1.6212955, 1.3519731, -2.9732647, 2.9732628
9: -8.8518543, -6.4022141, -8.8518600, -6.4022117, -2.4078827, 2.4076965

Time for backsubstitution: 12.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4558

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6606685, upper bound: 1.6652238
time: 4.44 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6606683, upper bound: 1.6652234
time: 4.46 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -5.2346039, -2.8000364, -5.2739596, -2.7649689, -2.3561869, 2.3650255
1: -10.1198444, -7.2782874, -10.1628351, -7.2171402, -2.8690968, 2.8445015
2: -5.5315886, -2.7399569, -5.5854416, -2.6934569, -2.8381317, 2.8454847
3: -12.1553612, -9.0423527, -12.2125072, -9.0054302, -3.0835328, 3.1042597
4: -8.7170601, -5.4710073, -8.7899857, -5.4011855, -3.3039155, 3.3070097
5: -0.9121457, 1.5365093, -0.9469142, 1.5657938, -2.4779396, 2.4834235
6: 5.1548109, 7.4289107, 5.0873923, 7.4640956, -2.3092847, 2.3415184
7: -18.7968864, -15.4372711, -18.8711643, -15.3148785, -2.9995909, 2.9992833
8: -1.6115003, 1.3447394, -1.6472318, 1.3714018, -2.9829021, 2.9919713
9: -8.8446941, -6.4180069, -8.8932037, -6.3543534, -2.4378433, 2.4325898

Time for backsubstitution: 12.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4558

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6683989, upper bound: 1.6606686
time: 4.55 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6683989, upper bound: 1.6606688
time: 4.69 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -5.2418308, -2.7888572, -5.2747049, -2.7635653, -2.3654084, 2.3753767
1: -10.1253710, -7.2708378, -10.1634569, -7.2160163, -2.8773398, 2.8553591
2: -5.5440469, -2.7253413, -5.5874424, -2.6927962, -2.8512506, 2.8621011
3: -12.1657419, -9.0344257, -12.2139702, -9.0044994, -3.0949183, 3.1194775
4: -8.7394075, -5.4556999, -8.7939568, -5.4005060, -3.3199949, 3.3250461
5: -0.9226316, 1.5465596, -0.9479733, 1.5675302, -2.4901619, 2.4945328
6: 5.1388321, 7.4433641, 5.0864220, 7.4670000, -2.3281679, 2.3569422
7: -18.8108368, -15.4186287, -18.8736877, -15.3140488, -3.0109701, 3.0200777
8: -1.6212916, 1.3519673, -1.6480672, 1.3725476, -2.9938393, 3.0000346
9: -8.8518543, -6.4022141, -8.8942366, -6.3536835, -2.4462378, 2.4503412

Time for backsubstitution: 12.73 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=6, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=2.363316535949707
rel_dist={6: [-1.6889095697923429, 1.6889090371909665]}

## Binary search (step 1) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 6221
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 481

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5461592, upper bound: 1.5620941
time: 3.95 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5620923, upper bound: 1.5620939
time: 4.38 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 8.49 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 8.49
Output dim: 6, lower bound: -1.5461592, upper bound: 1.5620941
IS_A2, status: Status.UNKNOWN, split count: 1, time: 8.49
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

Time for backsubstitution: 12.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 76

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 481

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5461592, upper bound: 1.5461598
time: 4.32 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5461592, upper bound: 1.5620939
time: 4.34 seconds

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

Time for backsubstitution: 12.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 481

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5620933, upper bound: 1.5461598
time: 3.96 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5620933, upper bound: 1.5620939
time: 4.02 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 20.86 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 20.86
Output dim: 6, lower bound: -1.5461592, upper bound: 1.5461598
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 20.86
Output dim: 6, lower bound: -1.5461592, upper bound: 1.5620939
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 20.86
Output dim: 6, lower bound: -1.5620933, upper bound: 1.5461598
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 20.86
Output dim: 6, lower bound: -1.5620933, upper bound: 1.5620939

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -5.2482443, -2.7802007, -5.2482443, -2.7802007, -2.2392144, 2.2392147
1: -10.1429062, -7.2638254, -10.1429062, -7.2638254, -2.6964359, 2.6964359
2: -5.5605159, -2.7213488, -5.5605159, -2.7213488, -2.8391671, 2.8391671
3: -12.1751766, -9.0126343, -12.1751766, -9.0126343, -2.9533863, 2.9533863
4: -8.7701921, -5.4502583, -8.7701921, -5.4502583, -3.1397781, 3.1397786
5: -0.9283434, 1.5568354, -0.9283434, 1.5568354, -2.4851789, 2.4851789
6: 5.1297588, 7.4612336, 5.1297588, 7.4612336, -2.3137436, 2.3137436
7: -18.8547440, -15.4136238, -18.8547440, -15.4136238, -2.8164430, 2.8164425
8: -1.6276751, 1.3615103, -1.6276751, 1.3615103, -2.9891853, 2.9891853
9: -8.8711119, -6.3979683, -8.8711119, -6.3979683, -2.3056417, 2.3056414

Time for backsubstitution: 12.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6221
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6221

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5461482, upper bound: 1.5384910
time: 4.43 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5461482, upper bound: 1.5461530
time: 4.20 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -5.2482443, -2.7802007, -5.2933302, -2.7619328, -2.2580166, 2.2832294
1: -10.1429062, -7.2638254, -10.1672659, -7.2483907, -2.7123723, 2.7195048
2: -5.5605159, -2.7213488, -5.5987654, -2.6825676, -2.8779483, 2.8774166
3: -12.1751766, -9.0126343, -12.1994400, -8.9904881, -2.9766874, 2.9781370
4: -8.7701921, -5.4502583, -8.8240805, -5.4225054, -3.1689491, 3.1888561
5: -0.9283434, 1.5568354, -0.9681324, 1.5843486, -2.5126920, 2.5249677
6: 5.1297588, 7.4612336, 5.0850186, 7.5114970, -2.3362956, 2.3599682
7: -18.8547440, -15.4136238, -18.8848648, -15.3919125, -2.8382516, 2.8491688
8: -1.6276751, 1.3615103, -1.6544285, 1.3965702, -3.0242453, 3.0159388
9: -8.8711119, -6.3979683, -8.9248171, -6.3680735, -2.3364553, 2.3366842

Time for backsubstitution: 12.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6221
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6221

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5461482, upper bound: 1.5544220
time: 4.59 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5461482, upper bound: 1.5620826
time: 4.10 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -5.2933302, -2.7619328, -5.2482443, -2.7802007, -2.2832294, 2.2580171
1: -10.1672659, -7.2483907, -10.1429062, -7.2638254, -2.7195053, 2.7123718
2: -5.5987654, -2.6825676, -5.5605159, -2.7213488, -2.8774166, 2.8779483
3: -12.1994400, -8.9904881, -12.1751766, -9.0126343, -2.9781370, 2.9766874
4: -8.8240805, -5.4225054, -8.7701921, -5.4502583, -3.1888566, 3.1689491
5: -0.9681324, 1.5843486, -0.9283434, 1.5568354, -2.5249677, 2.5126920
6: 5.0850186, 7.5114970, 5.1297588, 7.4612336, -2.3599682, 2.3362956
7: -18.8848648, -15.3919125, -18.8547440, -15.4136238, -2.8491693, 2.8382521
8: -1.6544285, 1.3965702, -1.6276751, 1.3615103, -3.0159388, 3.0242453
9: -8.9248171, -6.3680735, -8.8711119, -6.3979683, -2.3366845, 2.3364551

Time for backsubstitution: 12.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6221
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6221

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5620812, upper bound: 1.5384864
time: 3.98 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5620812, upper bound: 1.5461482
time: 4.08 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -5.2933302, -2.7619328, -5.2933302, -2.7619328, -2.2895489, 2.2895489
1: -10.1672659, -7.2483907, -10.1672659, -7.2483907, -2.7521000, 2.7521005
2: -5.5987654, -2.6825676, -5.5987654, -2.6825676, -2.9161978, 2.9161978
3: -12.1994400, -8.9904881, -12.1994400, -8.9904881, -3.0031896, 3.0031891
4: -8.8240805, -5.4225054, -8.8240805, -5.4225054, -3.2080126, 3.2080126
5: -0.9681324, 1.5843486, -0.9681324, 1.5843486, -2.5524809, 2.5524809
6: 5.0850186, 7.5114970, 5.0850186, 7.5114970, -2.3829041, 2.3829041
7: -18.8848648, -15.3919125, -18.8848648, -15.3919125, -2.8621621, 2.8621616
8: -1.6544285, 1.3965702, -1.6544285, 1.3965702, -3.0509987, 3.0509987
9: -8.9248171, -6.3680735, -8.9248171, -6.3680735, -2.3551521, 2.3551524

Time for backsubstitution: 12.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6221
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6221

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5620822, upper bound: 1.5394151
time: 4.13 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5620822, upper bound: 1.5470676
time: 4.35 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 21.32 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 21.32
Output dim: 6, lower bound: -1.5461482, upper bound: 1.5384910
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 21.32
Output dim: 6, lower bound: -1.5461482, upper bound: 1.5461530
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 21.32
Output dim: 6, lower bound: -1.5461482, upper bound: 1.5544220
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 21.32
Output dim: 6, lower bound: -1.5461482, upper bound: 1.5620826
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 21.32
Output dim: 6, lower bound: -1.5620812, upper bound: 1.5384864
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 21.32
Output dim: 6, lower bound: -1.5620812, upper bound: 1.5461482
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 21.32
Output dim: 6, lower bound: -1.5620822, upper bound: 1.5394151
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 21.32
Output dim: 6, lower bound: -1.5620822, upper bound: 1.5470676

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -5.2418346, -2.7888536, -5.2482443, -2.7802007, -2.2282658, 2.2262301
1: -10.1253729, -7.2708321, -10.1429062, -7.2638254, -2.6755581, 2.6875076
2: -5.5440569, -2.7253385, -5.5605159, -2.7213488, -2.8227081, 2.8351774
3: -12.1657486, -9.0344219, -12.1751766, -9.0126343, -2.9441900, 2.9312649
4: -8.7394257, -5.4556971, -8.7701921, -5.4502583, -3.1081944, 3.1344881
5: -0.9226367, 1.5465655, -0.9283434, 1.5568354, -2.4794722, 2.4749088
6: 5.1388283, 7.4433804, 5.1297588, 7.4612336, -2.3048658, 2.2948656
7: -18.8108521, -15.4186230, -18.8547440, -15.4136238, -2.7705617, 2.8104887
8: -1.6212955, 1.3519731, -1.6276751, 1.3615103, -2.9828057, 2.9796481
9: -8.8518600, -6.4022117, -8.8711119, -6.3979683, -2.2849193, 2.3018227

Time for backsubstitution: 12.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 76

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6221

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5384835, upper bound: 1.5384836
time: 4.04 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5384835, upper bound: 1.5384840
time: 4.65 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -5.2747049, -2.7635653, -5.2482395, -2.7802069, -2.2811170, 2.2524574
1: -10.1634569, -7.2160163, -10.1428976, -7.2638297, -2.7145395, 2.7545424
2: -5.5874424, -2.6927962, -5.5605068, -2.7213521, -2.8660903, 2.8677106
3: -12.2139702, -9.0044994, -12.1751699, -9.0126505, -2.9966874, 2.9631100
4: -8.7939568, -5.4005060, -8.7701769, -5.4502621, -3.1646209, 3.1927795
5: -0.9479733, 1.5675302, -0.9283406, 1.5568259, -2.5047991, 2.4958706
6: 5.0864220, 7.4670000, 5.1297674, 7.4612246, -2.3592539, 2.3195405
7: -18.8736877, -15.3140488, -18.8547211, -15.4136248, -2.8358440, 2.8673477
8: -1.6480672, 1.3725476, -1.6276693, 1.3615031, -3.0095704, 3.0002170
9: -8.8942366, -6.3536835, -8.8711033, -6.3979726, -2.3275881, 2.3400042

Time for backsubstitution: 12.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6221

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5384835, upper bound: 1.5461528
time: 4.02 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5384835, upper bound: 1.5461529
time: 4.60 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -5.2418346, -2.7888536, -5.2933302, -2.7619328, -2.2470679, 2.2702446
1: -10.1253729, -7.2708321, -10.1672659, -7.2483907, -2.6914935, 2.7105765
2: -5.5440569, -2.7253385, -5.5987654, -2.6825676, -2.8614893, 2.8734269
3: -12.1657486, -9.0344219, -12.1994400, -8.9904881, -2.9674902, 2.9560156
4: -8.7394257, -5.4556971, -8.8240805, -5.4225054, -3.1373653, 3.1836004
5: -0.9226367, 1.5465655, -0.9681324, 1.5843486, -2.5069852, 2.5146980
6: 5.1388283, 7.4433804, 5.0850186, 7.5114970, -2.3274503, 2.3410902
7: -18.8108521, -15.4186230, -18.8848648, -15.3919125, -2.7923713, 2.8432155
8: -1.6212955, 1.3519731, -1.6544285, 1.3965702, -3.0178657, 3.0064015
9: -8.8518600, -6.4022117, -8.9248171, -6.3680735, -2.3157330, 2.3330531

Time for backsubstitution: 12.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6221

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5384788, upper bound: 1.5544140
time: 4.37 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5384788, upper bound: 1.5544140
time: 4.37 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -5.2747049, -2.7635653, -5.2933249, -2.7619393, -2.2999196, 2.2964716
1: -10.1634569, -7.2160163, -10.1672535, -7.2483959, -2.7304778, 2.7776122
2: -5.5874424, -2.6927962, -5.5987558, -2.6825714, -2.9048710, 2.9059596
3: -12.2139702, -9.0044994, -12.1994333, -8.9905052, -3.0199904, 2.9878621
4: -8.7939568, -5.4005060, -8.8240633, -5.4225092, -3.1937919, 3.2152233
5: -0.9479733, 1.5675302, -0.9681282, 1.5843400, -2.5323133, 2.5356584
6: 5.0864220, 7.4670000, 5.0850234, 7.5114889, -2.3674536, 2.3657649
7: -18.8736877, -15.3140488, -18.8848419, -15.3919172, -2.8576531, 2.9000888
8: -1.6480672, 1.3725476, -1.6544223, 1.3965640, -3.0446312, 3.0269699
9: -8.8942366, -6.3536835, -8.9248095, -6.3680792, -2.3584008, 2.3549428

Time for backsubstitution: 12.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 76

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6221

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5384788, upper bound: 1.5620819
time: 4.18 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5384788, upper bound: 1.5620807
time: 8.44 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -5.2868509, -2.7705951, -5.2482443, -2.7802007, -2.2722087, 2.2449648
1: -10.1497879, -7.2555809, -10.1429062, -7.2638254, -2.6986866, 2.7032757
2: -5.5822864, -2.6866245, -5.5605159, -2.7213488, -2.8609376, 2.8738914
3: -12.1900940, -9.0125284, -12.1751766, -9.0126343, -2.9689331, 2.9542890
4: -8.7934961, -5.4281244, -8.7701921, -5.4502583, -3.1572046, 3.1634731
5: -0.9622895, 1.5740933, -0.9283434, 1.5568354, -2.5191250, 2.5024366
6: 5.0948477, 7.4937735, 5.1297588, 7.4612336, -2.3503861, 2.3169775
7: -18.8406715, -15.3970165, -18.8547440, -15.4136238, -2.8029609, 2.8321877
8: -1.6476550, 1.3869667, -1.6276751, 1.3615103, -3.0091653, 3.0146418
9: -8.9056864, -6.3724403, -8.8711119, -6.3979683, -2.3158877, 2.3325326

Time for backsubstitution: 12.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6221

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5544135, upper bound: 1.5384789
time: 4.53 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5544135, upper bound: 1.5384784
time: 5.85 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -5.3194294, -2.7453735, -5.2482395, -2.7802069, -2.3147202, 2.2712069
1: -10.1881475, -7.2005653, -10.1428976, -7.2638297, -2.7378893, 2.7705078
2: -5.6257677, -2.6540318, -5.5605068, -2.7213521, -2.9044156, 2.9064751
3: -12.2383289, -8.9823074, -12.1751699, -9.0126505, -3.0213704, 2.9864593
4: -8.8480415, -5.3725166, -8.7701769, -5.4502621, -3.2137303, 3.2220764
5: -0.9878474, 1.5951185, -0.9283406, 1.5568259, -2.5446734, 2.5234590
6: 5.0415444, 7.5174222, 5.1297674, 7.4612246, -2.4061036, 2.3418822
7: -18.9039097, -15.2923832, -18.8547211, -15.4136248, -2.8686352, 2.8790209
8: -1.6748238, 1.4077749, -1.6276693, 1.3615031, -3.0363269, 3.0354443
9: -8.9481421, -6.3236170, -8.8711033, -6.3979726, -2.3586292, 2.3709848

Time for backsubstitution: 13.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6221

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5544135, upper bound: 1.5461483
time: 4.23 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5544135, upper bound: 1.5461474
time: 6.18 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -5.2868509, -2.7705951, -5.2933302, -2.7619328, -2.2785158, 2.2765381
1: -10.1497879, -7.2555809, -10.1672659, -7.2483907, -2.7312822, 2.7430029
2: -5.5822864, -2.6866245, -5.5987654, -2.6825676, -2.8997188, 2.9121408
3: -12.1900940, -9.0125284, -12.1994400, -8.9904881, -2.9938807, 2.9807901
4: -8.7934961, -5.4281244, -8.8240805, -5.4225054, -3.1762676, 3.2025557
5: -0.9622895, 1.5740933, -0.9681324, 1.5843486, -2.5466380, 2.5422258
6: 5.0948477, 7.4937735, 5.0850186, 7.5114970, -2.3732867, 2.3635859
7: -18.8406715, -15.3970165, -18.8848648, -15.3919125, -2.8159537, 2.8561096
8: -1.6476550, 1.3869667, -1.6544285, 1.3965702, -3.0442252, 3.0413952
9: -8.9056864, -6.3724403, -8.9248171, -6.3680735, -2.3343315, 2.3512306

Time for backsubstitution: 12.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6221

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5544133, upper bound: 1.5394076
time: 4.30 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5544133, upper bound: 1.5394073
time: 12.19 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -5.3194294, -2.7453735, -5.2933249, -2.7619393, -2.3316097, 2.3028610
1: -10.1881475, -7.2005653, -10.1672535, -7.2483959, -2.7704868, 2.8047562
2: -5.6257677, -2.6540318, -5.5987558, -2.6825714, -2.9431963, 2.9447241
3: -12.2383289, -8.9823074, -12.1994333, -8.9905052, -3.0466928, 3.0129614
4: -8.8480415, -5.3725166, -8.8240633, -5.4225092, -3.2329726, 3.2535095
5: -0.9878474, 1.5951185, -0.9681282, 1.5843400, -2.5721874, 2.5632467
6: 5.0415444, 7.5174222, 5.0850234, 7.5114889, -2.4143038, 2.3884909
7: -18.9039097, -15.2923832, -18.8848419, -15.3919172, -2.8816276, 2.9128184
8: -1.6748238, 1.4077749, -1.6544223, 1.3965640, -3.0713878, 3.0621972
9: -8.9481421, -6.3236170, -8.9248095, -6.3680792, -2.3771343, 2.3888299

Time for backsubstitution: 12.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6221

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5544133, upper bound: 1.5470675
time: 4.21 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5544133, upper bound: 1.5470666
time: 12.30 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 29.56 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 29.56
Output dim: 6, lower bound: -1.5384835, upper bound: 1.5384836
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.56
Output dim: 6, lower bound: -1.5384835, upper bound: 1.5384840
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 29.56
Output dim: 6, lower bound: -1.5384835, upper bound: 1.5461528
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 29.56
Output dim: 6, lower bound: -1.5384835, upper bound: 1.5461529
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 29.56
Output dim: 6, lower bound: -1.5384788, upper bound: 1.5544140
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.56
Output dim: 6, lower bound: -1.5384788, upper bound: 1.5544140
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 29.56
Output dim: 6, lower bound: -1.5384788, upper bound: 1.5620819
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 29.56
Output dim: 6, lower bound: -1.5384788, upper bound: 1.5620807
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 29.56
Output dim: 6, lower bound: -1.5544135, upper bound: 1.5384789
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.56
Output dim: 6, lower bound: -1.5544135, upper bound: 1.5384784
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 29.56
Output dim: 6, lower bound: -1.5544135, upper bound: 1.5461483
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 29.56
Output dim: 6, lower bound: -1.5544135, upper bound: 1.5461474
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 29.56
Output dim: 6, lower bound: -1.5544133, upper bound: 1.5394076
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.56
Output dim: 6, lower bound: -1.5544133, upper bound: 1.5394073
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 29.56
Output dim: 6, lower bound: -1.5544133, upper bound: 1.5470675
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 29.56
Output dim: 6, lower bound: -1.5544133, upper bound: 1.5470666

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -5.2418346, -2.7888536, -5.2418346, -2.7888536, -2.2152810, 2.2152810
1: -10.1253729, -7.2708321, -10.1253729, -7.2708321, -2.6666298, 2.6666293
2: -5.5440569, -2.7253385, -5.5440569, -2.7253385, -2.8187184, 2.8187184
3: -12.1657486, -9.0344219, -12.1657486, -9.0344219, -2.9220681, 2.9220681
4: -8.7394257, -5.4556971, -8.7394257, -5.4556971, -3.1029034, 3.1029038
5: -0.9226367, 1.5465655, -0.9226367, 1.5465655, -2.4692023, 2.4692023
6: 5.1388283, 7.4433804, 5.1388283, 7.4433804, -2.2859879, 2.2859883
7: -18.8108521, -15.4186230, -18.8108521, -15.4186230, -2.7646079, 2.7646089
8: -1.6212955, 1.3519731, -1.6212955, 1.3519731, -2.9732685, 2.9732685
9: -8.8518600, -6.4022117, -8.8518600, -6.4022117, -2.2811007, 2.2811005

Time for backsubstitution: 12.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4558

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5384511, upper bound: 1.5349150
time: 4.11 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5384700, upper bound: 1.5384766
time: 4.19 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -5.2418346, -2.7888536, -5.2747049, -2.7635653, -2.2415161, 2.2530816
1: -10.1253729, -7.2708321, -10.1634569, -7.2160163, -2.7278686, 2.7056208
2: -5.5440569, -2.7253385, -5.5874424, -2.6927962, -2.8512607, 2.8621039
3: -12.1657486, -9.0344219, -12.2139702, -9.0044994, -2.9539232, 2.9745789
4: -8.7394257, -5.4556971, -8.7939568, -5.4005060, -3.1611989, 3.1593070
5: -0.9226367, 1.5465655, -0.9479733, 1.5675302, -2.4901669, 2.4945388
6: 5.1388283, 7.4433804, 5.0864220, 7.4670000, -2.3106699, 2.3397603
7: -18.8108521, -15.4186230, -18.8736877, -15.3140488, -2.8214502, 2.8298492
8: -1.6212955, 1.3519731, -1.6480672, 1.3725476, -2.9938431, 3.0000403
9: -8.8518600, -6.4022117, -8.8942366, -6.3536835, -2.3192432, 2.3237455

Time for backsubstitution: 12.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4558

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5384511, upper bound: 1.5349152
time: 4.27 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5384700, upper bound: 1.5384763
time: 5.02 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -5.2747049, -2.7635653, -5.2418346, -2.7888536, -2.2530813, 2.2415159
1: -10.1634569, -7.2160163, -10.1253729, -7.2708321, -2.7056208, 2.7278681
2: -5.5874424, -2.6927962, -5.5440569, -2.7253385, -2.8621039, 2.8512607
3: -12.2139702, -9.0044994, -12.1657486, -9.0344219, -2.9745793, 2.9539227
4: -8.7939568, -5.4005060, -8.7394257, -5.4556971, -3.1593075, 3.1611991
5: -0.9479733, 1.5675302, -0.9226367, 1.5465655, -2.4945388, 2.4901669
6: 5.0864220, 7.4670000, 5.1388283, 7.4433804, -2.3397603, 2.3106704
7: -18.8736877, -15.3140488, -18.8108521, -15.4186230, -2.8298488, 2.8214507
8: -1.6480672, 1.3725476, -1.6212955, 1.3519731, -3.0000403, 2.9938431
9: -8.8942366, -6.3536835, -8.8518600, -6.4022117, -2.3237457, 2.3192432

Time for backsubstitution: 12.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4558

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5384511, upper bound: 1.5425709
time: 4.33 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5384700, upper bound: 1.5461387
time: 4.56 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -5.2747049, -2.7635653, -5.2747049, -2.7635653, -2.2987370, 2.2987370
1: -10.1634569, -7.2160163, -10.1634569, -7.2160163, -2.7743540, 2.7743540
2: -5.5874424, -2.6927962, -5.5874424, -2.6927962, -2.8946462, 2.8946462
3: -12.2139702, -9.0044994, -12.2139702, -9.0044994, -3.0063858, 3.0063853
4: -8.7939568, -5.4005060, -8.7939568, -5.4005060, -3.2088671, 3.2088671
5: -0.9479733, 1.5675302, -0.9479733, 1.5675302, -2.5155034, 2.5155034
6: 5.0864220, 7.4670000, 5.0864220, 7.4670000, -2.3650446, 2.3650448
7: -18.8736877, -15.3140488, -18.8736877, -15.3140488, -2.8869095, 2.8869095
8: -1.6480672, 1.3725476, -1.6480672, 1.3725476, -3.0206149, 3.0206149
9: -8.8942366, -6.3536835, -8.8942366, -6.3536835, -2.3618562, 2.3618560

Time for backsubstitution: 12.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4558

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5384511, upper bound: 1.5425715
time: 4.33 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5384700, upper bound: 1.5461383
time: 4.84 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -5.2418346, -2.7888536, -5.2868509, -2.7705951, -2.2340159, 2.2592244
1: -10.1253729, -7.2708321, -10.1497879, -7.2555809, -2.6823969, 2.6897583
2: -5.5440569, -2.7253385, -5.5822864, -2.6866245, -2.8574324, 2.8569479
3: -12.1657486, -9.0344219, -12.1900940, -9.0125284, -2.9450912, 2.9468117
4: -8.7394257, -5.4556971, -8.7934961, -5.4281244, -3.1318893, 3.1519489
5: -0.9226367, 1.5465655, -0.9622895, 1.5740933, -2.4967301, 2.5088549
6: 5.1388283, 7.4433804, 5.0948477, 7.4937735, -2.3081322, 2.3315084
7: -18.8108521, -15.4186230, -18.8406715, -15.3970165, -2.7863064, 2.7970080
8: -1.6212955, 1.3519731, -1.6476550, 1.3869667, -3.0082622, 2.9996281
9: -8.8518600, -6.4022117, -8.9056864, -6.3724403, -2.3118105, 2.3122566

Time for backsubstitution: 12.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4558

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5384464, upper bound: 1.5508444
time: 4.10 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5384654, upper bound: 1.5544067
time: 4.24 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -5.2418346, -2.7888536, -5.3194294, -2.7453735, -2.2602654, 2.2966373
1: -10.1253729, -7.2708321, -10.1881475, -7.2005653, -2.7438331, 2.7289691
2: -5.5440569, -2.7253385, -5.6257677, -2.6540318, -2.8900251, 2.9004292
3: -12.1657486, -9.0344219, -12.2383289, -8.9823074, -2.9772711, 2.9992614
4: -8.7394257, -5.4556971, -8.8480415, -5.3725166, -3.1904960, 3.2084551
5: -0.9226367, 1.5465655, -0.9878474, 1.5951185, -2.5177553, 2.5344129
6: 5.1388283, 7.4433804, 5.0415444, 7.5174222, -2.3330436, 2.3866100
7: -18.8108521, -15.4186230, -18.9039097, -15.2923832, -2.8331242, 2.8626399
8: -1.6212955, 1.3519731, -1.6748238, 1.4077749, -3.0290704, 3.0267968
9: -8.8518600, -6.4022117, -8.9481421, -6.3236170, -2.3502235, 2.3549755

Time for backsubstitution: 12.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4558

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5384464, upper bound: 1.5508446
time: 4.45 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5384654, upper bound: 1.5544073
time: 6.43 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -5.2747049, -2.7635653, -5.2868509, -2.7705951, -2.2718163, 2.2854590
1: -10.1634569, -7.2160163, -10.1497879, -7.2555809, -2.7213888, 2.7509971
2: -5.5874424, -2.6927962, -5.5822864, -2.6866245, -2.9008179, 2.8894901
3: -12.2139702, -9.0044994, -12.1900940, -9.0125284, -2.9976015, 2.9786663
4: -8.7939568, -5.4005060, -8.7934961, -5.4281244, -3.1882925, 3.1835804
5: -0.9479733, 1.5675302, -0.9622895, 1.5740933, -2.5220666, 2.5298195
6: 5.0864220, 7.4670000, 5.0948477, 7.4937735, -2.3478756, 2.3561904
7: -18.8736877, -15.3140488, -18.8406715, -15.3970165, -2.8515472, 2.8538525
8: -1.6480672, 1.3725476, -1.6476550, 1.3869667, -3.0350339, 3.0202026
9: -8.8942366, -6.3536835, -8.9056864, -6.3724403, -2.3544555, 2.3341520

Time for backsubstitution: 12.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4558

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5384464, upper bound: 1.5584984
time: 4.33 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5384654, upper bound: 1.5620673
time: 4.31 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -5.2747049, -2.7635653, -5.3194294, -2.7453735, -2.3174863, 2.3322141
1: -10.1634569, -7.2160163, -10.1881475, -7.2005653, -2.7903204, 2.7977009
2: -5.5874424, -2.6927962, -5.6257677, -2.6540318, -2.9334106, 2.9329715
3: -12.2139702, -9.0044994, -12.2383289, -8.9823074, -3.0297346, 3.0310669
4: -8.7939568, -5.4005060, -8.8480415, -5.3725166, -3.2383270, 3.2400866
5: -0.9479733, 1.5675302, -0.9878474, 1.5951185, -2.5430918, 2.5553775
6: 5.0864220, 7.4670000, 5.0415444, 7.5174222, -2.3733149, 2.4118946
7: -18.8736877, -15.3140488, -18.9039097, -15.2923832, -2.8985825, 2.9197578
8: -1.6480672, 1.3725476, -1.6748238, 1.4077749, -3.0558422, 3.0473714
9: -8.8942366, -6.3536835, -8.9481421, -6.3236170, -2.3928366, 2.3768709

Time for backsubstitution: 12.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4558

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5384464, upper bound: 1.5584990
time: 4.63 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5384654, upper bound: 1.5620680
time: 7.45 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -5.2868509, -2.7705951, -5.2418346, -2.7888536, -2.2592244, 2.2340157
1: -10.1497879, -7.2555809, -10.1253729, -7.2708321, -2.6897583, 2.6823974
2: -5.5822864, -2.6866245, -5.5440569, -2.7253385, -2.8569479, 2.8574324
3: -12.1900940, -9.0125284, -12.1657486, -9.0344219, -2.9468122, 2.9450912
4: -8.7934961, -5.4281244, -8.7394257, -5.4556971, -3.1519489, 3.1318893
5: -0.9622895, 1.5740933, -0.9226367, 1.5465655, -2.5088549, 2.4967301
6: 5.0948477, 7.4937735, 5.1388283, 7.4433804, -2.3315082, 2.3081322
7: -18.8406715, -15.3970165, -18.8108521, -15.4186230, -2.7970080, 2.7863069
8: -1.6476550, 1.3869667, -1.6212955, 1.3519731, -2.9996281, 3.0082622
9: -8.9056864, -6.3724403, -8.8518600, -6.4022117, -2.3122561, 2.3118107

Time for backsubstitution: 12.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4558

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5543788, upper bound: 1.5349103
time: 4.30 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5544000, upper bound: 1.5384720
time: 4.27 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -5.2868509, -2.7705951, -5.2747049, -2.7635653, -2.2854595, 2.2718163
1: -10.1497879, -7.2555809, -10.1634569, -7.2160163, -2.7509971, 2.7213888
2: -5.5822864, -2.6866245, -5.5874424, -2.6927962, -2.8894901, 2.9008179
3: -12.1900940, -9.0125284, -12.2139702, -9.0044994, -2.9786663, 2.9976020
4: -8.7934961, -5.4281244, -8.7939568, -5.4005060, -3.1835802, 3.1882930
5: -0.9622895, 1.5740933, -0.9479733, 1.5675302, -2.5298195, 2.5220666
6: 5.0948477, 7.4937735, 5.0864220, 7.4670000, -2.3561902, 2.3478756
7: -18.8406715, -15.3970165, -18.8736877, -15.3140488, -2.8538527, 2.8515463
8: -1.6476550, 1.3869667, -1.6480672, 1.3725476, -3.0202026, 3.0350339
9: -8.9056864, -6.3724403, -8.8942366, -6.3536835, -2.3341520, 2.3544557

Time for backsubstitution: 12.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4558

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5543788, upper bound: 1.5349105
time: 4.29 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5544000, upper bound: 1.5384715
time: 5.09 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -5.3194294, -2.7453735, -5.2418346, -2.7888536, -2.2966371, 2.2602651
1: -10.1881475, -7.2005653, -10.1253729, -7.2708321, -2.7289686, 2.7438335
2: -5.6257677, -2.6540318, -5.5440569, -2.7253385, -2.9004292, 2.8900251
3: -12.2383289, -8.9823074, -12.1657486, -9.0344219, -2.9992614, 2.9772711
4: -8.8480415, -5.3725166, -8.7394257, -5.4556971, -3.2084551, 3.1904960
5: -0.9878474, 1.5951185, -0.9226367, 1.5465655, -2.5344129, 2.5177553
6: 5.0415444, 7.5174222, 5.1388283, 7.4433804, -2.3866100, 2.3330436
7: -18.9039097, -15.2923832, -18.8108521, -15.4186230, -2.8626390, 2.8331239
8: -1.6748238, 1.4077749, -1.6212955, 1.3519731, -3.0267968, 3.0290704
9: -8.9481421, -6.3236170, -8.8518600, -6.4022117, -2.3549755, 2.3502235

Time for backsubstitution: 12.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4558

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5543788, upper bound: 1.5425663
time: 4.39 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5544000, upper bound: 1.5461341
time: 4.23 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -5.3194294, -2.7453735, -5.2747049, -2.7635653, -2.3322139, 2.3174868
1: -10.1881475, -7.2005653, -10.1634569, -7.2160163, -2.7977009, 2.7903194
2: -5.6257677, -2.6540318, -5.5874424, -2.6927962, -2.9329715, 2.9334106
3: -12.2383289, -8.9823074, -12.2139702, -9.0044994, -3.0310669, 3.0297346
4: -8.8480415, -5.3725166, -8.7939568, -5.4005060, -3.2400863, 3.2383270
5: -0.9878474, 1.5951185, -0.9479733, 1.5675302, -2.5553775, 2.5430918
6: 5.0415444, 7.5174222, 5.0864220, 7.4670000, -2.4118943, 2.3733149
7: -18.9039097, -15.2923832, -18.8736877, -15.3140488, -2.9197578, 2.8985827
8: -1.6748238, 1.4077749, -1.6480672, 1.3725476, -3.0473714, 3.0558422
9: -8.9481421, -6.3236170, -8.8942366, -6.3536835, -2.3768709, 2.3928366

Time for backsubstitution: 12.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4558

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5543788, upper bound: 1.5425677
time: 4.47 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5544000, upper bound: 1.5461365
time: 6.13 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -5.2868509, -2.7705951, -5.2868509, -2.7705951, -2.2655044, 2.2655051
1: -10.1497879, -7.2555809, -10.1497879, -7.2555809, -2.7221842, 2.7221842
2: -5.5822864, -2.6866245, -5.5822864, -2.6866245, -2.8956618, 2.8956618
3: -12.1900940, -9.0125284, -12.1900940, -9.0125284, -2.9714823, 2.9714823
4: -8.7934961, -5.4281244, -8.7934961, -5.4281244, -3.1708107, 3.1708112
5: -0.9622895, 1.5740933, -0.9622895, 1.5740933, -2.5363827, 2.5363827
6: 5.0948477, 7.4937735, 5.0948477, 7.4937735, -2.3539691, 2.3539686
7: -18.8406715, -15.3970165, -18.8406715, -15.3970165, -2.8099031, 2.8099027
8: -1.6476550, 1.3869667, -1.6476550, 1.3869667, -3.0346217, 3.0346217
9: -8.9056864, -6.3724403, -8.9056864, -6.3724403, -2.3304095, 2.3304100

Time for backsubstitution: 12.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4558

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5543786, upper bound: 1.5358450
time: 4.21 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5543998, upper bound: 1.5394008
time: 4.18 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -5.2868509, -2.7705951, -5.3194294, -2.7453735, -2.2918358, 2.3035479
1: -10.1497879, -7.2555809, -10.1881475, -7.2005653, -2.7821908, 2.7613959
2: -5.5822864, -2.6866245, -5.6257677, -2.6540318, -2.9282546, 2.9391432
3: -12.1900940, -9.0125284, -12.2383289, -8.9823074, -3.0036612, 3.0243058
4: -8.7934961, -5.4281244, -8.8480415, -5.3725166, -3.2217650, 3.2274923
5: -0.9622895, 1.5740933, -0.9878474, 1.5951185, -2.5574079, 2.5619407
6: 5.0948477, 7.4937735, 5.0415444, 7.5174222, -2.3788800, 2.3947253
7: -18.8406715, -15.3970165, -18.9039097, -15.2923832, -2.8665814, 2.8755350
8: -1.6476550, 1.3869667, -1.6748238, 1.4077749, -3.0554299, 3.0617905
9: -8.9056864, -6.3724403, -8.9481421, -6.3236170, -2.3679690, 2.3731892

Time for backsubstitution: 12.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4558

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5543786, upper bound: 1.5358448
time: 4.64 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5543998, upper bound: 1.5394002
time: 7.03 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -5.3194294, -2.7453735, -5.2868509, -2.7705951, -2.3035479, 2.2918355
1: -10.1881475, -7.2005653, -10.1497879, -7.2555809, -2.7613955, 2.7821908
2: -5.6257677, -2.6540318, -5.5822864, -2.6866245, -2.9391432, 2.9282546
3: -12.2383289, -8.9823074, -12.1900940, -9.0125284, -3.0243063, 3.0036612
4: -8.8480415, -5.3725166, -8.7934961, -5.4281244, -3.2274923, 3.2217650
5: -0.9878474, 1.5951185, -0.9622895, 1.5740933, -2.5619407, 2.5574079
6: 5.0415444, 7.5174222, 5.0948477, 7.4937735, -2.3947253, 2.3788798
7: -18.9039097, -15.2923832, -18.8406715, -15.3970165, -2.8755350, 2.8665819
8: -1.6748238, 1.4077749, -1.6476550, 1.3869667, -3.0617905, 3.0554299
9: -8.9481421, -6.3236170, -8.9056864, -6.3724403, -2.3731894, 2.3679686

Time for backsubstitution: 12.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4558

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5543786, upper bound: 1.5434959
time: 4.33 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5543998, upper bound: 1.5470533
time: 4.38 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -5.3194294, -2.7453735, -5.3194294, -2.7453735, -2.3492999, 2.3492999
1: -10.1881475, -7.2005653, -10.1881475, -7.2005653, -2.8249402, 2.8249404
2: -5.6257677, -2.6540318, -5.6257677, -2.6540318, -2.9717360, 2.9717360
3: -12.2383289, -8.9823074, -12.2383289, -8.9823074, -3.0564375, 3.0564375
4: -8.8480415, -5.3725166, -8.8480415, -5.3725166, -3.2772856, 3.2772861
5: -0.9878474, 1.5951185, -0.9878474, 1.5951185, -2.5829659, 2.5829659
6: 5.0415444, 7.5174222, 5.0415444, 7.5174222, -2.4201646, 2.4201646
7: -18.9039097, -15.2923832, -18.9039097, -15.2923832, -2.9324875, 2.9324875
8: -1.6748238, 1.4077749, -1.6748238, 1.4077749, -3.0825987, 3.0825987
9: -8.9481421, -6.3236170, -8.9481421, -6.3236170, -2.4107251, 2.4107251

Time for backsubstitution: 12.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4558

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5543783, upper bound: 1.5434962
time: 4.70 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5543998, upper bound: 1.5470531
time: 6.93 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 24.53 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.53
Output dim: 6, lower bound: -1.5384511, upper bound: 1.5349150
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.53
Output dim: 6, lower bound: -1.5384700, upper bound: 1.5384766
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.53
Output dim: 6, lower bound: -1.5384511, upper bound: 1.5349152
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.53
Output dim: 6, lower bound: -1.5384700, upper bound: 1.5384763
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.53
Output dim: 6, lower bound: -1.5384511, upper bound: 1.5425709
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.53
Output dim: 6, lower bound: -1.5384700, upper bound: 1.5461387
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.53
Output dim: 6, lower bound: -1.5384511, upper bound: 1.5425715
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.53
Output dim: 6, lower bound: -1.5384700, upper bound: 1.5461383
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.53
Output dim: 6, lower bound: -1.5384464, upper bound: 1.5508444
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.53
Output dim: 6, lower bound: -1.5384654, upper bound: 1.5544067
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.53
Output dim: 6, lower bound: -1.5384464, upper bound: 1.5508446
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.53
Output dim: 6, lower bound: -1.5384654, upper bound: 1.5544073
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.53
Output dim: 6, lower bound: -1.5384464, upper bound: 1.5584984
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.53
Output dim: 6, lower bound: -1.5384654, upper bound: 1.5620673
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.53
Output dim: 6, lower bound: -1.5384464, upper bound: 1.5584990
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.53
Output dim: 6, lower bound: -1.5384654, upper bound: 1.5620680
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.53
Output dim: 6, lower bound: -1.5543788, upper bound: 1.5349103
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.53
Output dim: 6, lower bound: -1.5544000, upper bound: 1.5384720
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.53
Output dim: 6, lower bound: -1.5543788, upper bound: 1.5349105
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.53
Output dim: 6, lower bound: -1.5544000, upper bound: 1.5384715
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.53
Output dim: 6, lower bound: -1.5543788, upper bound: 1.5425663
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.53
Output dim: 6, lower bound: -1.5544000, upper bound: 1.5461341
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.53
Output dim: 6, lower bound: -1.5543788, upper bound: 1.5425677
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.53
Output dim: 6, lower bound: -1.5544000, upper bound: 1.5461365
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.53
Output dim: 6, lower bound: -1.5543786, upper bound: 1.5358450
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.53
Output dim: 6, lower bound: -1.5543998, upper bound: 1.5394008
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.53
Output dim: 6, lower bound: -1.5543786, upper bound: 1.5358448
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.53
Output dim: 6, lower bound: -1.5543998, upper bound: 1.5394002
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.53
Output dim: 6, lower bound: -1.5543786, upper bound: 1.5434959
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.53
Output dim: 6, lower bound: -1.5543998, upper bound: 1.5470533
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.53
Output dim: 6, lower bound: -1.5543783, upper bound: 1.5434962
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.53
Output dim: 6, lower bound: -1.5543998, upper bound: 1.5470531

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -5.2346039, -2.8000364, -5.2407598, -2.7908676, -2.2054152, 2.2029359
1: -10.1198444, -7.2782874, -10.1244917, -7.2724447, -2.6574421, 2.6554637
2: -5.5315886, -2.7399569, -5.5411825, -2.7262869, -2.8053017, 2.8012257
3: -12.1553612, -9.0423527, -12.1637211, -9.0357323, -2.9099188, 2.9098063
4: -8.7170601, -5.4710073, -8.7337608, -5.4566703, -3.0777159, 3.0823874
5: -0.9121457, 1.5365093, -0.9211200, 1.5440814, -2.4562273, 2.4576292
6: 5.1548109, 7.4289107, 5.1402245, 7.4392343, -2.2639990, 2.2696080
7: -18.7968864, -15.4372711, -18.8072548, -15.4198227, -2.7487884, 2.7426844
8: -1.6115003, 1.3447394, -1.6200962, 1.3503351, -2.9618354, 2.9648356
9: -8.8446941, -6.4180069, -8.8504019, -6.4031687, -2.2717505, 2.2629390

Time for backsubstitution: 12.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4558

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5349147, upper bound: 1.5349147
time: 4.98 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5349147, upper bound: 1.5349144
time: 4.69 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -5.2418308, -2.7888572, -5.2418342, -2.7888551, -2.2152762, 2.2134666
1: -10.1253710, -7.2708378, -10.1253748, -7.2708316, -2.6663561, 2.6665807
2: -5.5440469, -2.7253413, -5.5440559, -2.7253380, -2.8187089, 2.8187146
3: -12.1657419, -9.0344257, -12.1657486, -9.0344219, -2.9216909, 2.9252896
4: -8.7394075, -5.4556999, -8.7394257, -5.4556990, -3.0924950, 3.1022167
5: -0.9226316, 1.5465596, -0.9226375, 1.5465662, -2.4691978, 2.4691970
6: 5.1388321, 7.4433641, 5.1388288, 7.4433804, -2.2859826, 2.2847686
7: -18.8108368, -15.4186287, -18.8108521, -15.4186230, -2.7598877, 2.7646041
8: -1.6212916, 1.3519673, -1.6212959, 1.3519721, -2.9732637, 2.9732633
9: -8.8518543, -6.4022141, -8.8518581, -6.4022131, -2.2810869, 2.2810948

Time for backsubstitution: 12.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4558

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5349147, upper bound: 1.5384580
time: 4.92 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5349145, upper bound: 1.5384578
time: 4.88 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -5.2346039, -2.8000364, -5.2736387, -2.7655737, -2.2316475, 2.2407441
1: -10.1198444, -7.2782874, -10.1625690, -7.2176256, -2.7186794, 2.6944547
2: -5.5315886, -2.7399569, -5.5845785, -2.6937423, -2.8378463, 2.8446217
3: -12.1553612, -9.0423527, -12.2118855, -9.0058327, -2.9417534, 2.9623113
4: -8.7170601, -5.4710073, -8.7882757, -5.4014788, -3.1360250, 3.1387687
5: -0.9121457, 1.5365093, -0.9464555, 1.5650451, -2.4771910, 2.4829648
6: 5.1548109, 7.4289107, 5.0878105, 7.4628439, -2.2886624, 2.3230896
7: -18.7968864, -15.4372711, -18.8700809, -15.3152409, -2.8054361, 2.8079472
8: -1.6115003, 1.3447394, -1.6468687, 1.3709102, -2.9824104, 2.9916081
9: -8.8446941, -6.4180069, -8.8927593, -6.3546448, -2.3099332, 2.3055584

Time for backsubstitution: 12.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4558

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5425704, upper bound: 1.5349147
time: 4.70 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.5425704, upper bound: 1.5349150
time: 4.47 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 22.03 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 22.03
Output dim: 6, lower bound: -1.5349147, upper bound: 1.5349147
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 22.03
Output dim: 6, lower bound: -1.5349147, upper bound: 1.5349144
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 22.03
Output dim: 6, lower bound: -1.5349147, upper bound: 1.5384580
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 22.03
Output dim: 6, lower bound: -1.5349145, upper bound: 1.5384578
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 22.03
Output dim: 6, lower bound: -1.5425704, upper bound: 1.5349147
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 22.03
Output dim: 6, lower bound: -1.5425704, upper bound: 1.5349150
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.03
Output dim: 6, lower bound: -1.5384700, upper bound: 1.5384763
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.03
Output dim: 6, lower bound: -1.5384511, upper bound: 1.5425709
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.03
Output dim: 6, lower bound: -1.5384700, upper bound: 1.5461387
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.03
Output dim: 6, lower bound: -1.5384511, upper bound: 1.5425715
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.03
Output dim: 6, lower bound: -1.5384700, upper bound: 1.5461383
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.03
Output dim: 6, lower bound: -1.5384464, upper bound: 1.5508444
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.03
Output dim: 6, lower bound: -1.5384654, upper bound: 1.5544067
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.03
Output dim: 6, lower bound: -1.5384464, upper bound: 1.5508446
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.03
Output dim: 6, lower bound: -1.5384654, upper bound: 1.5544073
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.03
Output dim: 6, lower bound: -1.5384464, upper bound: 1.5584984
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.03
Output dim: 6, lower bound: -1.5384654, upper bound: 1.5620673
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.03
Output dim: 6, lower bound: -1.5384464, upper bound: 1.5584990
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.03
Output dim: 6, lower bound: -1.5384654, upper bound: 1.5620680
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.03
Output dim: 6, lower bound: -1.5543788, upper bound: 1.5349103
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.03
Output dim: 6, lower bound: -1.5544000, upper bound: 1.5384720
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.03
Output dim: 6, lower bound: -1.5543788, upper bound: 1.5349105
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.03
Output dim: 6, lower bound: -1.5544000, upper bound: 1.5384715
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.03
Output dim: 6, lower bound: -1.5543788, upper bound: 1.5425663
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.03
Output dim: 6, lower bound: -1.5544000, upper bound: 1.5461341
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.03
Output dim: 6, lower bound: -1.5543788, upper bound: 1.5425677
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.03
Output dim: 6, lower bound: -1.5544000, upper bound: 1.5461365
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.03
Output dim: 6, lower bound: -1.5543786, upper bound: 1.5358450
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.03
Output dim: 6, lower bound: -1.5543998, upper bound: 1.5394008
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.03
Output dim: 6, lower bound: -1.5543786, upper bound: 1.5358448
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.03
Output dim: 6, lower bound: -1.5543998, upper bound: 1.5394002
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.03
Output dim: 6, lower bound: -1.5543786, upper bound: 1.5434959
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.03
Output dim: 6, lower bound: -1.5543998, upper bound: 1.5470533
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.03
Output dim: 6, lower bound: -1.5543783, upper bound: 1.5434962
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.03
Output dim: 6, lower bound: -1.5543998, upper bound: 1.5470531
Binary search (step 1): status=Status.UNKNOWN, k_low=6, k_high=8, k_mid=7, eps_mid=0.0273438, abs_max=2.3460235595703125
rel_dist={6: [-1.5621134080175434, 1.562113956458468]}

## Binary search (step 2) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 6221
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 481

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4800157, upper bound: 1.4943217
time: 4.33 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4944183, upper bound: 1.4944194
time: 4.07 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 8.56 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 8.56
Output dim: 6, lower bound: -1.4800157, upper bound: 1.4943217
IS_A2, status: Status.UNKNOWN, split count: 1, time: 8.56
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

Time for backsubstitution: 12.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 481

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4800157, upper bound: 1.4800156
time: 4.26 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4800157, upper bound: 1.4943216
time: 4.27 seconds

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

Time for backsubstitution: 12.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 481

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4943216, upper bound: 1.4800157
time: 5.06 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4943216, upper bound: 1.4800157
time: 4.54 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 22.60 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 22.60
Output dim: 6, lower bound: -1.4800157, upper bound: 1.4800156
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 22.60
Output dim: 6, lower bound: -1.4800157, upper bound: 1.4943216
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 22.60
Output dim: 6, lower bound: -1.4943216, upper bound: 1.4800157
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 22.60
Output dim: 6, lower bound: -1.4943216, upper bound: 1.4800157

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -5.2482443, -2.7802007, -5.2933302, -2.7619328, -2.1960688, 2.2212811
1: -10.1429062, -7.2638254, -10.1672659, -7.2483907, -2.6375003, 2.6446333
2: -5.5605159, -2.7213488, -5.5987654, -2.6825676, -2.8779483, 2.8774166
3: -12.1751766, -9.0126343, -12.1994400, -8.9904881, -2.9060040, 2.9074531
4: -8.7701921, -5.4502583, -8.8240805, -5.4225054, -3.0857363, 3.1040339
5: -0.9283434, 1.5568354, -0.9681324, 1.5843486, -2.5126920, 2.5249677
6: 5.1297588, 7.4612336, 5.0850186, 7.5114970, -2.2994142, 2.3235903
7: -18.8547440, -15.4136238, -18.8848648, -15.3919125, -2.7431350, 2.7540522
8: -1.6276751, 1.3615103, -1.6544285, 1.3965702, -3.0242453, 3.0159388
9: -8.8711119, -6.3979683, -8.9248171, -6.3680735, -2.2731557, 2.2728732

Time for backsubstitution: 12.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6221
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6221

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4797304, upper bound: 1.4872872
time: 4.58 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4800074, upper bound: 1.4943128
time: 4.71 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -5.2933302, -2.7619328, -5.2482443, -2.7802007, -2.2212811, 2.1960688
1: -10.1672659, -7.2483907, -10.1429062, -7.2638254, -2.6446323, 2.6375003
2: -5.5987654, -2.6825676, -5.5605159, -2.7213488, -2.8774166, 2.8779483
3: -12.1994400, -8.9904881, -12.1751766, -9.0126343, -2.9074526, 2.9060040
4: -8.8240805, -5.4225054, -8.7701921, -5.4502583, -3.1040344, 3.0857363
5: -0.9681324, 1.5843486, -0.9283434, 1.5568354, -2.5249677, 2.5126920
6: 5.0850186, 7.5114970, 5.1297588, 7.4612336, -2.3235903, 2.2994142
7: -18.8848648, -15.3919125, -18.8547440, -15.4136238, -2.7540526, 2.7431355
8: -1.6544285, 1.3965702, -1.6276751, 1.3615103, -3.0159388, 3.0242453
9: -8.9248171, -6.3680735, -8.8711119, -6.3979683, -2.2728732, 2.2731557

Time for backsubstitution: 12.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6221
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6221

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4940353, upper bound: 1.4729759
time: 4.65 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4943122, upper bound: 1.4800070
time: 4.49 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -5.2933302, -2.7619328, -5.2933302, -2.7619328, -2.2258863, 2.2258863
1: -10.1672659, -7.2483907, -10.1672659, -7.2483907, -2.6766911, 2.6766906
2: -5.5987654, -2.6825676, -5.5987654, -2.6825676, -2.9161978, 2.9161978
3: -12.1994400, -8.9904881, -12.1994400, -8.9904881, -2.9321399, 2.9321399
4: -8.8240805, -5.4225054, -8.8240805, -5.4225054, -3.1221371, 3.1221366
5: -0.9681324, 1.5843486, -0.9681324, 1.5843486, -2.5524809, 2.5524809
6: 5.0850186, 7.5114970, 5.0850186, 7.5114970, -2.3449454, 2.3449454
7: -18.8848648, -15.3919125, -18.8848648, -15.3919125, -2.7663064, 2.7663069
8: -1.6544285, 1.3965702, -1.6544285, 1.3965702, -3.0509987, 3.0509987
9: -8.9248171, -6.3680735, -8.9248171, -6.3680735, -2.2892427, 2.2892425

Time for backsubstitution: 12.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6221
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6221

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4940363, upper bound: 1.4739038
time: 4.46 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4943133, upper bound: 1.4809505
time: 4.71 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 22.13 seconds
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 22.13
Output dim: 6, lower bound: -1.4797304, upper bound: 1.4872872
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 22.13
Output dim: 6, lower bound: -1.4800074, upper bound: 1.4943128
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 22.13
Output dim: 6, lower bound: -1.4940353, upper bound: 1.4729759
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 22.13
Output dim: 6, lower bound: -1.4943122, upper bound: 1.4800070
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 22.13
Output dim: 6, lower bound: -1.4940363, upper bound: 1.4739038
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 22.13
Output dim: 6, lower bound: -1.4943133, upper bound: 1.4809505

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -5.2747049, -2.7635653, -5.2933249, -2.7619407, -2.2373438, 2.2345219
1: -10.1634569, -7.2160163, -10.1672506, -7.2483978, -2.6556034, 2.7024984
2: -5.5874424, -2.6927962, -5.5987549, -2.6825719, -2.9048705, 2.9059587
3: -12.2139702, -9.0044994, -12.1994305, -8.9905081, -2.9493036, 2.9171729
4: -8.7939568, -5.4005060, -8.8240614, -5.4225101, -3.1100020, 3.1303978
5: -0.9479733, 1.5675302, -0.9681273, 1.5843400, -2.5323133, 2.5356574
6: 5.0864220, 7.4670000, 5.0850258, 7.5114875, -2.3305335, 2.3293862
7: -18.8736877, -15.3140488, -18.8848381, -15.3919182, -2.7612262, 2.8022876
8: -1.6480672, 1.3725476, -1.6544218, 1.3965635, -3.0446308, 3.0269694
9: -8.8942366, -6.3536835, -8.9248075, -6.3680801, -2.2944441, 2.2911310

Time for backsubstitution: 12.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6221

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4729701, upper bound: 1.4940356
time: 4.29 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4729701, upper bound: 1.4943115
time: 6.43 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -5.2868509, -2.7705951, -5.2475896, -2.7810910, -2.2089229, 2.1819015
1: -10.1497879, -7.2555809, -10.1411104, -7.2645540, -2.6228819, 2.6262660
2: -5.5822864, -2.6866245, -5.5588298, -2.7217584, -2.8605280, 2.8722053
3: -12.1900940, -9.0125284, -12.1741962, -9.0148754, -2.8959723, 2.8826728
4: -8.7934961, -5.4281244, -8.7670345, -5.4508424, -3.0718338, 3.0770192
5: -0.9622895, 1.5740933, -0.9277334, 1.5557830, -2.5180726, 2.5018268
6: 5.0948477, 7.4937735, 5.1307578, 7.4593897, -2.3120689, 2.2791066
7: -18.8406715, -15.3970165, -18.8502445, -15.4141397, -2.7072315, 2.7323580
8: -1.6476550, 1.3869667, -1.6269846, 1.3605270, -3.0081820, 3.0139513
9: -8.9056864, -6.3724403, -8.8691349, -6.3984089, -2.2516990, 2.2671049

Time for backsubstitution: 12.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6221

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4872802, upper bound: 1.4729705
time: 4.53 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4872802, upper bound: 1.4729726
time: 6.35 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -5.3194294, -2.7453735, -5.2482376, -2.7802081, -2.2509346, 2.2092562
1: -10.1881475, -7.2005653, -10.1428947, -7.2638302, -2.6630168, 2.6953950
2: -5.6257677, -2.6540318, -5.5605049, -2.7213521, -2.9044156, 2.9064732
3: -12.2383289, -8.9823074, -12.1751690, -9.0126524, -2.9506845, 2.9157710
4: -8.8480415, -5.3725166, -8.7701740, -5.4502635, -3.1283286, 3.1357088
5: -0.9878474, 1.5951185, -0.9283385, 1.5568247, -2.5446720, 2.5234571
6: 5.0415444, 7.5174222, 5.1297665, 7.4612226, -2.3675513, 2.3049998
7: -18.9039097, -15.2923832, -18.8547173, -15.4136276, -2.7722073, 2.7821751
8: -1.6748238, 1.4077749, -1.6276698, 1.3615022, -3.0363259, 3.0354447
9: -8.9481421, -6.3236170, -8.8711004, -6.3979740, -2.2941594, 2.3052812

Time for backsubstitution: 12.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6221

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4872802, upper bound: 1.4797309
time: 5.23 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4872802, upper bound: 1.4800072
time: 12.44 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -5.2868509, -2.7705951, -5.2926712, -2.7628245, -2.2135177, 2.2117541
1: -10.1497879, -7.2555809, -10.1654730, -7.2491174, -2.6549454, 2.6654606
2: -5.5822864, -2.6866245, -5.5970764, -2.6829801, -2.8993063, 2.9104519
3: -12.1900940, -9.0125284, -12.1984673, -8.9927549, -2.9205275, 2.9087996
4: -8.7934961, -5.4281244, -8.8209381, -5.4230919, -3.0898395, 3.1134224
5: -0.9622895, 1.5740933, -0.9675264, 1.5832969, -2.5455863, 2.5416198
6: 5.0948477, 7.4937735, 5.0860167, 7.5096712, -2.3329048, 2.3246374
7: -18.8406715, -15.3970165, -18.8803368, -15.3924284, -2.7194853, 2.7555132
8: -1.6476550, 1.3869667, -1.6537299, 1.3955851, -3.0432401, 3.0406966
9: -8.9056864, -6.3724403, -8.9228516, -6.3685179, -2.2680240, 2.2831833

Time for backsubstitution: 12.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6221

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4873762, upper bound: 1.4738982
time: 4.50 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4873771, upper bound: 1.4738980
time: 6.34 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -5.3194294, -2.7453735, -5.2933249, -2.7619407, -2.2673192, 2.2391965
1: -10.1881475, -7.2005653, -10.1672506, -7.2483978, -2.6950746, 2.7280085
2: -5.6257677, -2.6540318, -5.5987549, -2.6825719, -2.9431958, 2.9447231
3: -12.2383289, -8.9823074, -12.1994305, -8.9905081, -2.9756422, 2.9419074
4: -8.8480415, -5.3725166, -8.8240614, -5.4225101, -3.1465197, 3.1668620
5: -0.9878474, 1.5951185, -0.9681273, 1.5843400, -2.5721874, 2.5632458
6: 5.0415444, 7.5174222, 5.0850258, 7.5114875, -2.3763061, 2.3505309
7: -18.9039097, -15.2923832, -18.8848381, -15.3919182, -2.7844620, 2.8154664
8: -1.6748238, 1.4077749, -1.6544218, 1.3965635, -3.0713873, 3.0621967
9: -8.9481421, -6.3236170, -8.9248075, -6.3680801, -2.3105664, 2.3232293

Time for backsubstitution: 12.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6221
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6221

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4873771, upper bound: 1.4806633
time: 4.43 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4873771, upper bound: 1.4806657
time: 6.35 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 23.79 seconds
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.79
Output dim: 6, lower bound: -1.4729701, upper bound: 1.4940356
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.79
Output dim: 6, lower bound: -1.4729701, upper bound: 1.4943115
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 23.79
Output dim: 6, lower bound: -1.4872802, upper bound: 1.4729705
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 23.79
Output dim: 6, lower bound: -1.4872802, upper bound: 1.4729726
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 23.79
Output dim: 6, lower bound: -1.4872802, upper bound: 1.4797309
IS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 23.79
Output dim: 6, lower bound: -1.4872802, upper bound: 1.4800072
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 23.79
Output dim: 6, lower bound: -1.4873762, upper bound: 1.4738982
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 23.79
Output dim: 6, lower bound: -1.4873771, upper bound: 1.4738980
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 23.79
Output dim: 6, lower bound: -1.4873771, upper bound: 1.4806633
IS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 23.79
Output dim: 6, lower bound: -1.4873771, upper bound: 1.4806657

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -5.2747049, -2.7635653, -5.2868509, -2.7705951, -2.2098680, 2.2235110
1: -10.1634569, -7.2160163, -10.1497879, -7.2555809, -2.6465178, 2.6761255
2: -5.5874424, -2.6927962, -5.5822864, -2.6866245, -2.9008179, 2.8894901
3: -12.2139702, -9.0044994, -12.1900940, -9.0125284, -2.9269190, 2.9079823
4: -8.7939568, -5.4005060, -8.7934961, -5.4281244, -3.1050797, 3.0987577
5: -0.9479733, 1.5675302, -0.9622895, 1.5740933, -2.5220666, 2.5298195
6: 5.0864220, 7.4670000, 5.0948477, 7.4937735, -2.3109941, 2.3198125
7: -18.8736877, -15.3140488, -18.8406715, -15.3970165, -2.7564306, 2.7560701
8: -1.6480672, 1.3725476, -1.6476550, 1.3869667, -3.0350339, 3.0202026
9: -8.8942366, -6.3536835, -8.9056864, -6.3724403, -2.2911563, 2.2703412

Time for backsubstitution: 12.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4558

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4729582, upper bound: 1.4910692
time: 4.33 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4729581, upper bound: 1.4940232
time: 4.59 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -5.2747049, -2.7635653, -5.3194294, -2.7453735, -2.2549119, 2.2684550
1: -10.1634569, -7.2160163, -10.1881475, -7.2005653, -2.7152090, 2.7225895
2: -5.5874424, -2.6927962, -5.6257677, -2.6540318, -2.9334106, 2.9329715
3: -12.2139702, -9.0044994, -12.2383289, -8.9823074, -2.9590483, 2.9603810
4: -8.7939568, -5.4005060, -8.8480415, -5.3725166, -3.1545372, 3.1547089
5: -0.9479733, 1.5675302, -0.9878474, 1.5951185, -2.5430918, 2.5553775
6: 5.0864220, 7.4670000, 5.0415444, 7.5174222, -2.3363972, 2.3733447
7: -18.8736877, -15.3140488, -18.9039097, -15.2923832, -2.8004780, 2.8206940
8: -1.6480672, 1.3725476, -1.6748238, 1.4077749, -3.0558422, 3.0473714
9: -8.8942366, -6.3536835, -8.9481421, -6.3236170, -2.3265014, 2.3124278

Time for backsubstitution: 12.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4558

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4729581, upper bound: 1.4913458
time: 5.11 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4729581, upper bound: 1.4943005
time: 5.77 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 23.81 seconds
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.81
Output dim: 6, lower bound: -1.4729582, upper bound: 1.4910692
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.81
Output dim: 6, lower bound: -1.4729581, upper bound: 1.4940232
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.81
Output dim: 6, lower bound: -1.4729581, upper bound: 1.4913458
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.81
Output dim: 6, lower bound: -1.4729581, upper bound: 1.4943005

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -5.2674403, -2.7747190, -5.2856016, -2.7729805, -2.1995912, 2.2109208
1: -10.1578770, -7.2237649, -10.1487551, -7.2574825, -2.6369209, 2.6644258
2: -5.5750418, -2.7075229, -5.5788722, -2.6877503, -2.8872914, 2.8713493
3: -12.2034664, -9.0125160, -12.1877108, -9.0140982, -2.9144344, 2.8952956
4: -8.7715359, -5.4160876, -8.7867737, -5.4292846, -3.0795536, 3.0738916
5: -0.9372170, 1.5574658, -0.9604937, 1.5711503, -2.5083673, 2.5179596
6: 5.1034641, 7.4524994, 5.0964937, 7.4888735, -2.2839665, 2.3031058
7: -18.8597164, -15.3328199, -18.8363819, -15.3984394, -2.7403288, 2.7313075
8: -1.6378021, 1.3653221, -1.6462259, 1.3850164, -3.0228186, 3.0115480
9: -8.8869886, -6.3696585, -8.9039736, -6.3735762, -2.2814946, 2.2517848

Time for backsubstitution: 12.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4558

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4699885, upper bound: 1.4910691
time: 4.32 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4699885, upper bound: 1.4910685
time: 6.52 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -5.2747006, -2.7635689, -5.2868514, -2.7705948, -2.2098637, 2.2215903
1: -10.1634531, -7.2160225, -10.1497898, -7.2555823, -2.6462574, 2.6760678
2: -5.5874324, -2.6927991, -5.5822845, -2.6866250, -2.9008074, 2.8894854
3: -12.2139635, -9.0045013, -12.1900911, -9.0125275, -2.9264393, 2.9110632
4: -8.7939386, -5.4005084, -8.7934942, -5.4281249, -3.0938215, 3.0925999
5: -0.9479688, 1.5675244, -0.9622884, 1.5740926, -2.5220613, 2.5298128
6: 5.0864258, 7.4669843, 5.0948477, 7.4937706, -2.3072190, 2.3185229
7: -18.8736687, -15.3140526, -18.8406677, -15.3970184, -2.7514338, 2.7531013
8: -1.6480644, 1.3725410, -1.6476550, 1.3869658, -3.0350301, 3.0201960
9: -8.8942308, -6.3536859, -8.9056873, -6.3724403, -2.2910452, 2.2694337

Time for backsubstitution: 12.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4558

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4699885, upper bound: 1.4940234
time: 4.29 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4699884, upper bound: 1.4940227
time: 4.76 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -5.2674403, -2.7747190, -5.3181887, -2.7477527, -2.2446408, 2.2559314
1: -10.1578770, -7.2237649, -10.1871080, -7.2024622, -2.7056150, 2.7108898
2: -5.5750418, -2.7075229, -5.6223693, -2.6551564, -2.9198854, 2.9148464
3: -12.2034664, -9.0125160, -12.2358742, -8.9839020, -2.9465389, 2.9476905
4: -8.7715359, -5.4160876, -8.8413000, -5.3736749, -3.1290116, 3.1298218
5: -0.9372170, 1.5574658, -0.9860523, 1.5921739, -2.5293908, 2.5435181
6: 5.1034641, 7.4524994, 5.0431833, 7.5125113, -2.3093457, 2.3563447
7: -18.8597164, -15.3328199, -18.8996143, -15.2938023, -2.7841210, 2.7959332
8: -1.6378021, 1.3653221, -1.6733971, 1.4058266, -3.0436287, 3.0387192
9: -8.8869886, -6.3696585, -8.9464054, -6.3247590, -2.3168845, 2.2938316

Time for backsubstitution: 12.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4558

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4721285, upper bound: 1.4913459
time: 5.68 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4721285, upper bound: 1.4913453
time: 4.47 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -5.2747006, -2.7635689, -5.3194304, -2.7453744, -2.2549071, 2.2664566
1: -10.1634531, -7.2160225, -10.1881475, -7.2005644, -2.7149487, 2.7224951
2: -5.5874324, -2.6927991, -5.6257682, -2.6540313, -2.9334011, 2.9329691
3: -12.2139635, -9.0045013, -12.2383299, -8.9823093, -2.9585686, 2.9634700
4: -8.7939386, -5.4005084, -8.8480387, -5.3725176, -3.1432657, 3.1485510
5: -0.9479688, 1.5675244, -0.9878471, 1.5951186, -2.5430875, 2.5553715
6: 5.0864258, 7.4669843, 5.0415440, 7.5174198, -2.3326197, 2.3715789
7: -18.8736687, -15.3140526, -18.9039059, -15.2923832, -2.7953291, 2.8177269
8: -1.6480644, 1.3725410, -1.6748233, 1.4077730, -3.0558374, 3.0473642
9: -8.8942308, -6.3536859, -8.9481411, -6.3236184, -2.3255529, 2.3115191

Time for backsubstitution: 12.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4558

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4721285, upper bound: 1.4943018
time: 4.66 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4721283, upper bound: 1.4943002
time: 4.64 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 22.20 seconds
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 22.20
Output dim: 6, lower bound: -1.4699885, upper bound: 1.4910691
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 22.20
Output dim: 6, lower bound: -1.4699885, upper bound: 1.4910685
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 22.20
Output dim: 6, lower bound: -1.4699885, upper bound: 1.4940234
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 22.20
Output dim: 6, lower bound: -1.4699884, upper bound: 1.4940227
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 22.20
Output dim: 6, lower bound: -1.4721285, upper bound: 1.4913459
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 22.20
Output dim: 6, lower bound: -1.4721285, upper bound: 1.4913453
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 22.20
Output dim: 6, lower bound: -1.4721285, upper bound: 1.4943018
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 22.20
Output dim: 6, lower bound: -1.4721283, upper bound: 1.4943002

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -5.2674403, -2.7747190, -5.2796493, -2.7817819, -2.1910596, 2.2047207
1: -10.1578770, -7.2237649, -10.1442833, -7.2631788, -2.6292238, 2.6586704
2: -5.5750418, -2.7075229, -5.5698347, -2.7013993, -2.8736424, 2.8623118
3: -12.2034664, -9.0125160, -12.1797543, -9.0205116, -2.9056435, 2.8867950
4: -8.7715359, -5.4160876, -8.7711105, -5.4435892, -3.0664306, 3.0601902
5: -0.9372170, 1.5574658, -0.9516779, 1.5640391, -2.5012560, 2.5091438
6: 5.1034641, 7.4524994, 5.1113925, 7.4793386, -2.2766490, 2.2863150
7: -18.8597164, -15.3328199, -18.8266258, -15.4157400, -2.7238402, 2.7230825
8: -1.6378021, 1.3653221, -1.6375489, 1.3797069, -3.0175090, 3.0028710
9: -8.8869886, -6.3696585, -8.8985777, -6.3883209, -2.2657866, 2.2450745

Time for backsubstitution: 12.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6196

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4699907, upper bound: 1.4908399
time: 4.65 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4699907, upper bound: 1.4910689
time: 4.31 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -5.2674403, -2.7747190, -5.2868476, -2.7705989, -2.2021184, 2.2124119
1: -10.1578770, -7.2237649, -10.1497879, -7.2555866, -2.6395640, 2.6653247
2: -5.5750418, -2.7075229, -5.5822773, -2.6866274, -2.8884144, 2.8747544
3: -12.2034664, -9.0125160, -12.1900845, -9.0125303, -2.9149833, 2.8971839
4: -8.7715359, -5.4160876, -8.7934771, -5.4281278, -3.0804796, 3.0757871
5: -0.9372170, 1.5574658, -0.9622841, 1.5740867, -2.5113037, 2.5197499
6: 5.1034641, 7.4524994, 5.0948505, 7.4937572, -2.2852259, 2.3047738
7: -18.8597164, -15.3328199, -18.8406544, -15.3970213, -2.7421575, 2.7334175
8: -1.6378021, 1.3653221, -1.6476510, 1.3869619, -3.0247641, 3.0129731
9: -8.8869886, -6.3696585, -8.9056816, -6.3724427, -2.2826176, 2.2526255

Time for backsubstitution: 12.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6196

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4699907, upper bound: 1.4908396
time: 4.30 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4699907, upper bound: 1.4910686
time: 4.42 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -5.2747006, -2.7635689, -5.2796493, -2.7817819, -2.1988001, 2.2158089
1: -10.1634531, -7.2160225, -10.1442833, -7.2631788, -2.6359081, 2.6691799
2: -5.5874324, -2.6927991, -5.5698347, -2.7013993, -2.8860331, 2.8770356
3: -12.2139635, -9.0045013, -12.1797543, -9.0205116, -2.9161530, 2.8961706
4: -8.7939386, -5.4005084, -8.7711105, -5.4435892, -3.0903296, 3.0686789
5: -0.9479688, 1.5675244, -0.9516779, 1.5640391, -2.5120080, 2.5192022
6: 5.0864258, 7.4669843, 5.1113925, 7.4793386, -2.2919269, 2.3013368
7: -18.8736687, -15.3140526, -18.8266258, -15.4157400, -2.7381015, 2.7385044
8: -1.6480644, 1.3725410, -1.6375489, 1.3797069, -3.0277712, 3.0100899
9: -8.8942308, -6.3536859, -8.8985777, -6.3883209, -2.2736235, 2.2611871

Time for backsubstitution: 12.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6196

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4699883, upper bound: 1.4938022
time: 4.93 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4699883, upper bound: 1.4940239
time: 4.96 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -5.2747006, -2.7635689, -5.2868476, -2.7705989, -2.2079425, 2.2215860
1: -10.1634531, -7.2160225, -10.1497879, -7.2555866, -2.6463881, 2.6760645
2: -5.5874324, -2.6927991, -5.5822773, -2.6866274, -2.9008050, 2.8894782
3: -12.2139635, -9.0045013, -12.1900845, -9.0125303, -2.9299903, 2.9110560
4: -8.7939386, -5.4005084, -8.7934771, -5.4281278, -3.0938182, 3.0865779
5: -0.9479688, 1.5675244, -0.9622841, 1.5740867, -2.5220556, 2.5298085
6: 5.0864258, 7.4669843, 5.0948505, 7.4937572, -2.3064351, 2.3185189
7: -18.8736687, -15.3140526, -18.8406544, -15.3970213, -2.7514310, 2.7500606
8: -1.6480644, 1.3725410, -1.6476510, 1.3869619, -3.0350263, 3.0201919
9: -8.8942308, -6.3536859, -8.9056816, -6.3724427, -2.2910428, 2.2693326

Time for backsubstitution: 13.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6196

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4699883, upper bound: 1.4938012
time: 4.64 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4699881, upper bound: 1.4940229
time: 4.48 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -5.2674403, -2.7747190, -5.3122220, -2.7565305, -2.2360840, 2.2496848
1: -10.1578770, -7.2237649, -10.1825933, -7.2082911, -2.6977644, 2.7051368
2: -5.5750418, -2.7075229, -5.6133852, -2.6688716, -2.9061701, 2.9058623
3: -12.2034664, -9.0125160, -12.2278118, -8.9903717, -2.9376836, 2.9391403
4: -8.7715359, -5.4160876, -8.8256035, -5.3881207, -3.1156125, 3.1160517
5: -0.9372170, 1.5574658, -0.9771043, 1.5850576, -2.5222745, 2.5345702
6: 5.1034641, 7.4524994, 5.0585871, 7.5029554, -2.3019886, 2.3389964
7: -18.8597164, -15.3328199, -18.8898659, -15.3111687, -2.7675543, 2.7876945
8: -1.6378021, 1.3653221, -1.6644824, 1.4005313, -3.0383334, 3.0298045
9: -8.8869886, -6.3696585, -8.9409513, -6.3396049, -2.3011212, 2.2870159

Time for backsubstitution: 13.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6196

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4721307, upper bound: 1.4911421
time: 5.68 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4721307, upper bound: 1.4913458
time: 4.65 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -5.2674403, -2.7747190, -5.3194256, -2.7453787, -2.2471614, 2.2562137
1: -10.1578770, -7.2237649, -10.1881456, -7.2005692, -2.7082562, 2.7118034
2: -5.5750418, -2.7075229, -5.6257601, -2.6540337, -2.9210081, 2.9182372
3: -12.2034664, -9.0125160, -12.2383242, -8.9823103, -2.9471121, 2.9495907
4: -8.7715359, -5.4160876, -8.8480206, -5.3725204, -3.1297741, 3.1317415
5: -0.9372170, 1.5574658, -0.9878424, 1.5951128, -2.5323298, 2.5453081
6: 5.1034641, 7.4524994, 5.0415483, 7.5174060, -2.3106256, 2.3542931
7: -18.8597164, -15.3328199, -18.9038906, -15.2923889, -2.7829847, 2.7980652
8: -1.6378021, 1.3653221, -1.6748197, 1.4077692, -3.0455713, 3.0401418
9: -8.8869886, -6.3696585, -8.9481373, -6.3236208, -2.3172331, 2.2947009

Time for backsubstitution: 13.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6196

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4721307, upper bound: 1.4911406
time: 5.30 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4721307, upper bound: 1.4913456
time: 4.66 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -5.2747006, -2.7635689, -5.3122220, -2.7565305, -2.2438245, 2.2579818
1: -10.1634531, -7.2160225, -10.1825933, -7.2082911, -2.7044506, 2.7156463
2: -5.5874324, -2.6927991, -5.6133852, -2.6688716, -2.9185607, 2.9205861
3: -12.2139635, -9.0045013, -12.2278118, -8.9903717, -2.9481931, 2.9485154
4: -8.7939386, -5.4005084, -8.8256035, -5.3881207, -3.1368749, 3.1245403
5: -0.9479688, 1.5675244, -0.9771043, 1.5850576, -2.5330265, 2.5446286
6: 5.0864258, 7.4669843, 5.0585871, 7.5029554, -2.3172660, 2.3476014
7: -18.8736687, -15.3140526, -18.8898659, -15.3111687, -2.7778502, 2.8031168
8: -1.6480644, 1.3725410, -1.6644824, 1.4005313, -3.0485957, 3.0370233
9: -8.8942308, -6.3536859, -8.9409513, -6.3396049, -2.3087764, 2.3031282

Time for backsubstitution: 13.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6196

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4721283, upper bound: 1.4941025
time: 5.19 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4721280, upper bound: 1.4943002
time: 4.88 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -5.2747006, -2.7635689, -5.3194256, -2.7453787, -2.2529874, 2.2660422
1: -10.1634531, -7.2160225, -10.1881456, -7.2005692, -2.7150803, 2.7224913
2: -5.5874324, -2.6927991, -5.6257601, -2.6540337, -2.9333987, 2.9329610
3: -12.2139635, -9.0045013, -12.2383242, -8.9823103, -2.9621181, 2.9634628
4: -8.7939386, -5.4005084, -8.8480206, -5.3725204, -3.1432610, 3.1425357
5: -0.9479688, 1.5675244, -0.9878424, 1.5951128, -2.5430818, 2.5553670
6: 5.0864258, 7.4669843, 5.0415483, 7.5174060, -2.3318381, 2.3688195
7: -18.8736687, -15.3140526, -18.9038906, -15.2923889, -2.7944698, 2.8146846
8: -1.6480644, 1.3725410, -1.6748197, 1.4077692, -3.0558336, 3.0473607
9: -8.8942308, -6.3536859, -8.9481373, -6.3236208, -2.3254819, 2.3114080

Time for backsubstitution: 13.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6196
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4584
type: A, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6196

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4721283, upper bound: 1.4941019
time: 4.97 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4721280, upper bound: 1.4943002
time: 5.41 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 23.81 seconds
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 23.81
Output dim: 6, lower bound: -1.4699907, upper bound: 1.4908399
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 23.81
Output dim: 6, lower bound: -1.4699907, upper bound: 1.4910689
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 23.81
Output dim: 6, lower bound: -1.4699907, upper bound: 1.4908396
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 23.81
Output dim: 6, lower bound: -1.4699907, upper bound: 1.4910686
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 23.81
Output dim: 6, lower bound: -1.4699883, upper bound: 1.4938022
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 23.81
Output dim: 6, lower bound: -1.4699883, upper bound: 1.4940239
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 23.81
Output dim: 6, lower bound: -1.4699883, upper bound: 1.4938012
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 23.81
Output dim: 6, lower bound: -1.4699881, upper bound: 1.4940229
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 23.81
Output dim: 6, lower bound: -1.4721307, upper bound: 1.4911421
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 23.81
Output dim: 6, lower bound: -1.4721307, upper bound: 1.4913458
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 23.81
Output dim: 6, lower bound: -1.4721307, upper bound: 1.4911406
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 23.81
Output dim: 6, lower bound: -1.4721307, upper bound: 1.4913456
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 23.81
Output dim: 6, lower bound: -1.4721283, upper bound: 1.4941025
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 23.81
Output dim: 6, lower bound: -1.4721280, upper bound: 1.4943002
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 23.81
Output dim: 6, lower bound: -1.4721283, upper bound: 1.4941019
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 23.81
Output dim: 6, lower bound: -1.4721280, upper bound: 1.4943002

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -5.2417121, -2.7959731, -5.2742743, -2.7887924, -2.1588769, 2.1730342
1: -10.1048708, -7.2530217, -10.1255455, -7.2658758, -2.5771265, 2.6101475
2: -5.5493259, -2.7483177, -5.5664506, -2.7147577, -2.8345683, 2.8181329
3: -12.1757851, -9.0272007, -12.1707325, -9.0228138, -2.8694296, 2.8232856
4: -8.7512484, -5.4395514, -8.7681227, -5.4514971, -2.9992151, 2.9978704
5: -0.9234831, 1.5381267, -0.9474792, 1.5597303, -2.4832134, 2.4856060
6: 5.1213131, 7.4425364, 5.1150503, 7.4770751, -2.2563787, 2.3063374
7: -18.8351784, -15.3512173, -18.8202000, -15.4218578, -2.6854219, 2.6928728
8: -1.6066082, 1.3411727, -1.6276555, 1.3759875, -2.9825957, 2.9688282
9: -8.8534994, -6.4134684, -8.8946218, -6.4034939, -2.2195621, 2.1969519

Time for backsubstitution: 13.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6196

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4697613, upper bound: 1.4908441
time: 4.72 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4697613, upper bound: 1.4908428
time: 5.55 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -5.2674365, -2.7747283, -5.2796483, -2.7817836, -2.1910529, 2.1924644
1: -10.1578665, -7.2237692, -10.1442804, -7.2631798, -2.5990705, 2.6499245
2: -5.5750380, -2.7075315, -5.5698347, -2.7014036, -2.8736343, 2.8623033
3: -12.2034569, -9.0125189, -12.1797523, -9.0205116, -2.8986154, 2.8867908
4: -8.7715321, -5.4161019, -8.7711086, -5.4435911, -3.0598402, 3.0575356
5: -0.9372108, 1.5574605, -0.9516770, 1.5640385, -2.5012493, 2.5091376
6: 5.1034684, 7.4524956, 5.1113935, 7.4793382, -2.2766094, 2.2841005
7: -18.8597145, -15.3328323, -18.8266258, -15.4157438, -2.7184772, 2.7224393
8: -1.6377916, 1.3653193, -1.6375473, 1.3797083, -3.0174999, 3.0028665
9: -8.8869839, -6.3696814, -8.8985777, -6.3883247, -2.2594891, 2.2069924

Time for backsubstitution: 13.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6196

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4697613, upper bound: 1.4910725
time: 4.50 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4697613, upper bound: 1.4910710
time: 5.11 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -5.2417121, -2.7959731, -5.2814937, -2.7776053, -2.1699257, 2.1806021
1: -10.1048708, -7.2530217, -10.1310349, -7.2582855, -2.5862722, 2.6168299
2: -5.5493259, -2.7483177, -5.5788994, -2.6999862, -2.8493397, 2.8305817
3: -12.1757851, -9.0272007, -12.1810522, -9.0148325, -2.8787680, 2.8336768
4: -8.7512484, -5.4395514, -8.7904930, -5.4360447, -3.0132656, 3.0134652
5: -0.9234831, 1.5381267, -0.9580728, 1.5697793, -2.4932623, 2.4961996
6: 5.1213131, 7.4425364, 5.0985298, 7.4914637, -2.2649531, 2.3247635
7: -18.8351784, -15.3512173, -18.8342419, -15.4031382, -2.7037392, 2.7030656
8: -1.6066082, 1.3411727, -1.6377246, 1.3832393, -2.9898474, 2.9788973
9: -8.8534994, -6.4134684, -8.9017267, -6.3876209, -2.2363753, 2.2045174

Time for backsubstitution: 13.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6196

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4727209, upper bound: 1.4908389
time: 7.33 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4727209, upper bound: 1.4908391
time: 7.54 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -5.2674365, -2.7747283, -5.2868462, -2.7706017, -2.2021108, 2.2001565
1: -10.1578665, -7.2237692, -10.1497850, -7.2555866, -2.6094112, 2.6567459
2: -5.5750380, -2.7075315, -5.5822754, -2.6866312, -2.8884068, 2.8747439
3: -12.2034569, -9.0125189, -12.1900835, -9.0125313, -2.9079561, 2.8971796
4: -8.7715321, -5.4161019, -8.7934761, -5.4281311, -3.0738878, 3.0731125
5: -0.9372108, 1.5574605, -0.9622830, 1.5740865, -2.5112972, 2.5197434
6: 5.1034684, 7.4524956, 5.0948524, 7.4937563, -2.2852087, 2.3025601
7: -18.8597145, -15.3328323, -18.8406544, -15.3970261, -2.7363033, 2.7328191
8: -1.6377916, 1.3653193, -1.6476502, 1.3869619, -3.0247536, 3.0129695
9: -8.8869839, -6.3696814, -8.9056826, -6.3724475, -2.2755108, 2.2145693

Time for backsubstitution: 13.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6196

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4727209, upper bound: 1.4910695
time: 7.91 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4727209, upper bound: 1.4910683
time: 7.64 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -5.2489805, -2.7848194, -5.2742743, -2.7887924, -2.1666389, 2.1841323
1: -10.1104183, -7.2455659, -10.1255455, -7.2658758, -2.5839071, 2.6185594
2: -5.5617418, -2.7336984, -5.5664506, -2.7147577, -2.8469841, 2.8327522
3: -12.1861401, -9.0191774, -12.1707325, -9.0228138, -2.8799219, 2.8326993
4: -8.7736435, -5.4242363, -8.7681227, -5.4514971, -3.0231638, 3.0060010
5: -0.9339442, 1.5481896, -0.9474792, 1.5597303, -2.4936745, 2.4956689
6: 5.1053381, 7.4570141, 5.1150503, 7.4770751, -2.2706246, 2.3213768
7: -18.8491745, -15.3325787, -18.8202000, -15.4218578, -2.6996851, 2.7081621
8: -1.6163275, 1.3484054, -1.6276555, 1.3759875, -2.9923151, 2.9760609
9: -8.8607616, -6.3976564, -8.8946218, -6.4034939, -2.2272086, 2.2129142

Time for backsubstitution: 13.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 76

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6196

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4697587, upper bound: 1.4938032
time: 4.69 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4697587, upper bound: 1.4938008
time: 5.44 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -5.2746964, -2.7635789, -5.2796483, -2.7817836, -2.1987929, 2.2035522
1: -10.1634417, -7.2160244, -10.1442804, -7.2631798, -2.6057529, 2.6587446
2: -5.5874300, -2.6928096, -5.5698347, -2.7014036, -2.8860264, 2.8770251
3: -12.2139549, -9.0045052, -12.1797523, -9.0205116, -2.9090619, 2.8961649
4: -8.7939367, -5.4005203, -8.7711086, -5.4435911, -3.0837502, 3.0660095
5: -0.9479631, 1.5675197, -0.9516770, 1.5640385, -2.5120015, 2.5191965
6: 5.0864296, 7.4669814, 5.1113935, 7.4793382, -2.2918639, 2.2991128
7: -18.8736649, -15.3140621, -18.8266258, -15.4157438, -2.7312284, 2.7378650
8: -1.6480546, 1.3725376, -1.6375473, 1.3797083, -3.0277629, 3.0100849
9: -8.8942280, -6.3537107, -8.8985777, -6.3883247, -2.2671418, 2.2230971

Time for backsubstitution: 13.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6196

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4697587, upper bound: 1.4940249
time: 4.56 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4697587, upper bound: 1.4940223
time: 5.52 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -5.2489805, -2.7848194, -5.2814937, -2.7776053, -2.1757731, 2.1897855
1: -10.1104183, -7.2455659, -10.1310349, -7.2582855, -2.5931978, 2.6268196
2: -5.5617418, -2.7336984, -5.5788994, -2.6999862, -2.8617556, 2.8452010
3: -12.1861401, -9.0191774, -12.1810522, -9.0148325, -2.8937593, 2.8475881
4: -8.7736435, -5.4242363, -8.7904930, -5.4360447, -3.0266490, 3.0238976
5: -0.9339442, 1.5481896, -0.9580728, 1.5697793, -2.5037236, 2.5062623
6: 5.1053381, 7.4570141, 5.0985298, 7.4914637, -2.2851267, 2.3385267
7: -18.8491745, -15.3325787, -18.8342419, -15.4031382, -2.7130132, 2.7195706
8: -1.6163275, 1.3484054, -1.6377246, 1.3832393, -2.9995668, 2.9861300
9: -8.8607616, -6.3976564, -8.9017267, -6.3876209, -2.2446108, 2.2210741

Time for backsubstitution: 13.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 76

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 6196

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4698181, upper bound: 1.4938004
time: 5.16 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4698181, upper bound: 1.4938012
time: 4.72 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -5.2746964, -2.7635789, -5.2868462, -2.7706017, -2.2079363, 2.2093287
1: -10.1634417, -7.2160244, -10.1497850, -7.2555866, -2.6162338, 2.6670237
2: -5.5874300, -2.6928096, -5.5822754, -2.6866312, -2.9007988, 2.8894658
3: -12.2139549, -9.0045052, -12.1900835, -9.0125313, -2.9228992, 2.9110508
4: -8.7939367, -5.4005203, -8.7934761, -5.4281311, -3.0872359, 3.0838890
5: -0.9479631, 1.5675197, -0.9622830, 1.5740865, -2.5220497, 2.5298028
6: 5.0864296, 7.4669814, 5.0948524, 7.4937563, -2.3063946, 2.3162961
7: -18.8736649, -15.3140621, -18.8406544, -15.3970261, -2.7460394, 2.7494655
8: -1.6480546, 1.3725376, -1.6476502, 1.3869619, -3.0350165, 3.0201879
9: -8.8942280, -6.3537107, -8.9056826, -6.3724475, -2.2837577, 2.2312682

Time for backsubstitution: 13.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6196
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4584
type: B, layer: 1, pos: 76

Time for candidate selection: 0.19 seconds
Binary search (step 2): status=Status.UNKNOWN, k_low=6, k_high=6, k_mid=6, eps_mid=0.0234375, abs_max=2.309645652770996
rel_dist={6: [-1.4944306407619283, 1.4944305738519086]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01953125
execution time: 2417.58 seconds
