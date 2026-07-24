## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2000 seconds
Threshold: 43.1275729563
Search space: {k/256 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-24.6925583, 19.7640648, -24.6925583, 19.7640648, -44.4566231, 44.4566231)
1: (-22.1655941, 17.7110901, -22.1655941, 17.7110901, -39.8766861, 39.8766861)
2: (-28.0165939, 17.5855999, -28.0165939, 17.5855999, -45.6021957, 45.6021957)
3: (-30.1115532, 15.0882940, -30.1115532, 15.0882940, -45.1998444, 45.1998482)
4: (-28.4748173, 20.2006111, -28.4748173, 20.2006111, -48.6754303, 48.6754303)
5: (-24.4868717, 19.1075554, -24.4868717, 19.1075554, -43.5944290, 43.5944290)
6: (-22.5470924, 22.3522205, -22.5470924, 22.3522205, -44.8993111, 44.8993149)
7: (-24.8867416, 23.5568867, -24.8867416, 23.5568867, -48.4436264, 48.4436264)
8: (-34.8141861, 16.7162991, -34.8141861, 16.7162991, -51.5304832, 51.5304871)
9: (-21.9554176, 22.3245659, -21.9554176, 22.3245659, -44.2799835, 44.2799835)

## BASE Result
execution time: IAR + LP analysis = 1.37 + 8.10 = 9.47 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -43.1710632, upper bound: 43.1710632


# Binary Search by BASE starts (time budget: 1990.53 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=51.530487060546875
rel_dist={8: [-43.17090795605019, 43.17090794479026]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=51.530487060546875
rel_dist={8: [-43.17074367834749, 43.17074366907076]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=51.530487060546875
rel_dist={8: [-43.1705810903977, 43.17058109163787]}

## Binary Search Result
Binary search time: 36.61 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 1953.92 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 50

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1705784, upper bound: 43.1708452
time: 7.03 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1709079, upper bound: 43.1709079
time: 5.57 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 12.75 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 12.75
Output dim: 8, lower bound: -43.1705784, upper bound: 43.1708452
IS_A2, status: Status.UNKNOWN, split count: 1, time: 12.75
Output dim: 8, lower bound: -43.1709079, upper bound: 43.1709079

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -24.3706722, 19.5024204, -24.2903042, 19.4450378, -43.8157120, 43.7927246
1: -21.8586521, 17.4819832, -21.8066139, 17.4282684, -39.2869110, 39.2885971
2: -27.6241531, 17.3405914, -27.5614986, 17.3017693, -44.9259148, 44.9020882
3: -29.6621819, 14.9008579, -29.6202545, 14.8481913, -44.5103722, 44.5211105
4: -28.0579205, 19.9350491, -28.0166950, 19.8713856, -47.9293022, 47.9517441
5: -24.1463985, 18.8390179, -24.0903625, 18.7999134, -42.9463120, 42.9293823
6: -22.2339973, 22.0551300, -22.1759415, 21.9912338, -44.2252274, 44.2310715
7: -24.5400143, 23.1987343, -24.4799442, 23.1846714, -47.7246857, 47.6786690
8: -34.2982140, 16.5464706, -34.2710991, 16.4321918, -50.7304077, 50.8175697
9: -21.6667118, 22.0264530, -21.5955086, 21.9609833, -43.6276894, 43.6219635

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 50

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1683164, upper bound: 43.1688686
time: 12.26 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1681780, upper bound: 43.1686063
time: 6.10 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -24.2108173, 19.3813133, -24.6925583, 19.7640648, -43.9748764, 44.0738716
1: -21.7347298, 17.3723087, -22.1655941, 17.7110901, -39.4458199, 39.5379028
2: -27.4703083, 17.2466469, -28.0165939, 17.5855999, -45.0559082, 45.2632408
3: -29.5238304, 14.8024702, -30.1115532, 15.0882940, -44.6121216, 44.9140129
4: -27.9244385, 19.8066959, -28.4748173, 20.2006111, -48.1250496, 48.2815132
5: -24.0113335, 18.7387772, -24.4868717, 19.1075554, -43.1188889, 43.2256470
6: -22.1033802, 21.9187889, -22.5470924, 22.3522205, -44.4555969, 44.4658813
7: -24.3987465, 23.1103039, -24.8867416, 23.5568867, -47.9556351, 47.9970474
8: -34.1606636, 16.3779488, -34.8141861, 16.7162991, -50.8769569, 51.1921349
9: -21.5237293, 21.8889427, -21.9554176, 22.3245659, -43.8482971, 43.8443489

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 50

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1708452, upper bound: 43.1705784
time: 6.58 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1708452, upper bound: 43.1709079
time: 7.05 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 15.16 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 15.16
Output dim: 8, lower bound: -43.1683164, upper bound: 43.1688686
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 15.16
Output dim: 8, lower bound: -43.1681780, upper bound: 43.1686063
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 15.16
Output dim: 8, lower bound: -43.1708452, upper bound: 43.1705784
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 15.16
Output dim: 8, lower bound: -43.1708452, upper bound: 43.1709079

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -24.3706722, 19.5024204, -22.9695473, 18.3834763, -42.7541504, 42.4719658
1: -21.8586521, 17.4819832, -20.6135750, 16.4789028, -38.3375549, 38.0955582
2: -27.6241531, 17.3405914, -26.0605373, 16.3493233, -43.9734764, 43.4011230
3: -29.6621819, 14.9008579, -28.0160160, 14.0311756, -43.6933594, 42.9168739
4: -28.0579205, 19.9350491, -26.5320950, 18.7660389, -46.8239594, 46.4671402
5: -24.1463985, 18.8390179, -22.8005371, 17.7927551, -41.9391556, 41.6395569
6: -22.2339973, 22.0551300, -20.9451466, 20.7912788, -43.0252724, 43.0002747
7: -24.5400143, 23.1987343, -23.1314774, 22.0069542, -46.5469666, 46.3302078
8: -34.2982140, 16.5464706, -32.4924545, 15.3958769, -49.6940880, 49.0389252
9: -21.6667118, 22.0264530, -20.4149170, 20.7356796, -42.4023895, 42.4413681

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 183

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1651997, upper bound: 43.1640688
time: 7.63 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1623733, upper bound: 43.1629115
time: 6.63 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -23.9871731, 19.1935978, -26.8603287, 21.4188137, -45.4059792, 46.0539207
1: -21.5127792, 17.2048302, -24.1079884, 19.1830425, -40.6958199, 41.3128204
2: -27.1898804, 17.0628014, -30.4894905, 18.9899139, -46.1797867, 47.5522842
3: -29.1959038, 14.6628256, -32.7554779, 16.2291565, -45.4250603, 47.4183006
4: -27.6280746, 19.6133060, -31.0500908, 21.8624020, -49.4904747, 50.6633987
5: -23.7730808, 18.5468025, -26.6828442, 20.7423325, -44.5154114, 45.2296448
6: -21.8756542, 21.7054901, -24.4494171, 24.2518234, -46.1274719, 46.1548996
7: -24.1479301, 22.8586311, -27.0632915, 25.6961937, -49.8441238, 49.9219131
8: -33.7855797, 16.2419605, -37.8342972, 17.8004055, -51.5859795, 54.0762520
9: -21.3234997, 21.6695213, -23.8503571, 24.1431217, -45.4666214, 45.5198746

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 183

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1647623, upper bound: 43.1632351
time: 8.63 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1620611, upper bound: 43.1622301
time: 6.65 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -24.2108173, 19.3813133, -24.3706722, 19.5024204, -43.7132378, 43.7519836
1: -21.7347298, 17.3723087, -21.8586521, 17.4819832, -39.2167130, 39.2309570
2: -27.4703083, 17.2466469, -27.6241531, 17.3405914, -44.8108940, 44.8708000
3: -29.5238304, 14.8024702, -29.6621819, 14.9008579, -44.4246864, 44.4646530
4: -27.9244385, 19.8066959, -28.0579205, 19.9350491, -47.8594894, 47.8646164
5: -24.0113335, 18.7387772, -24.1463985, 18.8390179, -42.8503494, 42.8851776
6: -22.1033802, 21.9187889, -22.2339973, 22.0551300, -44.1585083, 44.1527863
7: -24.3987465, 23.1103039, -24.5400143, 23.1987343, -47.5974808, 47.6503181
8: -34.1606636, 16.3779488, -34.2982140, 16.5464706, -50.7071342, 50.6761627
9: -21.5237293, 21.8889427, -21.6667118, 22.0264530, -43.5501823, 43.5556412

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 50

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1688686, upper bound: 43.1683164
time: 8.18 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1686063, upper bound: 43.1681780
time: 7.06 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -24.2108173, 19.3813133, -24.2108173, 19.3813133, -43.5921211, 43.5921211
1: -21.7347298, 17.3723087, -21.7347298, 17.3723087, -39.1070404, 39.1070404
2: -27.4703083, 17.2466469, -27.4703083, 17.2466469, -44.7169533, 44.7169533
3: -29.5238304, 14.8024702, -29.5238304, 14.8024702, -44.3262901, 44.3262901
4: -27.9244385, 19.8066959, -27.9244385, 19.8066959, -47.7311325, 47.7311325
5: -24.0113335, 18.7387772, -24.0113335, 18.7387772, -42.7501106, 42.7501106
6: -22.1033802, 21.9187889, -22.1033802, 21.9187889, -44.0221710, 44.0221710
7: -24.3987465, 23.1103039, -24.3987465, 23.1103039, -47.5090485, 47.5090485
8: -34.1606636, 16.3779488, -34.1606636, 16.3779488, -50.5386047, 50.5386047
9: -21.5237293, 21.8889427, -21.5237293, 21.8889427, -43.4126587, 43.4126663

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 50

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1688686, upper bound: 43.1689468
time: 7.29 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1686063, upper bound: 43.1688233
time: 7.18 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 15.96 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 15.96
Output dim: 8, lower bound: -43.1651997, upper bound: 43.1640688
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 15.96
Output dim: 8, lower bound: -43.1623733, upper bound: 43.1629115
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 15.96
Output dim: 8, lower bound: -43.1647623, upper bound: 43.1632351
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 15.96
Output dim: 8, lower bound: -43.1620611, upper bound: 43.1622301
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 15.96
Output dim: 8, lower bound: -43.1688686, upper bound: 43.1683164
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 15.96
Output dim: 8, lower bound: -43.1686063, upper bound: 43.1681780
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 15.96
Output dim: 8, lower bound: -43.1688686, upper bound: 43.1689468
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 15.96
Output dim: 8, lower bound: -43.1686063, upper bound: 43.1688233

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -23.3544388, 18.6943626, -22.9695473, 18.3834763, -41.7379150, 41.6639099
1: -20.9448109, 16.7650013, -20.6135750, 16.4789028, -37.4237137, 37.3785782
2: -26.4616013, 16.6310959, -26.0605373, 16.3493233, -42.8109245, 42.6916313
3: -28.4084911, 14.3010941, -28.0160160, 14.0311756, -42.4396667, 42.3171005
4: -26.8841457, 19.1039753, -26.5320950, 18.7660389, -45.6501846, 45.6360664
5: -23.1337814, 18.0627384, -22.8005371, 17.7927551, -40.9265366, 40.8632736
6: -21.2974434, 21.1338425, -20.9451466, 20.7912788, -42.0887184, 42.0789871
7: -23.5010185, 22.2488403, -23.1314774, 22.0069542, -45.5079651, 45.3803177
8: -32.9020958, 15.8553543, -32.4924545, 15.3958769, -48.2979698, 48.3478088
9: -20.7569847, 21.1029491, -20.4149170, 20.7356796, -41.4926643, 41.5178680

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 216

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1623733, upper bound: 43.1629115
time: 5.59 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1623733, upper bound: 43.1629115
time: 7.97 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -30.0759506, 24.0378208, -22.6130066, 18.0994816, -48.1754303, 46.6508179
1: -27.0501595, 21.4804993, -20.2927265, 16.2279396, -43.2780914, 41.7732239
2: -34.1593552, 21.3206100, -25.6520538, 16.0999870, -50.2593422, 46.9726639
3: -36.6262054, 18.2534943, -27.5763969, 13.8203716, -50.4465790, 45.8298798
4: -34.6238632, 24.5312843, -26.1207771, 18.4741383, -53.0979996, 50.6520615
5: -29.7996025, 23.1861877, -22.4448395, 17.5200386, -47.3196373, 45.6310272
6: -27.4289970, 27.1393719, -20.6165390, 20.4677696, -47.8967628, 47.7559090
7: -30.3329735, 28.5733967, -22.7671223, 21.6736851, -52.0066605, 51.3405190
8: -42.1040802, 20.3677444, -32.0011215, 15.1525011, -57.2565727, 52.3688583
9: -26.7435570, 27.1401291, -20.0956688, 20.4105186, -47.1540718, 47.2357979

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 216

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1573072, upper bound: 43.1580590
time: 7.21 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1571007, upper bound: 43.1577655
time: 6.62 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -22.9783134, 18.3921165, -26.8603287, 21.4188137, -44.3971252, 45.2524452
1: -20.6052856, 16.4939213, -24.1079884, 19.1830425, -39.7883263, 40.6019058
2: -26.0343285, 16.3594894, -30.4894905, 18.9899139, -45.0242348, 46.8489799
3: -27.9512844, 14.0681858, -32.7554779, 16.2291565, -44.1804428, 46.8236580
4: -26.4619102, 18.7890644, -31.0500908, 21.8624020, -48.3243103, 49.8391571
5: -22.7677155, 17.7772408, -26.6828442, 20.7423325, -43.5100479, 44.4600754
6: -20.9465790, 20.7910156, -24.4494171, 24.2518234, -45.1984024, 45.2404327
7: -23.1167812, 21.9159794, -27.0632915, 25.6961937, -48.8129730, 48.9792709
8: -32.3983574, 15.5573521, -37.8342972, 17.8004055, -50.1987457, 53.3916435
9: -20.4210205, 20.7523079, -23.8503571, 24.1431217, -44.5641403, 44.6026649

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 216

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1598294, upper bound: 43.1585989
time: 7.76 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1594622, upper bound: 43.1580250
time: 7.62 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -29.6593227, 23.7053642, -26.4980354, 21.1317196, -50.7910423, 50.2033958
1: -26.6808624, 21.1823063, -23.7834892, 18.9279995, -45.6088638, 44.9657974
2: -33.7012062, 21.0249691, -30.0763187, 18.7361202, -52.4373245, 51.1012878
3: -36.1215057, 18.0007420, -32.3109741, 16.0149632, -52.1364670, 50.3117104
4: -34.1757507, 24.1698627, -30.6341476, 21.5670052, -55.7427559, 54.8040085
5: -29.3947144, 22.8692093, -26.3231087, 20.4652023, -49.8599052, 49.1923141
6: -27.0475101, 26.7642326, -24.1154251, 23.9240303, -50.9715424, 50.8796577
7: -29.9098320, 28.2047844, -26.6929607, 25.3595619, -55.2693939, 54.8977432
8: -41.5579529, 20.0392895, -37.3396416, 17.5513191, -59.1092682, 57.3789291
9: -26.3771362, 26.7613983, -23.5255718, 23.8152637, -50.1923981, 50.2869682

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 216

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1570147, upper bound: 43.1574511
time: 7.45 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1567568, upper bound: 43.1570083
time: 6.26 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -22.9020443, 18.3290501, -24.3706722, 19.5024204, -42.4044647, 42.6997223
1: -20.5521736, 16.4315586, -21.8586521, 17.4819832, -38.0341568, 38.2902107
2: -25.9828796, 16.3024616, -27.6241531, 17.3405914, -43.3234711, 43.9266129
3: -27.9341412, 13.9924498, -29.6621819, 14.9008579, -42.8349991, 43.6546326
4: -26.4539204, 18.7113247, -28.0579205, 19.9350491, -46.3889694, 46.7692451
5: -22.7331505, 17.7413349, -24.1463985, 18.8390179, -41.5721664, 41.8877335
6: -20.8838615, 20.7296562, -22.2339973, 22.0551300, -42.9389915, 42.9636536
7: -23.0626984, 21.9438381, -24.5400143, 23.1987343, -46.2614326, 46.4838524
8: -32.3981972, 15.3495474, -34.2982140, 16.5464706, -48.9446678, 49.6477547
9: -20.3538895, 20.6743355, -21.6667118, 22.0264530, -42.3803406, 42.3410492

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 183

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1640689, upper bound: 43.1651997
time: 7.41 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1627531, upper bound: 43.1623733
time: 7.33 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -26.7876816, 21.3605366, -23.9871731, 19.1935978, -45.9812775, 45.3477058
1: -24.0423031, 19.1318951, -21.5127792, 17.2048302, -41.2471313, 40.6446648
2: -30.4065971, 18.9392223, -27.1898804, 17.0628014, -47.4693947, 46.1290970
3: -32.6673126, 16.1869755, -29.1959038, 14.6628256, -47.3301315, 45.3828735
4: -30.9663239, 21.8034859, -27.6280746, 19.6133060, -50.5796280, 49.4315567
5: -26.6104794, 20.6868591, -23.7730808, 18.5468025, -45.1572800, 44.4599380
6: -24.3831501, 24.1858101, -21.8756542, 21.7054901, -46.0886383, 46.0614624
7: -26.9893494, 25.6285477, -24.1479301, 22.8586311, -49.8479767, 49.7764664
8: -37.7339745, 17.7501411, -33.7855797, 16.2419605, -53.9759331, 51.5357208
9: -23.7847672, 24.0774078, -21.3234997, 21.6695213, -45.4542809, 45.4009094

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 183

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1632351, upper bound: 43.1647623
time: 5.89 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1622301, upper bound: 43.1620611
time: 4.76 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -22.9020443, 18.3290501, -24.2108173, 19.3813133, -42.2833557, 42.5398598
1: -20.5521736, 16.4315586, -21.7347298, 17.3723087, -37.9244843, 38.1662903
2: -25.9828796, 16.3024616, -27.4703083, 17.2466469, -43.2295265, 43.7727661
3: -27.9341412, 13.9924498, -29.5238304, 14.8024702, -42.7366066, 43.5162773
4: -26.4539204, 18.7113247, -27.9244385, 19.8066959, -46.2606163, 46.6357651
5: -22.7331505, 17.7413349, -24.0113335, 18.7387772, -41.4719276, 41.7526703
6: -20.8838615, 20.7296562, -22.1033802, 21.9187889, -42.8026505, 42.8330383
7: -23.0626984, 21.9438381, -24.3987465, 23.1103039, -46.1730042, 46.3425827
8: -32.3981972, 15.3495474, -34.1606636, 16.3779488, -48.7761421, 49.5102005
9: -20.3538895, 20.6743355, -21.5237293, 21.8889427, -42.2428284, 42.1980667

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 50

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1688581, upper bound: 43.1688201
time: 6.52 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1688581, upper bound: 43.1688233
time: 7.05 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -26.7876816, 21.3605366, -23.8067875, 19.0570869, -45.8447685, 45.1673241
1: -24.0423031, 19.1318951, -21.3710442, 17.0813904, -41.1236954, 40.5029335
2: -30.4065971, 18.9392223, -27.0119305, 16.9552231, -47.3618164, 45.9511490
3: -32.6673126, 16.1869755, -29.0318947, 14.5540485, -47.2213593, 45.2188683
4: -30.9663239, 21.8034859, -27.4696999, 19.4689598, -50.4352837, 49.2731705
5: -26.6104794, 20.6868591, -23.6180153, 18.4304352, -45.0409164, 44.3048668
6: -24.3831501, 24.1858101, -21.7265987, 21.5511818, -45.9343338, 45.9124069
7: -26.9893494, 25.6285477, -23.9851780, 22.7505627, -49.7399063, 49.6137199
8: -37.7339745, 17.7501411, -33.6183624, 16.0626640, -53.7966347, 51.3685036
9: -23.7847672, 24.0774078, -21.1625423, 21.5145950, -45.2993546, 45.2399445

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 196

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1633208, upper bound: 43.1649944
time: 5.90 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1622567, upper bound: 43.1622057
time: 6.80 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 14.25 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 14.25
Output dim: 8, lower bound: -43.1623733, upper bound: 43.1629115
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 14.25
Output dim: 8, lower bound: -43.1623733, upper bound: 43.1629115
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 14.25
Output dim: 8, lower bound: -43.1573072, upper bound: 43.1580590
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 14.25
Output dim: 8, lower bound: -43.1571007, upper bound: 43.1577655
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 14.25
Output dim: 8, lower bound: -43.1598294, upper bound: 43.1585989
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 14.25
Output dim: 8, lower bound: -43.1594622, upper bound: 43.1580250
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 14.25
Output dim: 8, lower bound: -43.1570147, upper bound: 43.1574511
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 14.25
Output dim: 8, lower bound: -43.1567568, upper bound: 43.1570083
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 14.25
Output dim: 8, lower bound: -43.1640689, upper bound: 43.1651997
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 14.25
Output dim: 8, lower bound: -43.1627531, upper bound: 43.1623733
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 14.25
Output dim: 8, lower bound: -43.1632351, upper bound: 43.1647623
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 14.25
Output dim: 8, lower bound: -43.1622301, upper bound: 43.1620611
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 14.25
Output dim: 8, lower bound: -43.1688581, upper bound: 43.1688201
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 14.25
Output dim: 8, lower bound: -43.1688581, upper bound: 43.1688233
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 14.25
Output dim: 8, lower bound: -43.1633208, upper bound: 43.1649944
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 14.25
Output dim: 8, lower bound: -43.1622567, upper bound: 43.1622057

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -23.3544388, 18.6943626, -22.0465679, 17.6483231, -41.0027618, 40.7409286
1: -20.9448109, 16.7650013, -19.7806206, 15.8277225, -36.7725258, 36.5456200
2: -26.4616013, 16.6310959, -25.0022011, 15.7044277, -42.1660309, 41.6332932
3: -28.4084911, 14.3010941, -26.8748474, 13.4851856, -41.8936768, 41.1759415
4: -26.8841457, 19.1039753, -25.4636402, 18.0102234, -44.8943710, 44.5676155
5: -23.1337814, 18.0627384, -21.8784924, 17.0861721, -40.2199554, 39.9412308
6: -21.2974434, 21.1338425, -20.0948544, 19.9537792, -41.2512169, 41.2286987
7: -23.5010185, 22.2488403, -22.1873360, 21.1411304, -44.6421471, 44.4361763
8: -32.9020958, 15.8553543, -31.2173977, 14.7717609, -47.6738586, 47.0727539
9: -20.7569847, 21.1029491, -19.5883236, 19.8942719, -40.6512566, 40.6912727

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 210

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1651997, upper bound: 43.1640689
time: 8.22 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1651997, upper bound: 43.1640689
time: 7.50 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -23.3544388, 18.6943626, -27.6074238, 22.0687752, -45.4232140, 46.3017883
1: -20.9448109, 16.7650013, -24.8290768, 19.7148075, -40.6596184, 41.5940781
2: -26.4616013, 16.6310959, -31.3623295, 19.5716553, -46.0332565, 47.9934235
3: -28.4084911, 14.3010941, -33.6544647, 16.7689457, -45.1774368, 47.9555550
4: -26.8841457, 19.1039753, -31.8542442, 22.4868736, -49.3710175, 50.9582214
5: -23.1337814, 18.0627384, -27.3937416, 21.3090439, -44.4428253, 45.4564819
6: -21.2974434, 21.1338425, -25.1621094, 24.9186096, -46.2160530, 46.2959518
7: -23.5010185, 22.2488403, -27.8134918, 26.3501854, -49.8512039, 50.0623322
8: -32.9020958, 15.8553543, -38.8210754, 18.5217113, -51.4238052, 54.6764297
9: -20.7569847, 21.1029491, -24.5357742, 24.9042416, -45.6612244, 45.6387253

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 210

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1651997, upper bound: 43.1640689
time: 7.19 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1651997, upper bound: 43.1640689
time: 6.92 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -30.0759506, 24.0378208, -20.6513538, 16.5334301, -46.6093788, 44.6891708
1: -27.0501595, 21.4804993, -18.5695152, 14.8639717, -41.9141312, 40.0500145
2: -34.1593552, 21.3206100, -23.4471931, 14.7044811, -48.8638344, 44.7678032
3: -36.6262054, 18.2534943, -25.2305984, 12.6094494, -49.2356567, 43.4840889
4: -34.6238632, 24.5312843, -23.9328575, 16.8519592, -51.4758224, 48.4641418
5: -29.7996025, 23.1861877, -20.5346985, 16.0403442, -45.8399429, 43.7208862
6: -27.4289970, 27.1393719, -18.7502670, 18.7113514, -46.1403503, 45.8896332
7: -30.3329735, 28.5733967, -20.7773762, 19.9313812, -50.2643547, 49.3507690
8: -42.1040802, 20.3677444, -29.4292374, 13.6304979, -55.7345734, 49.7969780
9: -26.7435570, 27.1401291, -18.3260479, 18.6176414, -45.3611984, 45.4661789

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 161

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1573072, upper bound: 43.1580590
time: 6.39 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1573072, upper bound: 43.1580590
time: 15.49 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -29.5967026, 23.6540718, -23.8698330, 19.0783596, -48.6750641, 47.5239029
1: -26.6339684, 21.1427345, -21.5229435, 17.1373291, -43.7712975, 42.6656799
2: -33.6242752, 20.9768353, -27.1399231, 16.9191303, -50.5433922, 48.1167603
3: -36.0558052, 17.9562683, -29.2412033, 14.4722672, -50.5280724, 47.1974678
4: -34.0949936, 24.1274338, -27.7045708, 19.4359055, -53.5308990, 51.8320045
5: -29.3371983, 22.8226223, -23.7715969, 18.4950581, -47.8322563, 46.5942154
6: -26.9723930, 26.7115154, -21.6679306, 21.6142559, -48.5866432, 48.3794479
7: -29.8493900, 28.1540871, -24.0594501, 23.0136719, -52.8630600, 52.2135391
8: -41.4805984, 19.9784050, -33.9011383, 15.6630182, -57.1436157, 53.8795395
9: -26.3093834, 26.6982288, -21.1748047, 21.4906883, -47.8000717, 47.8730316

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 161

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1571007, upper bound: 43.1577655
time: 4.79 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1571007, upper bound: 43.1577655
time: 6.97 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -22.9783134, 18.3921165, -24.8212605, 19.7938938, -42.7722054, 43.2133789
1: -20.6052856, 16.4939213, -22.3278275, 17.7673168, -38.3725929, 38.8217468
2: -26.0343285, 16.3594894, -28.2095890, 17.5403271, -43.5746536, 44.5690765
3: -27.9512844, 14.0681858, -30.3276253, 14.9705868, -42.9218712, 44.3958054
4: -26.4619102, 18.7890644, -28.7944679, 20.1804276, -46.6423378, 47.5835342
5: -22.7677155, 17.7772408, -24.7005329, 19.2105408, -41.9782562, 42.4777679
6: -20.9465790, 20.7910156, -22.5112324, 22.4322281, -43.3787994, 43.3022461
7: -23.1167812, 21.9159794, -25.0141010, 23.8934288, -47.0102081, 46.9300766
8: -32.3983574, 15.5573521, -35.1863060, 16.2160282, -48.6143761, 50.7436600
9: -20.4210205, 20.7523079, -22.0157547, 22.2884941, -42.7095108, 42.7680626

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1594191, upper bound: 43.1580225
time: 6.31 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1594191, upper bound: 43.1580225
time: 6.72 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -22.5202942, 18.0240364, -27.8921204, 22.2238121, -44.7441025, 45.9161568
1: -20.2053413, 16.1734238, -25.1461849, 19.9353065, -40.1406479, 41.3196106
2: -25.5182896, 16.0306435, -31.7306423, 19.6542072, -45.1724930, 47.7612839
3: -27.4041805, 13.7831717, -34.1551666, 16.7500763, -44.1542511, 47.9383354
4: -25.9538174, 18.4077549, -32.3896484, 22.6425076, -48.5963249, 50.7973938
5: -22.3251839, 17.4311295, -27.7880669, 21.5548458, -43.8800278, 45.2191925
6: -20.5104103, 20.3796368, -25.2981987, 25.1997604, -45.7101631, 45.6778336
7: -22.6541176, 21.5146999, -28.1388454, 26.8377609, -49.4918709, 49.6535454
8: -31.7989807, 15.1887999, -39.4463654, 18.1586037, -49.9575806, 54.6351662
9: -20.0065079, 20.3269386, -24.7342224, 25.0306740, -45.0371780, 45.0611496

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 210

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1473536, upper bound: 43.1436175
time: 10.59 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1389669, upper bound: 43.1387569
time: 8.72 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -29.6593227, 23.7053642, -24.4746323, 19.5180588, -49.1773834, 48.1799850
1: -26.6808624, 21.1823063, -22.0153656, 17.5235672, -44.2044296, 43.1976700
2: -33.7012062, 21.0249691, -27.8125877, 17.2981548, -50.9993591, 48.8375549
3: -36.1215057, 18.0007420, -29.8995171, 14.7661695, -50.8876762, 47.9002533
4: -34.1757507, 24.1698627, -28.3936348, 19.8971443, -54.0728951, 52.5634995
5: -29.3947144, 22.8692093, -24.3548241, 18.9453869, -48.3400955, 47.2240334
6: -27.0475101, 26.7642326, -22.1915016, 22.1180954, -49.1656036, 48.9557343
7: -29.9098320, 28.2047844, -24.6587791, 23.5690098, -53.4788437, 52.8635559
8: -41.5579529, 20.0392895, -34.7089233, 15.9813070, -57.5392570, 54.7482071
9: -26.3771362, 26.7613983, -21.7053986, 21.9736004, -48.3507385, 48.4667969

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 161

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1567568, upper bound: 43.1570004
time: 6.12 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1567568, upper bound: 43.1570004
time: 5.74 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -29.1908417, 23.3290462, -27.5377159, 21.9423008, -51.1331406, 50.8667603
1: -26.2728672, 20.8513699, -24.8277969, 19.6864758, -45.9593430, 45.6791649
2: -33.1753807, 20.6869125, -31.3256493, 19.4062290, -52.5816078, 52.0125618
3: -35.5628853, 17.7088757, -33.7190247, 16.5409698, -52.1038551, 51.4279022
4: -33.6543655, 23.7771778, -31.9813900, 22.3533421, -56.0077057, 55.7585678
5: -28.9429417, 22.5127697, -27.4354668, 21.2839050, -50.2268448, 49.9482346
6: -26.5988827, 26.3452511, -24.9715385, 24.8791656, -51.4780464, 51.3167877
7: -29.4361343, 27.7946434, -27.7759438, 26.5072517, -55.9433861, 55.5705795
8: -40.9457855, 19.6578312, -38.9597321, 17.9170761, -58.8628578, 58.6175613
9: -25.9510574, 26.3269081, -24.4171486, 24.7100067, -50.6610527, 50.7440491

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 161

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1452912, upper bound: 43.1427162
time: 17.38 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1344041, upper bound: 43.1364610
time: 6.48 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -22.9020443, 18.3290501, -23.3544388, 18.6943626, -41.5964050, 41.6834869
1: -20.5521736, 16.4315586, -20.9448109, 16.7650013, -37.3171730, 37.3763695
2: -25.9828796, 16.3024616, -26.4616013, 16.6310959, -42.6139755, 42.7640610
3: -27.9341412, 13.9924498, -28.4084911, 14.3010941, -42.2352295, 42.4009399
4: -26.4539204, 18.7113247, -26.8841457, 19.1039753, -45.5578957, 45.5954704
5: -22.7331505, 17.7413349, -23.1337814, 18.0627384, -40.7958794, 40.8751106
6: -20.8838615, 20.7296562, -21.2974434, 21.1338425, -42.0177040, 42.0270996
7: -23.0626984, 21.9438381, -23.5010185, 22.2488403, -45.3115387, 45.4448547
8: -32.3981972, 15.3495474, -32.9020958, 15.8553543, -48.2535515, 48.2516365
9: -20.3538895, 20.6743355, -20.7569847, 21.1029491, -41.4568405, 41.4313202

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 216

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1629115, upper bound: 43.1623733
time: 10.80 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1629115, upper bound: 43.1623733
time: 6.79 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -22.5403938, 18.0409832, -30.0759506, 24.0378208, -46.5782166, 48.1169357
1: -20.2266579, 16.1769676, -27.0501595, 21.4804993, -41.7071571, 43.2271271
2: -25.5684662, 16.0495071, -34.1593552, 21.3206100, -46.8890762, 50.2088585
3: -27.4882431, 13.7786808, -36.6262054, 18.2534943, -45.7417297, 50.4048843
4: -26.0366344, 18.4152279, -34.6238632, 24.5312843, -50.5679169, 53.0390930
5: -22.3723412, 17.4647179, -29.7996025, 23.1861877, -45.5585289, 47.2643127
6: -20.5505695, 20.4014969, -27.4289970, 27.1393719, -47.6899338, 47.8304939
7: -22.6931267, 21.6056499, -30.3329735, 28.5733967, -51.2665253, 51.9386215
8: -31.8997707, 15.1027889, -42.1040802, 20.3677444, -52.2675095, 57.2068672
9: -20.0301132, 20.3445053, -26.7435570, 27.1401291, -47.1702423, 47.0880623

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1580590, upper bound: 43.1573072
time: 6.48 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1577655, upper bound: 43.1571007
time: 7.90 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -26.7876816, 21.3605366, -22.9783134, 18.3921165, -45.1797981, 44.3388519
1: -24.0423031, 19.1318951, -20.6052856, 16.4939213, -40.5362167, 39.7371750
2: -30.4065971, 18.9392223, -26.0343285, 16.3594894, -46.7660866, 44.9735374
3: -32.6673126, 16.1869755, -27.9512844, 14.0681858, -46.7354889, 44.1382523
4: -30.9663239, 21.8034859, -26.4619102, 18.7890644, -49.7553864, 48.2653885
5: -26.6104794, 20.6868591, -22.7677155, 17.7772408, -44.3877106, 43.4545746
6: -24.3831501, 24.1858101, -20.9465790, 20.7910156, -45.1741638, 45.1323891
7: -26.9893494, 25.6285477, -23.1167812, 21.9159794, -48.9053268, 48.7453308
8: -37.7339745, 17.7501411, -32.3983574, 15.5573521, -53.2913246, 50.1484947
9: -23.7847672, 24.0774078, -20.4210205, 20.7523079, -44.5370750, 44.4984283

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 216

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1585989, upper bound: 43.1598294
time: 8.64 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1580250, upper bound: 43.1594623
time: 8.27 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -26.4208336, 21.0698738, -29.6593227, 23.7053642, -50.1261940, 50.7291946
1: -23.7136192, 18.8737316, -26.6808624, 21.1823063, -44.8959198, 45.5545959
2: -29.9881554, 18.6823311, -33.7012062, 21.0249691, -51.0131226, 52.3835373
3: -32.2171783, 15.9701700, -36.1215057, 18.0007420, -50.2179184, 52.0916748
4: -30.5451317, 21.5043716, -34.1757507, 24.1698627, -54.7149963, 55.6801224
5: -26.2462006, 20.4063492, -29.3947144, 22.8692093, -49.1154099, 49.8010559
6: -24.0450592, 23.8539181, -27.0475101, 26.7642326, -50.8092880, 50.9014282
7: -26.6144028, 25.2875996, -29.9098320, 28.2047844, -54.8191872, 55.1974220
8: -37.2330360, 17.4981289, -41.5579529, 20.0392895, -57.2723236, 59.0560837
9: -23.4559555, 23.7454834, -26.3771362, 26.7613983, -50.2173538, 50.1226196

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 216

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1574511, upper bound: 43.1570147
time: 5.70 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1570084, upper bound: 43.1567568
time: 6.98 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -22.9020443, 18.3290501, -22.9020443, 18.3290501, -41.2310905, 41.2310944
1: -20.5521736, 16.4315586, -20.5521736, 16.4315586, -36.9837341, 36.9837341
2: -25.9828796, 16.3024616, -25.9828796, 16.3024616, -42.2853394, 42.2853394
3: -27.9341412, 13.9924498, -27.9341412, 13.9924498, -41.9265900, 41.9265900
4: -26.4539204, 18.7113247, -26.4539204, 18.7113247, -45.1652451, 45.1652451
5: -22.7331505, 17.7413349, -22.7331505, 17.7413349, -40.4744797, 40.4744759
6: -20.8838615, 20.7296562, -20.8838615, 20.7296562, -41.6135178, 41.6135178
7: -23.0626984, 21.9438381, -23.0626984, 21.9438381, -45.0065384, 45.0065384
8: -32.3981972, 15.3495474, -32.3981972, 15.3495474, -47.7477379, 47.7477379
9: -20.3538895, 20.6743355, -20.3538895, 20.6743355, -41.0282249, 41.0282249

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 216

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1653313, upper bound: 43.1634412
time: 10.24 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1629441, upper bound: 43.1625268
time: 5.42 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -22.9020443, 18.3290501, -26.7876816, 21.3605366, -44.2625809, 45.1167297
1: -20.5521736, 16.4315586, -24.0423031, 19.1318951, -39.6840668, 40.4738617
2: -25.9828796, 16.3024616, -30.4065971, 18.9392223, -44.9221001, 46.7090530
3: -27.9341412, 13.9924498, -32.6673126, 16.1869755, -44.1211090, 46.6597595
4: -26.4539204, 18.7113247, -30.9663239, 21.8034859, -48.2574043, 49.6776505
5: -22.7331505, 17.7413349, -26.6104794, 20.6868591, -43.4200020, 44.3518066
6: -20.8838615, 20.7296562, -24.3831501, 24.1858101, -45.0696716, 45.1128082
7: -23.0626984, 21.9438381, -26.9893494, 25.6285477, -48.6912460, 48.9331894
8: -32.3981972, 15.3495474, -37.7339745, 17.7501411, -50.1483345, 53.0835114
9: -20.3538895, 20.6743355, -23.7847672, 24.0774078, -44.4312973, 44.4591026

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 216

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1653313, upper bound: 43.1634412
time: 6.71 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1629441, upper bound: 43.1625268
time: 5.49 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -26.7876816, 21.3605366, -22.8694687, 18.3120689, -45.0997505, 44.2300034
1: -24.0423031, 19.1318951, -20.5282936, 16.4207478, -40.4630470, 39.6601830
2: -30.4065971, 18.9392223, -25.9386673, 16.3010864, -46.7076797, 44.8778877
3: -32.6673126, 16.1869755, -27.8743763, 14.0004358, -46.6677399, 44.0613403
4: -30.9663239, 21.8034859, -26.3873043, 18.7023487, -49.6686707, 48.1907806
5: -26.6104794, 20.6868591, -22.6841850, 17.7144623, -44.3249435, 43.3710442
6: -24.3831501, 24.1858101, -20.8631172, 20.7021942, -45.0853424, 45.0489273
7: -26.9893494, 25.6285477, -23.0273075, 21.8752136, -48.8645630, 48.6558418
8: -37.7339745, 17.7501411, -32.3298035, 15.4246254, -53.1585960, 50.0799446
9: -23.7847672, 24.0774078, -20.3240070, 20.6624603, -44.4472275, 44.4014130

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 216

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1587995, upper bound: 43.1602809
time: 7.86 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1582401, upper bound: 43.1599464
time: 8.93 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -26.4208336, 21.0698738, -28.7722588, 23.0044365, -49.4252663, 49.8421288
1: -23.7136192, 18.8737316, -25.8817673, 20.5520782, -44.2656898, 44.7554893
2: -29.9881554, 18.6823311, -32.7011948, 20.4135303, -50.4016876, 51.3835258
3: -32.2171783, 15.9701700, -35.0884666, 17.4733181, -49.6904907, 51.0586319
4: -30.5451317, 21.5043716, -33.1833267, 23.4605789, -54.0057030, 54.6876984
5: -26.2462006, 20.4063492, -28.5302162, 22.2002411, -48.4464417, 48.9365616
6: -24.0450592, 23.8539181, -26.2552032, 25.9747066, -50.0197601, 50.1091232
7: -26.6144028, 25.2875996, -29.0176468, 27.4087143, -54.0231171, 54.3052330
8: -37.2330360, 17.4981289, -40.3886566, 19.4138985, -56.6469345, 57.8867874
9: -23.4559555, 23.7454834, -25.5820370, 25.9644318, -49.4203873, 49.3275223

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 216

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1575896, upper bound: 43.1573526
time: 17.62 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1571696, upper bound: 43.1571412
time: 7.80 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 27.00 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 27.00
Output dim: 8, lower bound: -43.1651997, upper bound: 43.1640689
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 27.00
Output dim: 8, lower bound: -43.1651997, upper bound: 43.1640689
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 27.00
Output dim: 8, lower bound: -43.1651997, upper bound: 43.1640689
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 27.00
Output dim: 8, lower bound: -43.1651997, upper bound: 43.1640689
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 27.00
Output dim: 8, lower bound: -43.1573072, upper bound: 43.1580590
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 27.00
Output dim: 8, lower bound: -43.1573072, upper bound: 43.1580590
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 27.00
Output dim: 8, lower bound: -43.1571007, upper bound: 43.1577655
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 27.00
Output dim: 8, lower bound: -43.1571007, upper bound: 43.1577655
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 27.00
Output dim: 8, lower bound: -43.1594191, upper bound: 43.1580225
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 27.00
Output dim: 8, lower bound: -43.1594191, upper bound: 43.1580225
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 27.00
Output dim: 8, lower bound: -43.1473536, upper bound: 43.1436175
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 27.00
Output dim: 8, lower bound: -43.1389669, upper bound: 43.1387569
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 27.00
Output dim: 8, lower bound: -43.1567568, upper bound: 43.1570004
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 27.00
Output dim: 8, lower bound: -43.1567568, upper bound: 43.1570004
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 27.00
Output dim: 8, lower bound: -43.1452912, upper bound: 43.1427162
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 27.00
Output dim: 8, lower bound: -43.1344041, upper bound: 43.1364610
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 27.00
Output dim: 8, lower bound: -43.1629115, upper bound: 43.1623733
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 27.00
Output dim: 8, lower bound: -43.1629115, upper bound: 43.1623733
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 27.00
Output dim: 8, lower bound: -43.1580590, upper bound: 43.1573072
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 27.00
Output dim: 8, lower bound: -43.1577655, upper bound: 43.1571007
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 27.00
Output dim: 8, lower bound: -43.1585989, upper bound: 43.1598294
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 27.00
Output dim: 8, lower bound: -43.1580250, upper bound: 43.1594623
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 27.00
Output dim: 8, lower bound: -43.1574511, upper bound: 43.1570147
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 27.00
Output dim: 8, lower bound: -43.1570084, upper bound: 43.1567568
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 27.00
Output dim: 8, lower bound: -43.1653313, upper bound: 43.1634412
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 27.00
Output dim: 8, lower bound: -43.1629441, upper bound: 43.1625268
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 27.00
Output dim: 8, lower bound: -43.1653313, upper bound: 43.1634412
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 27.00
Output dim: 8, lower bound: -43.1629441, upper bound: 43.1625268
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 27.00
Output dim: 8, lower bound: -43.1587995, upper bound: 43.1602809
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 27.00
Output dim: 8, lower bound: -43.1582401, upper bound: 43.1599464
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 27.00
Output dim: 8, lower bound: -43.1575896, upper bound: 43.1573526
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 27.00
Output dim: 8, lower bound: -43.1571696, upper bound: 43.1571412

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -22.0838280, 17.6713276, -22.0465679, 17.6483231, -39.7321434, 39.7178917
1: -19.7946892, 15.8504782, -19.7806206, 15.8277225, -35.6224060, 35.6310959
2: -25.0176582, 15.7122202, -25.0022011, 15.7044277, -40.7220840, 40.7144203
3: -26.8657780, 13.5100555, -26.8748474, 13.4851856, -40.3509636, 40.3849030
4: -25.4603004, 18.0406265, -25.4636402, 18.0102234, -43.4705200, 43.5042648
5: -21.8930244, 17.0952950, -21.8784924, 17.0861721, -38.9791946, 38.9737854
6: -20.1117935, 19.9776020, -20.0948544, 19.9537792, -40.0655708, 40.0724564
7: -22.2057571, 21.1207352, -22.1873360, 21.1411304, -43.3468819, 43.3080673
8: -31.1940880, 14.8486691, -31.2173977, 14.7717609, -45.9658508, 46.0660667
9: -19.6218357, 19.9186592, -19.5883236, 19.8942719, -39.5161057, 39.5069809

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1682997, upper bound: 43.1684840
time: 6.83 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1682997, upper bound: 43.1688686
time: 7.71 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -26.0283699, 20.7483025, -22.0465679, 17.6483231, -43.6766930, 42.7948685
1: -23.3428631, 18.5926342, -19.7806206, 15.8277225, -39.1705818, 38.3732491
2: -29.5100613, 18.3886719, -25.0022011, 15.7044277, -45.2144852, 43.3908653
3: -31.6753693, 15.7395706, -26.8748474, 13.4851856, -45.1605530, 42.6144180
4: -30.0476589, 21.1802006, -25.4636402, 18.0102234, -48.0578842, 46.6438408
5: -25.8350582, 20.0866661, -21.8784924, 17.0861721, -42.9212303, 41.9651566
6: -23.6673775, 23.4878654, -20.0948544, 19.9537792, -43.6211548, 43.5827141
7: -26.1899757, 24.8677521, -22.1873360, 21.1411304, -47.3311043, 47.0550880
8: -36.6159744, 17.2767296, -31.2173977, 14.7717609, -51.3877335, 48.4941254
9: -23.1045094, 23.3759117, -19.5883236, 19.8942719, -42.9987793, 42.9642334

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1682997, upper bound: 43.1684840
time: 6.63 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1682997, upper bound: 43.1688686
time: 6.50 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -22.0838280, 17.6713276, -27.6074238, 22.0687752, -44.1525993, 45.2787514
1: -19.7946892, 15.8504782, -24.8290768, 19.7148075, -39.5094986, 40.6795425
2: -25.0176582, 15.7122202, -31.3623295, 19.5716553, -44.5893135, 47.0745468
3: -26.8657780, 13.5100555, -33.6544647, 16.7689457, -43.6347237, 47.1645203
4: -25.4603004, 18.0406265, -31.8542442, 22.4868736, -47.9471741, 49.8948708
5: -21.8930244, 17.0952950, -27.3937416, 21.3090439, -43.2020607, 44.4890366
6: -20.1117935, 19.9776020, -25.1621094, 24.9186096, -45.0304031, 45.1397095
7: -22.2057571, 21.1207352, -27.8134918, 26.3501854, -48.5559387, 48.9342270
8: -31.1940880, 14.8486691, -38.8210754, 18.5217113, -49.7157974, 53.6697464
9: -19.6218357, 19.9186592, -24.5357742, 24.9042416, -44.5260773, 44.4544334

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 210

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1602410, upper bound: 43.1593955
time: 7.69 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1600547, upper bound: 43.1589882
time: 9.44 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -26.0283699, 20.7483025, -27.6074238, 22.0687752, -48.0971451, 48.3557281
1: -23.3428631, 18.5926342, -24.8290768, 19.7148075, -43.0576706, 43.4216995
2: -29.5100613, 18.3886719, -31.3623295, 19.5716553, -49.0817108, 49.7509918
3: -31.6753693, 15.7395706, -33.6544647, 16.7689457, -48.4443130, 49.3940353
4: -30.0476589, 21.1802006, -31.8542442, 22.4868736, -52.5345306, 53.0344467
5: -25.8350582, 20.0866661, -27.3937416, 21.3090439, -47.1440964, 47.4804077
6: -23.6673775, 23.4878654, -25.1621094, 24.9186096, -48.5859871, 48.6499748
7: -26.1899757, 24.8677521, -27.8134918, 26.3501854, -52.5401611, 52.6812439
8: -36.6159744, 17.2767296, -38.8210754, 18.5217113, -55.1376877, 56.0978050
9: -23.1045094, 23.3759117, -24.5357742, 24.9042416, -48.0087509, 47.9116859

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 210

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1602410, upper bound: 43.1593955
time: 7.56 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1600547, upper bound: 43.1589882
time: 6.27 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -28.7815094, 22.9958019, -20.6513538, 16.5334301, -45.3149338, 43.6471558
1: -25.8849869, 20.5460300, -18.5695152, 14.8639717, -40.7489471, 39.1155434
2: -32.6967621, 20.3818874, -23.4471931, 14.7044811, -47.4012451, 43.8290787
3: -35.0503082, 17.4472389, -25.2305984, 12.6094494, -47.6597595, 42.6778374
4: -33.1796188, 23.4401627, -23.9328575, 16.8519592, -50.0315742, 47.3730202
5: -28.5380459, 22.1972466, -20.5346985, 16.0403442, -44.5783920, 42.7319412
6: -26.2180119, 25.9650707, -18.7502670, 18.7113514, -44.9293633, 44.7153244
7: -29.0135136, 27.4244308, -20.7773762, 19.9313812, -48.9448891, 48.2018051
8: -40.3718414, 19.3335190, -29.4292374, 13.6304979, -54.0023384, 48.7627563
9: -25.5833416, 25.9409561, -18.3260479, 18.6176414, -44.2009811, 44.2670059

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 92

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1572821, upper bound: 43.1579116
time: 8.24 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1572821, upper bound: 43.1580590
time: 7.06 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 16.84 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 16.84
Output dim: 8, lower bound: -43.1682997, upper bound: 43.1684840
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 16.84
Output dim: 8, lower bound: -43.1682997, upper bound: 43.1688686
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 16.84
Output dim: 8, lower bound: -43.1682997, upper bound: 43.1684840
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 16.84
Output dim: 8, lower bound: -43.1682997, upper bound: 43.1688686
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 16.84
Output dim: 8, lower bound: -43.1602410, upper bound: 43.1593955
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 16.84
Output dim: 8, lower bound: -43.1600547, upper bound: 43.1589882
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 16.84
Output dim: 8, lower bound: -43.1602410, upper bound: 43.1593955
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 16.84
Output dim: 8, lower bound: -43.1600547, upper bound: 43.1589882
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 16.84
Output dim: 8, lower bound: -43.1572821, upper bound: 43.1579116
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 16.84
Output dim: 8, lower bound: -43.1572821, upper bound: 43.1580590
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.84
Output dim: 8, lower bound: -43.1573072, upper bound: 43.1580590
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.84
Output dim: 8, lower bound: -43.1571007, upper bound: 43.1577655
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.84
Output dim: 8, lower bound: -43.1571007, upper bound: 43.1577655
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.84
Output dim: 8, lower bound: -43.1594191, upper bound: 43.1580225
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.84
Output dim: 8, lower bound: -43.1594191, upper bound: 43.1580225
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.84
Output dim: 8, lower bound: -43.1473536, upper bound: 43.1436175
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.84
Output dim: 8, lower bound: -43.1389669, upper bound: 43.1387569
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.84
Output dim: 8, lower bound: -43.1567568, upper bound: 43.1570004
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.84
Output dim: 8, lower bound: -43.1567568, upper bound: 43.1570004
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.84
Output dim: 8, lower bound: -43.1452912, upper bound: 43.1427162
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.84
Output dim: 8, lower bound: -43.1344041, upper bound: 43.1364610
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.84
Output dim: 8, lower bound: -43.1629115, upper bound: 43.1623733
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.84
Output dim: 8, lower bound: -43.1629115, upper bound: 43.1623733
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.84
Output dim: 8, lower bound: -43.1580590, upper bound: 43.1573072
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.84
Output dim: 8, lower bound: -43.1577655, upper bound: 43.1571007
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.84
Output dim: 8, lower bound: -43.1585989, upper bound: 43.1598294
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.84
Output dim: 8, lower bound: -43.1580250, upper bound: 43.1594623
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.84
Output dim: 8, lower bound: -43.1574511, upper bound: 43.1570147
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.84
Output dim: 8, lower bound: -43.1570084, upper bound: 43.1567568
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.84
Output dim: 8, lower bound: -43.1653313, upper bound: 43.1634412
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.84
Output dim: 8, lower bound: -43.1629441, upper bound: 43.1625268
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.84
Output dim: 8, lower bound: -43.1653313, upper bound: 43.1634412
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.84
Output dim: 8, lower bound: -43.1629441, upper bound: 43.1625268
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.84
Output dim: 8, lower bound: -43.1587995, upper bound: 43.1602809
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.84
Output dim: 8, lower bound: -43.1582401, upper bound: 43.1599464
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.84
Output dim: 8, lower bound: -43.1575896, upper bound: 43.1573526
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.84
Output dim: 8, lower bound: -43.1571696, upper bound: 43.1571412
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=51.530487060546875
rel_dist={8: [-43.17090795605019, 43.17090794479026]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 50

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1705002, upper bound: 43.1706408
time: 6.75 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1707437, upper bound: 43.1707437
time: 5.37 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 12.27 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 12.27
Output dim: 8, lower bound: -43.1705002, upper bound: 43.1706408
IS_A2, status: Status.UNKNOWN, split count: 1, time: 12.27
Output dim: 8, lower bound: -43.1707437, upper bound: 43.1707437

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -24.3706722, 19.5024204, -23.7293777, 18.9999046, -43.3705750, 43.2317963
1: -21.8586521, 17.4819832, -21.3050957, 17.0339031, -38.8925552, 38.7870789
2: -27.6241531, 17.3405914, -26.9258842, 16.9057484, -44.5298996, 44.2664757
3: -29.6621819, 14.9008579, -28.9340725, 14.5140362, -44.1762161, 43.8349304
4: -28.0579205, 19.9350491, -27.3763237, 19.4126072, -47.4705276, 47.3113708
5: -24.1463985, 18.8390179, -23.5370712, 18.3707199, -42.5171204, 42.3760872
6: -22.2339973, 22.0551300, -21.6582737, 21.4876881, -43.7216873, 43.7134018
7: -24.5400143, 23.1987343, -23.9124851, 22.6648006, -47.2048111, 47.1112175
8: -34.2982140, 16.5464706, -33.5120163, 16.0368385, -50.3350487, 50.0584869
9: -21.6667118, 22.0264530, -21.0936375, 21.4534988, -43.1202087, 43.1200905

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 183

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1681183, upper bound: 43.1684686
time: 6.80 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1680409, upper bound: 43.1683212
time: 6.30 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -24.2108173, 19.3813133, -24.6299152, 19.7143040, -43.9251137, 44.0112228
1: -21.7347298, 17.3723087, -22.1095390, 17.6670246, -39.4017563, 39.4818420
2: -27.4703083, 17.2466469, -27.9456024, 17.5415535, -45.0118561, 45.1922493
3: -29.5238304, 14.8024702, -30.0350666, 15.0510893, -44.5749130, 44.8375359
4: -27.9244385, 19.8066959, -28.4033089, 20.1493416, -48.0737801, 48.2100067
5: -24.0113335, 18.7387772, -24.4249897, 19.0596142, -43.0709457, 43.1637650
6: -22.1033802, 21.9187889, -22.4893761, 22.2958450, -44.3992233, 44.4081650
7: -24.3987465, 23.1103039, -24.8233261, 23.4988098, -47.8975563, 47.9336281
8: -34.1606636, 16.3779488, -34.7292480, 16.6722507, -50.8329163, 51.1071930
9: -21.5237293, 21.8889427, -21.8992653, 22.2679310, -43.7916603, 43.7882080

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 50

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1687986, upper bound: 43.1688728
time: 7.66 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1687308, upper bound: 43.1687308
time: 25.09 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 34.20 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 34.20
Output dim: 8, lower bound: -43.1681183, upper bound: 43.1684686
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 34.20
Output dim: 8, lower bound: -43.1680409, upper bound: 43.1683212
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 34.20
Output dim: 8, lower bound: -43.1687986, upper bound: 43.1688728
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 34.20
Output dim: 8, lower bound: -43.1687308, upper bound: 43.1687308

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -24.0274734, 19.2259598, -22.4434185, 17.9652138, -41.9926872, 41.6693802
1: -21.5484715, 17.2337952, -20.1415367, 16.1099892, -37.6584511, 37.3753319
2: -27.2355461, 17.0914726, -25.4634438, 15.9779587, -43.2135010, 42.5549164
3: -29.2451973, 14.6875143, -27.3715172, 13.7165775, -42.9617767, 42.0590248
4: -27.6738758, 19.6465912, -25.9315643, 18.3364410, -46.0103149, 45.5781517
5: -23.8128948, 18.5773220, -22.2795792, 17.3917427, -41.2046356, 40.8569031
6: -21.9131813, 21.7422218, -20.4598961, 20.3185863, -42.2317657, 42.2021179
7: -24.1890182, 22.8952007, -22.6008167, 21.5184364, -45.7074432, 45.4960136
8: -33.8406487, 16.2714901, -31.7780952, 15.0259771, -48.8666267, 48.0495834
9: -21.3593922, 21.7078362, -19.9446354, 20.2580185, -41.6174088, 41.6524734

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 183

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1638498, upper bound: 43.1631056
time: 8.21 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1621450, upper bound: 43.1624340
time: 7.46 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -23.4692764, 18.7772446, -26.2538834, 20.9394150, -44.4086838, 45.0311279
1: -21.0457382, 16.8315544, -23.5655441, 18.7577667, -39.8035011, 40.3970909
2: -26.6018257, 16.6888313, -29.8010178, 18.5634918, -45.1653175, 46.4898453
3: -28.5657501, 14.3411970, -32.0147247, 15.8700066, -44.4357567, 46.3559227
4: -27.0464745, 19.1799641, -30.3568516, 21.3696156, -48.4160919, 49.5368118
5: -23.2684937, 18.1534843, -26.0840454, 20.2800159, -43.5485077, 44.2375298
6: -21.3925285, 21.2332096, -23.8919868, 23.7084827, -45.1010132, 45.1251984
7: -23.6194878, 22.4003086, -26.4481239, 25.1325226, -48.7520103, 48.8484344
8: -33.0904808, 15.8325100, -37.0121346, 17.3814564, -50.4719391, 52.8446426
9: -20.8608608, 21.1862316, -23.3076611, 23.5980835, -44.4589424, 44.4938927

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 183

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1635388, upper bound: 43.1626125
time: 7.32 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1619750, upper bound: 43.1620769
time: 21.81 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -23.8581047, 19.0980396, -23.2867889, 18.6356487, -42.4937515, 42.3848267
1: -21.4170055, 17.1183109, -20.8980331, 16.7013950, -38.1184006, 38.0163422
2: -27.0702362, 16.9915276, -26.4206753, 16.5733204, -43.6435547, 43.4121971
3: -29.0950127, 14.5852718, -28.4045334, 14.2210846, -43.3160973, 42.9898071
4: -27.5283089, 19.5113564, -26.8937950, 19.0252609, -46.5535660, 46.4051514
5: -23.6685734, 18.4695988, -23.1144886, 18.0344944, -41.7030678, 41.5840874
6: -21.7743282, 21.5979595, -21.2378349, 21.0762005, -42.8505287, 42.8357925
7: -24.0376511, 22.7972603, -23.4515667, 22.3013382, -46.3389893, 46.2488174
8: -33.6883965, 16.0996914, -32.9226608, 15.6192646, -49.3076553, 49.0223503
9: -21.2082195, 21.5628204, -20.6984024, 21.0237617, -42.2319794, 42.2612228

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 196

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1641638, upper bound: 43.1641551
time: 7.20 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1640451, upper bound: 43.1642098
time: 7.25 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -23.2807846, 18.6340904, -27.1942177, 21.6846485, -44.9654312, 45.8283081
1: -20.8955154, 16.7026577, -24.4111290, 19.4196281, -40.3151398, 41.1137848
2: -26.4127350, 16.5755234, -30.8662395, 19.2280502, -45.6407776, 47.4417648
3: -28.3894768, 14.2289448, -33.1678391, 16.4308186, -44.8202972, 47.3967819
4: -26.8767815, 19.0287514, -31.4320526, 22.1347961, -49.0115776, 50.4608040
5: -23.1044922, 18.0298748, -27.0135536, 20.9974918, -44.1019821, 45.0434265
6: -21.2365646, 21.0722923, -24.7594185, 24.5535793, -45.7901459, 45.8317108
7: -23.4471989, 22.2817726, -27.3982124, 26.0073681, -49.4545631, 49.6799850
8: -32.9089661, 15.6511354, -38.2802734, 18.0389023, -50.9478645, 53.9314079
9: -20.6923428, 21.0251579, -24.1517849, 24.4449902, -45.1373329, 45.1769371

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 50

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1637315, upper bound: 43.1627296
time: 8.01 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1621556, upper bound: 43.1621556
time: 9.18 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 18.64 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 18.64
Output dim: 8, lower bound: -43.1638498, upper bound: 43.1631056
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 18.64
Output dim: 8, lower bound: -43.1621450, upper bound: 43.1624340
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 18.64
Output dim: 8, lower bound: -43.1635388, upper bound: 43.1626125
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 18.64
Output dim: 8, lower bound: -43.1619750, upper bound: 43.1620769
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 18.64
Output dim: 8, lower bound: -43.1641638, upper bound: 43.1641551
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 18.64
Output dim: 8, lower bound: -43.1640451, upper bound: 43.1642098
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 18.64
Output dim: 8, lower bound: -43.1637315, upper bound: 43.1627296
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 18.64
Output dim: 8, lower bound: -43.1621556, upper bound: 43.1621556

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -23.0166969, 18.4229088, -22.3311176, 17.8756771, -40.8923721, 40.7540283
1: -20.6393051, 16.5213699, -20.0402870, 16.0308933, -36.6701889, 36.5616570
2: -26.0778675, 16.3866272, -25.3347206, 15.8994627, -41.9773293, 41.7213364
3: -27.9982262, 14.0914955, -27.2325745, 13.6502914, -41.6485176, 41.3240700
4: -26.5054054, 18.8207893, -25.8016224, 18.2445984, -44.7500038, 44.6224136
5: -22.8055935, 17.8061886, -22.1673889, 17.3057728, -40.1113586, 39.9735794
6: -20.9821720, 20.8259583, -20.3563099, 20.2168274, -41.1989975, 41.1822624
7: -23.1558418, 21.9507389, -22.4859543, 21.4131775, -44.5690193, 44.4366913
8: -32.4507065, 15.5854120, -31.6229496, 14.9501095, -47.4008179, 47.2083626
9: -20.4549904, 20.7888737, -19.8441353, 20.1557331, -40.6107140, 40.6330109

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 14

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1587060, upper bound: 43.1581355
time: 9.07 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1585692, upper bound: 43.1578447
time: 6.72 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -29.7193317, 23.7543755, -21.6301098, 17.3174362, -47.0367661, 45.3844833
1: -26.7321701, 21.2260361, -19.4073029, 15.5375290, -42.2696991, 40.6333237
2: -33.7635612, 21.0684204, -24.5320721, 15.4089994, -49.1725616, 45.6004944
3: -36.1951561, 18.0335808, -26.3678398, 13.2361298, -49.4312859, 44.4014091
4: -34.2366905, 24.2265396, -24.9904232, 17.6698704, -51.9065628, 49.2169571
5: -29.4471245, 22.9202232, -21.4668045, 16.7697449, -46.2168617, 44.3870163
6: -27.1004105, 26.8186226, -19.7100830, 19.5798626, -46.6802750, 46.5286980
7: -29.9742260, 28.2555981, -21.7682114, 20.7562103, -50.7304382, 50.0238113
8: -41.6308746, 20.0867081, -30.6566315, 14.4753723, -56.1062431, 50.7433395
9: -26.4267998, 26.8178082, -19.2152061, 19.5178223, -45.9446220, 46.0330124

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 14

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1568953, upper bound: 43.1573334
time: 8.86 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1567754, upper bound: 43.1571593
time: 7.78 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -22.4738045, 17.9855995, -26.1413803, 20.8500710, -43.3238678, 44.1269798
1: -20.1489410, 16.1305790, -23.4642181, 18.6785488, -38.8274841, 39.5947952
2: -25.4607315, 15.9941998, -29.6723118, 18.4850159, -43.9457436, 45.6665115
3: -27.3360424, 13.7547235, -31.8757477, 15.8035908, -43.1396332, 45.6304703
4: -25.8956299, 18.3678799, -30.2271194, 21.2777100, -47.1733398, 48.5950012
5: -22.2747192, 17.3931942, -25.9717522, 20.1941795, -42.4688988, 43.3649445
6: -20.4757290, 20.3309135, -23.7883873, 23.6066399, -44.0823631, 44.1193008
7: -22.6025696, 21.4675064, -26.3332062, 25.0272579, -47.6298294, 47.8007126
8: -31.7193565, 15.1603193, -36.8576431, 17.3051434, -49.0244980, 52.0179596
9: -19.9706726, 20.2803593, -23.2070560, 23.4960251, -43.4666977, 43.4874153

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 216

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1583765, upper bound: 43.1576914
time: 8.28 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1581200, upper bound: 43.1573022
time: 7.38 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -29.0856514, 23.2412491, -25.4289837, 20.2833443, -49.3689957, 48.6702347
1: -26.1672516, 20.7624626, -22.8232231, 18.1771851, -44.3444366, 43.5856857
2: -33.0477180, 20.6027565, -28.8580952, 17.9862194, -51.0339279, 49.4608536
3: -35.4258995, 17.6480961, -30.9976692, 15.3823566, -50.8082581, 48.6457672
4: -33.5265732, 23.6888657, -29.4064198, 20.6946602, -54.2212334, 53.0952835
5: -28.8367386, 22.4262333, -25.2613945, 19.6503983, -48.4871368, 47.6876259
6: -26.5031662, 26.2482586, -23.1313934, 22.9610634, -49.4642258, 49.3796539
7: -29.3153381, 27.6850815, -25.6049995, 24.3621254, -53.6774635, 53.2900810
8: -40.7987404, 19.5758820, -35.8796997, 16.8193722, -57.6181068, 55.4555817
9: -25.8558006, 26.2390919, -22.5691872, 22.8491077, -48.7048988, 48.8082809

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1567514, upper bound: 43.1570541
time: 6.66 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1565995, upper bound: 43.1567698
time: 6.67 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -21.8107224, 17.4580841, -22.6156349, 18.0982246, -39.9089470, 40.0737190
1: -19.6237011, 15.6917381, -20.3104324, 16.2345753, -35.8582726, 36.0021706
2: -24.7673454, 15.5326052, -25.6675911, 16.0956974, -40.8630371, 41.2001953
3: -26.6479301, 13.3193970, -27.6043167, 13.8069324, -40.4548607, 40.9237137
4: -25.2511616, 17.8157063, -26.1498985, 18.4678497, -43.7190094, 43.9656067
5: -21.6812630, 16.9218407, -22.4616966, 17.5285378, -39.2098007, 39.3835335
6: -19.8263588, 19.7629681, -20.5993176, 20.4757118, -40.3020630, 40.3622818
7: -21.9688492, 20.9880142, -22.7770596, 21.7092152, -43.6780624, 43.7650757
8: -31.0022793, 14.4879150, -32.0442657, 15.0915318, -46.0938072, 46.5321693
9: -19.3596573, 19.6809883, -20.0937881, 20.4067116, -39.7663689, 39.7747765

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1577456, upper bound: 43.1589220
time: 6.44 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1574921, upper bound: 43.1575417
time: 6.39 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -25.0404472, 20.0120049, -22.1884670, 17.7573490, -42.7977982, 42.2004700
1: -22.5797195, 17.9686184, -19.9391575, 15.9402485, -38.5199661, 37.9077759
2: -28.4670925, 17.7540874, -25.1885605, 15.7885637, -44.2556572, 42.9426498
3: -30.6687870, 15.1888447, -27.0988503, 13.5413494, -44.2101364, 42.2876930
4: -29.0285683, 20.4057255, -25.6771030, 18.1155663, -47.1441345, 46.0828247
5: -24.9263039, 19.3847256, -22.0480652, 17.2077980, -42.1340942, 41.4327927
6: -22.7554379, 22.6729450, -20.1921520, 20.0951920, -42.8506317, 42.8650970
7: -25.2554893, 24.0742607, -22.3433514, 21.3348999, -46.5903778, 46.4176025
8: -35.4793243, 16.5323257, -31.4918404, 14.7521801, -50.2315063, 48.0241661
9: -22.2196407, 22.5628891, -19.7063713, 20.0137653, -42.2333984, 42.2692604

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 92

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1577081, upper bound: 43.1588368
time: 8.47 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1572100, upper bound: 43.1574639
time: 6.93 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -22.3649044, 17.9046574, -27.0820293, 21.5957279, -43.9606323, 44.9866867
1: -20.0698032, 16.0568008, -24.3104362, 19.3404942, -39.4102974, 40.3672371
2: -25.3626633, 15.9357929, -30.7382355, 19.1495304, -44.5121880, 46.6740265
3: -27.2571144, 13.6871891, -33.0297165, 16.3644104, -43.6215248, 46.7169037
4: -25.8176422, 18.2792492, -31.3030739, 22.0433445, -47.8609848, 49.5823212
5: -22.1898022, 17.3288345, -26.9020863, 20.9115620, -43.1013641, 44.2309189
6: -20.3925343, 20.2419491, -24.6558762, 24.4521084, -44.8446426, 44.8978271
7: -22.5107479, 21.4234581, -27.2835636, 25.9029083, -48.4136581, 48.7070236
8: -31.6445427, 15.0310926, -38.1271133, 17.9621181, -49.6066589, 53.1582069
9: -19.8725357, 20.1904659, -24.0511398, 24.3434982, -44.2160301, 44.2416000

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1587542, upper bound: 43.1579115
time: 6.39 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1586416, upper bound: 43.1576015
time: 7.73 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -27.9443855, 22.3447609, -26.3535748, 21.0176640, -48.9620476, 48.6983337
1: -25.1368427, 19.9594421, -23.6564178, 18.8270035, -43.9638443, 43.6158562
2: -31.7523346, 19.8244228, -29.9073887, 18.6384430, -50.3907776, 49.7318115
3: -34.0747910, 16.9819317, -32.1344528, 15.9329300, -50.0077209, 49.1163864
4: -32.2333450, 22.7744884, -30.4659004, 21.4487400, -53.6820831, 53.2403870
5: -27.7168789, 21.5684204, -26.1780128, 20.3541985, -48.0710678, 47.7464256
6: -25.4803562, 25.2302933, -23.9835815, 23.7920818, -49.2724380, 49.2138748
7: -28.1641350, 26.6416416, -26.5392857, 25.2251186, -53.3892517, 53.1809273
8: -39.2703972, 18.8056641, -37.1325912, 17.4603386, -56.7307358, 55.9382553
9: -24.8422623, 25.2171440, -23.3975105, 23.6833572, -48.5256195, 48.6146507

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 216

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1571258, upper bound: 43.1572659
time: 6.57 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1570256, upper bound: 43.1570256
time: 6.78 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 14.81 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 14.81
Output dim: 8, lower bound: -43.1587060, upper bound: 43.1581355
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 14.81
Output dim: 8, lower bound: -43.1585692, upper bound: 43.1578447
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 14.81
Output dim: 8, lower bound: -43.1568953, upper bound: 43.1573334
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 14.81
Output dim: 8, lower bound: -43.1567754, upper bound: 43.1571593
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 14.81
Output dim: 8, lower bound: -43.1583765, upper bound: 43.1576914
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 14.81
Output dim: 8, lower bound: -43.1581200, upper bound: 43.1573022
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 14.81
Output dim: 8, lower bound: -43.1567514, upper bound: 43.1570541
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 14.81
Output dim: 8, lower bound: -43.1565995, upper bound: 43.1567698
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 14.81
Output dim: 8, lower bound: -43.1577456, upper bound: 43.1589220
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 14.81
Output dim: 8, lower bound: -43.1574921, upper bound: 43.1575417
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 14.81
Output dim: 8, lower bound: -43.1577081, upper bound: 43.1588368
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 14.81
Output dim: 8, lower bound: -43.1572100, upper bound: 43.1574639
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 14.81
Output dim: 8, lower bound: -43.1587542, upper bound: 43.1579115
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 14.81
Output dim: 8, lower bound: -43.1586416, upper bound: 43.1576015
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 14.81
Output dim: 8, lower bound: -43.1571258, upper bound: 43.1572659
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 14.81
Output dim: 8, lower bound: -43.1570256, upper bound: 43.1570256

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -22.3169975, 17.8605328, -20.3841782, 16.3228855, -38.6398849, 38.2447128
1: -20.0269089, 16.0306282, -18.3283730, 14.6772566, -34.7041588, 34.3589973
2: -25.2892475, 15.8863840, -23.1450958, 14.5153866, -39.8046341, 39.0314789
3: -27.1594620, 13.6564550, -24.9023781, 12.4489145, -39.6083755, 38.5588303
4: -25.7276840, 18.2401600, -23.6283379, 16.6355133, -42.3631973, 41.8684998
5: -22.1280937, 17.2756977, -20.2714119, 15.8366966, -37.9647903, 37.5471115
6: -20.3164902, 20.1969223, -18.5031204, 18.4741344, -38.7906113, 38.7000427
7: -22.4498482, 21.3343124, -20.5076866, 19.6823425, -42.1321793, 41.8419991
8: -31.5306454, 15.0288601, -29.0695992, 13.4439240, -44.9745636, 44.0984573
9: -19.8234673, 20.1410465, -18.0882988, 18.3783207, -38.2017746, 38.2293396

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 216

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1438005, upper bound: 43.1426490
time: 7.55 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1378680, upper bound: 43.1381622
time: 8.27 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -21.8758907, 17.5039005, -23.4327221, 18.7319031, -40.6077881, 40.9366226
1: -19.6435127, 15.7230968, -21.1363316, 16.8326645, -36.4761772, 36.8594284
2: -24.7908268, 15.5678511, -26.6434669, 16.6088963, -41.3997231, 42.2113190
3: -26.6363258, 13.3798409, -28.7116852, 14.2134190, -40.8497467, 42.0915260
4: -25.2399521, 17.8734360, -27.2091789, 19.0822544, -44.3222046, 45.0826149
5: -21.7022305, 16.9421654, -23.3472023, 18.1645470, -39.8667679, 40.2893677
6: -19.8945198, 19.8017559, -21.2644615, 21.2288094, -41.1233292, 41.0662155
7: -22.0012169, 20.9485722, -23.6152458, 22.6135998, -44.6148148, 44.5638199
8: -30.9528065, 14.6686840, -33.3240204, 15.3498402, -46.3026428, 47.9927063
9: -19.4216499, 19.7281837, -20.7852230, 21.0994339, -40.5210800, 40.5134048

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1434155, upper bound: 43.1420335
time: 9.00 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1376026, upper bound: 43.1378050
time: 6.99 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -28.9991226, 23.1766033, -19.7339573, 15.8070164, -44.8061371, 42.9105568
1: -26.1037750, 20.7163696, -17.7355175, 14.2161045, -40.3198738, 38.4518890
2: -32.9571571, 20.5505810, -22.3953686, 14.0610046, -47.0181618, 42.9459496
3: -35.3330536, 17.5885735, -24.0975838, 12.0625973, -47.3956528, 41.6861572
4: -33.4344025, 23.6209812, -22.8682518, 16.1034164, -49.5378189, 46.4892349
5: -28.7532806, 22.3700504, -19.6212692, 15.3362827, -44.0895615, 41.9913177
6: -26.4134827, 26.1744766, -17.9026260, 17.8808994, -44.2943726, 44.0771027
7: -29.2463264, 27.6234932, -19.8336678, 19.0664406, -48.3127670, 47.4571609
8: -40.6896744, 19.5064812, -28.1663380, 13.0126324, -53.7023010, 47.6728210
9: -25.7749023, 26.1528854, -17.5053406, 17.7861862, -43.5610847, 43.6582184

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 161

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1425270, upper bound: 43.1418076
time: 9.52 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1343339, upper bound: 43.1359994
time: 23.02 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -28.5200386, 22.7891617, -22.7457905, 18.1865540, -46.7065926, 45.5349503
1: -25.6881828, 20.3784237, -20.5171432, 16.3503380, -42.0385208, 40.8955650
2: -32.4198990, 20.2016048, -25.8569126, 16.1291561, -48.5490570, 46.0585175
3: -34.7637291, 17.2879791, -27.8628922, 13.8107882, -48.5745163, 45.1508713
4: -32.9036064, 23.2180977, -26.4138432, 18.5216484, -51.4252548, 49.6319427
5: -28.2930984, 22.0044937, -22.6610374, 17.6394863, -45.9325867, 44.6655312
6: -25.9522171, 25.7461300, -20.6313820, 20.6074867, -46.5597000, 46.3774986
7: -28.7591000, 27.2065048, -22.9091454, 21.9694061, -50.7285080, 50.1156502
8: -40.0657578, 19.1100826, -32.3758049, 14.8907099, -54.9564667, 51.4858856
9: -25.3376102, 25.7051449, -20.1742649, 20.4777622, -45.8153648, 45.8794098

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 52

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1421543, upper bound: 43.1413049
time: 8.64 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1341723, upper bound: 43.1357489
time: 7.72 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -21.7958412, 17.4394684, -24.1253586, 19.2420921, -41.0379333, 41.5648270
1: -19.5551434, 15.6554518, -21.7005310, 17.2795792, -36.8347168, 37.3559837
2: -24.6955929, 15.5096703, -27.4160900, 17.0530434, -41.7486267, 42.9257584
3: -26.5236702, 13.3317862, -29.4712086, 14.5597439, -41.0834122, 42.8029938
4: -25.1417885, 17.8061275, -27.9915352, 19.6146431, -44.7564316, 45.7976608
5: -21.6169014, 16.8782578, -24.0095882, 18.6792908, -40.2961922, 40.8878479
6: -19.8299046, 19.7216530, -21.8713455, 21.8066940, -41.6365967, 41.5929985
7: -21.9173546, 20.8683548, -24.3057594, 23.2413406, -45.1586914, 45.1741142
8: -30.8247433, 14.6236200, -34.2338943, 15.7437305, -46.5684700, 48.8575134
9: -19.3577938, 19.6525574, -21.3932610, 21.6604080, -41.0182037, 41.0458183

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 210

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1427077, upper bound: 43.1413273
time: 8.30 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1372735, upper bound: 43.1372832
time: 7.62 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -21.3600750, 17.0900135, -27.0307732, 21.5381966, -42.8982697, 44.1207886
1: -19.1763878, 15.3527975, -24.3743401, 19.3348789, -38.5112686, 39.7271385
2: -24.2049255, 15.1963387, -30.7471828, 19.0490608, -43.2539749, 45.9435196
3: -26.0071373, 13.0599041, -33.1030426, 16.2420673, -42.2492065, 46.1629486
4: -24.6591816, 17.4437008, -31.4012032, 21.9439926, -46.6031647, 48.8449020
5: -21.1954422, 16.5507832, -26.9374962, 20.8993435, -42.0947800, 43.4882812
6: -19.4135933, 19.3326721, -24.5042477, 24.4293079, -43.8429031, 43.8369179
7: -21.4726944, 20.4869003, -27.2594223, 26.0355949, -47.5082893, 47.7463226
8: -30.2601585, 14.2740879, -38.2786560, 17.5610981, -47.8212433, 52.5527420
9: -18.9601250, 19.2496719, -23.9632912, 24.2564220, -43.2165451, 43.2129593

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 210

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1420324, upper bound: 43.1404044
time: 14.73 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1368476, upper bound: 43.1367260
time: 7.92 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -28.4017792, 22.6917801, -23.4391308, 18.6964054, -47.0981827, 46.1309128
1: -25.5681000, 20.2822857, -21.0818481, 16.7960377, -42.3641357, 41.3641357
2: -32.2794495, 20.1122131, -26.6298447, 16.5730305, -48.8524780, 46.7420578
3: -34.6057930, 17.2232819, -28.6226120, 14.1559124, -48.7617035, 45.8458939
4: -32.7666817, 23.1198578, -27.1958847, 19.0534210, -51.8201027, 50.3157349
5: -28.1739635, 21.9091892, -23.3252678, 18.1538982, -46.3278580, 45.2344551
6: -25.8496265, 25.6351662, -21.2387161, 21.1845589, -47.0341873, 46.8738823
7: -28.6274643, 27.0838280, -23.6002445, 22.5979500, -51.2254066, 50.6840706
8: -39.8980560, 19.0331230, -33.2873535, 15.2814550, -55.1795120, 52.3204765
9: -25.2398548, 25.6046677, -20.7794189, 21.0372200, -46.2770653, 46.3840790

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 52

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1416049, upper bound: 43.1407731
time: 8.89 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1340917, upper bound: 43.1354146
time: 20.51 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -27.9290390, 22.3094101, -26.3333244, 20.9832726, -48.9123039, 48.6427345
1: -25.1565628, 19.9507065, -23.7458420, 18.8447609, -44.0013237, 43.6965446
2: -31.7471256, 19.7691383, -29.9491272, 18.5607662, -50.3078880, 49.7182579
3: -34.0428238, 16.9251957, -32.2431068, 15.8305244, -49.8733482, 49.1682968
4: -32.2432442, 22.7248764, -30.5955238, 21.3737946, -53.6170349, 53.3204002
5: -27.7181053, 21.5514145, -26.2416916, 20.3659039, -48.0840034, 47.7931061
6: -25.3948174, 25.2115002, -23.8606510, 23.7977352, -49.1925507, 49.0721474
7: -28.1485901, 26.6721764, -26.5444374, 25.3830585, -53.5316429, 53.2166138
8: -39.2772369, 18.6460743, -37.3183327, 17.0864391, -56.3636780, 55.9644089
9: -24.8099785, 25.1608829, -23.3388252, 23.6231518, -48.4331284, 48.4996948

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 161

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1409992, upper bound: 43.1398193
time: 8.88 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1338473, upper bound: 43.1350177
time: 7.16 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -21.7038612, 17.3731079, -21.7182274, 17.3838654, -39.0877190, 39.0913353
1: -19.5273743, 15.6165009, -19.4996376, 15.6013842, -35.1287575, 35.1161346
2: -24.6448936, 15.4580469, -24.6385574, 15.4691200, -40.1140137, 40.0966034
3: -26.5152397, 13.2564468, -26.4934959, 13.2756948, -39.7909355, 39.7499428
4: -25.1272545, 17.7283669, -25.1090870, 17.7332897, -42.8605423, 42.8374557
5: -21.5744591, 16.8401279, -21.5647202, 16.8413258, -38.4157829, 38.4048462
6: -19.7277431, 19.6662560, -19.7728977, 19.6606407, -39.3883820, 39.4391479
7: -21.8589344, 20.8877678, -21.8563232, 20.8656693, -42.7246017, 42.7440872
8: -30.8547745, 14.4161091, -30.8046761, 14.4887667, -45.3435402, 45.2207870
9: -19.2638569, 19.5840149, -19.2886238, 19.5903473, -38.8542023, 38.8726349

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1574921, upper bound: 43.1575417
time: 6.11 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1574921, upper bound: 43.1575417
time: 6.23 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -21.0068817, 16.8179626, -27.1860657, 21.7152252, -42.7221031, 44.0040283
1: -18.8960724, 15.1230173, -24.4577141, 19.4102249, -38.3062973, 39.5807304
2: -23.8438721, 14.9697189, -30.8860817, 19.2588673, -43.1027374, 45.8558006
3: -25.6524258, 12.8441973, -33.1571007, 16.4992638, -42.1516838, 46.0012970
4: -24.3171959, 17.1558609, -31.3926735, 22.1173325, -46.4345284, 48.5485344
5: -20.8778877, 16.3058319, -26.9891968, 20.9838791, -41.8617668, 43.2950287
6: -19.0856133, 19.0315094, -24.7466927, 24.5343742, -43.6199875, 43.7782021
7: -21.1400852, 20.2319489, -27.3989487, 25.9918861, -47.1319656, 47.6308899
8: -29.8894424, 13.9451599, -38.2470932, 18.1375275, -48.0269661, 52.1922379
9: -18.6371956, 18.9486618, -24.1517353, 24.4822845, -43.1194801, 43.1003914

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 210

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1449620, upper bound: 43.1436103
time: 9.03 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1413022, upper bound: 43.1415268
time: 7.77 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -24.9341640, 19.9274826, -21.2979813, 17.0487576, -41.9829216, 41.2254524
1: -22.4838905, 17.8939018, -19.1336842, 15.3107500, -37.7946396, 37.0275841
2: -28.3454323, 17.6798630, -24.1668587, 15.1669312, -43.5123558, 41.8467102
3: -30.5372505, 15.1262417, -25.9951591, 13.0145378, -43.5517883, 41.1213951
4: -28.9056511, 20.3189201, -24.6424789, 17.3853550, -46.2910042, 44.9613914
5: -24.8202057, 19.3034248, -21.1586151, 16.5252228, -41.3454285, 40.4620361
6: -22.6574230, 22.5767212, -19.3718967, 19.2862377, -41.9436607, 41.9486160
7: -25.1466045, 23.9747276, -21.4271183, 20.4969845, -45.6435890, 45.4018478
8: -35.3328400, 16.4606743, -30.2603302, 14.1538296, -49.4866714, 46.7210045
9: -22.1244774, 22.4664288, -18.9067459, 19.2028351, -41.3273125, 41.3731766

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 196

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1577081, upper bound: 43.1588368
time: 7.30 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1577081, upper bound: 43.1588368
time: 5.97 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -24.2138195, 19.3546333, -26.6997089, 21.3254318, -45.5392532, 46.0543442
1: -21.8347931, 17.3879509, -24.0449448, 19.0699654, -40.9047585, 41.4328957
2: -27.5215034, 17.1759491, -30.3420677, 18.9088898, -46.4303932, 47.5180130
3: -29.6486187, 14.7020111, -32.5881577, 16.1946564, -45.8432732, 47.2901688
4: -28.0732269, 19.7303581, -30.8643646, 21.7137661, -49.7869911, 50.5947189
5: -24.1014748, 18.7528439, -26.5209045, 20.6223755, -44.7238464, 45.2737503
6: -21.9934044, 21.9241676, -24.2843971, 24.1059074, -46.0993118, 46.2085609
7: -24.4084358, 23.3007526, -26.9155331, 25.5737267, -49.9821548, 50.2162743
8: -34.3411064, 15.9729719, -37.6258392, 17.7476177, -52.0887222, 53.5988045
9: -21.4796066, 21.8124199, -23.7152672, 24.0362663, -45.5158730, 45.5276871

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1572100, upper bound: 43.1574639
time: 8.09 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1572100, upper bound: 43.1574636
time: 6.71 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -21.7195969, 17.3864822, -25.0382080, 19.9666920, -41.6862869, 42.4246902
1: -19.5034771, 15.6070557, -22.5253639, 17.9201698, -37.4236450, 38.1324158
2: -24.6378365, 15.4754276, -28.4537754, 17.6953850, -42.3332176, 43.9291992
3: -26.4865685, 13.2872143, -30.5958500, 15.1019583, -41.5885277, 43.8830643
4: -25.1001129, 17.7426262, -29.0424080, 20.3569641, -45.4570770, 46.7850304
5: -21.5622501, 16.8408623, -24.9149742, 19.3758545, -40.9381027, 41.7558250
6: -19.7775402, 19.6628361, -22.7123909, 22.6275234, -42.4050598, 42.3752289
7: -21.8593941, 20.8528175, -25.2303772, 24.0958557, -45.9552498, 46.0831947
8: -30.7973328, 14.5224876, -35.4749031, 16.3717785, -47.1691132, 49.9973907
9: -19.2886696, 19.5960045, -22.2112560, 22.4839439, -41.7726135, 41.8072548

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 210

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1457409, upper bound: 43.1436490
time: 8.43 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1427877, upper bound: 43.1417305
time: 9.44 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -21.2774258, 17.0338707, -28.1179867, 22.4046669, -43.6820908, 45.1518555
1: -19.1183128, 15.3009081, -25.3549309, 20.0962009, -39.2145157, 40.6558380
2: -24.1410046, 15.1577320, -31.9842739, 19.8172855, -43.9582863, 47.1419945
3: -25.9615479, 13.0123472, -34.4375916, 16.8887711, -42.8503189, 47.4499359
4: -24.6095200, 17.3766994, -32.6495323, 22.8266106, -47.4361305, 50.0262222
5: -21.1348934, 16.5083351, -28.0123405, 21.7274818, -42.8623695, 44.5206757
6: -19.3566265, 19.2684631, -25.5089626, 25.4051495, -44.7617760, 44.7774200
7: -21.4080811, 20.4647503, -28.3622723, 27.0502148, -48.4582901, 48.8270226
8: -30.2238121, 14.1715326, -39.7445335, 18.3234577, -48.5472603, 53.9160614
9: -18.8868256, 19.1889114, -24.9398117, 25.2347279, -44.1215515, 44.1287155

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 210

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1450311, upper bound: 43.1423399
time: 8.05 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1422602, upper bound: 43.1407977
time: 7.12 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -27.2799320, 21.8099499, -24.3366165, 19.4080429, -46.6879730, 46.1465683
1: -24.5545216, 19.4934082, -21.8922882, 17.4264603, -41.9809723, 41.3856964
2: -31.0054207, 19.3472977, -27.6507912, 17.2043571, -48.2097778, 46.9980888
3: -33.2779045, 16.5700607, -29.7290955, 14.6876383, -47.9655418, 46.2991562
4: -31.4943810, 22.2204762, -28.2314034, 19.7834511, -51.2778320, 50.4518738
5: -27.0735741, 21.0658665, -24.2151489, 18.8393478, -45.9129219, 45.2810020
6: -24.8458023, 24.6351814, -22.0648117, 21.9912262, -46.8370285, 46.6999931
7: -27.4951591, 26.0578537, -24.5118542, 23.4391556, -50.9343147, 50.5697060
8: -38.3936996, 18.2769184, -34.5103645, 15.8949738, -54.2886581, 52.7872734
9: -24.2430763, 24.6009483, -21.5826473, 21.8466644, -46.0897408, 46.1835938

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 210

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1444121, upper bound: 43.1429397
time: 7.22 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1403407, upper bound: 43.1407971
time: 6.81 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -26.8044033, 21.4258156, -27.4044991, 21.8371391, -48.6415405, 48.8303146
1: -24.1394386, 19.1599255, -24.7123489, 19.5944328, -43.7338638, 43.8722725
2: -30.4694462, 19.0020618, -31.1687622, 19.3173122, -49.7867584, 50.1708183
3: -32.7104225, 16.2710152, -33.5580597, 16.4670315, -49.1774521, 49.8290749
4: -30.9669762, 21.8229828, -31.8266106, 22.2438049, -53.2107811, 53.6495934
5: -26.6149635, 20.7054348, -27.3014183, 21.1820145, -47.7969780, 48.0068512
6: -24.3904247, 24.2090874, -24.8505726, 24.7587662, -49.1491852, 49.0596504
7: -27.0131378, 25.6428547, -27.6320801, 26.3836746, -53.3968124, 53.2749329
8: -37.7680969, 17.8897305, -38.7647247, 17.8362865, -55.6043854, 56.6544495
9: -23.8110466, 24.1565228, -24.3009472, 24.5880337, -48.3990746, 48.4574699

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 210

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1437519, upper bound: 43.1417729
time: 7.37 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1400550, upper bound: 43.1400550
time: 5.36 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 14.20 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 8, lower bound: -43.1438005, upper bound: 43.1426490
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 8, lower bound: -43.1378680, upper bound: 43.1381622
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 8, lower bound: -43.1434155, upper bound: 43.1420335
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 8, lower bound: -43.1376026, upper bound: 43.1378050
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 8, lower bound: -43.1425270, upper bound: 43.1418076
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 8, lower bound: -43.1343339, upper bound: 43.1359994
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 8, lower bound: -43.1421543, upper bound: 43.1413049
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 8, lower bound: -43.1341723, upper bound: 43.1357489
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 8, lower bound: -43.1427077, upper bound: 43.1413273
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 8, lower bound: -43.1372735, upper bound: 43.1372832
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 8, lower bound: -43.1420324, upper bound: 43.1404044
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 8, lower bound: -43.1368476, upper bound: 43.1367260
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 8, lower bound: -43.1416049, upper bound: 43.1407731
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 8, lower bound: -43.1340917, upper bound: 43.1354146
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 8, lower bound: -43.1409992, upper bound: 43.1398193
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 8, lower bound: -43.1338473, upper bound: 43.1350177
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 8, lower bound: -43.1574921, upper bound: 43.1575417
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 8, lower bound: -43.1574921, upper bound: 43.1575417
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 8, lower bound: -43.1449620, upper bound: 43.1436103
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 8, lower bound: -43.1413022, upper bound: 43.1415268
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 8, lower bound: -43.1577081, upper bound: 43.1588368
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 8, lower bound: -43.1577081, upper bound: 43.1588368
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 8, lower bound: -43.1572100, upper bound: 43.1574639
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 8, lower bound: -43.1572100, upper bound: 43.1574636
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 8, lower bound: -43.1457409, upper bound: 43.1436490
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 8, lower bound: -43.1427877, upper bound: 43.1417305
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 8, lower bound: -43.1450311, upper bound: 43.1423399
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 8, lower bound: -43.1422602, upper bound: 43.1407977
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 8, lower bound: -43.1444121, upper bound: 43.1429397
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 8, lower bound: -43.1403407, upper bound: 43.1407971
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 8, lower bound: -43.1437519, upper bound: 43.1417729
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 8, lower bound: -43.1400550, upper bound: 43.1400550

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -21.6825409, 17.3576088, -20.3826065, 16.3216476, -38.0041885, 37.7402153
1: -19.4594078, 15.5867901, -18.3269596, 14.6761532, -34.1355591, 33.9137421
2: -24.5701447, 15.4438124, -23.1433144, 14.5142899, -39.0844345, 38.5871201
3: -26.3900719, 13.2765951, -24.9004669, 12.4479694, -38.8380356, 38.1770554
4: -25.0076675, 17.7255077, -23.6265450, 16.6342411, -41.6419067, 41.3520432
5: -21.5060825, 16.7976513, -20.2698669, 15.8355122, -37.3415947, 37.0675125
6: -19.7360935, 19.6281929, -18.5016804, 18.4727192, -38.2088089, 38.1298676
7: -21.8061352, 20.7579479, -20.5060749, 19.6809006, -41.4870338, 41.2640228
8: -30.6779327, 14.5811291, -29.0674839, 13.4428358, -44.1207657, 43.6486130
9: -19.2596035, 19.5652466, -18.0869102, 18.3768997, -37.6364975, 37.6521568

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 92

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1438005, upper bound: 43.1426490
time: 8.92 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1438005, upper bound: 43.1426490
time: 8.03 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 18.39 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 18.39
Output dim: 8, lower bound: -43.1438005, upper bound: 43.1426490
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 18.39
Output dim: 8, lower bound: -43.1438005, upper bound: 43.1426490
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.39
Output dim: 8, lower bound: -43.1378680, upper bound: 43.1381622
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.39
Output dim: 8, lower bound: -43.1434155, upper bound: 43.1420335
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.39
Output dim: 8, lower bound: -43.1376026, upper bound: 43.1378050
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.39
Output dim: 8, lower bound: -43.1425270, upper bound: 43.1418076
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.39
Output dim: 8, lower bound: -43.1343339, upper bound: 43.1359994
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.39
Output dim: 8, lower bound: -43.1421543, upper bound: 43.1413049
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.39
Output dim: 8, lower bound: -43.1341723, upper bound: 43.1357489
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.39
Output dim: 8, lower bound: -43.1427077, upper bound: 43.1413273
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.39
Output dim: 8, lower bound: -43.1372735, upper bound: 43.1372832
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.39
Output dim: 8, lower bound: -43.1420324, upper bound: 43.1404044
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.39
Output dim: 8, lower bound: -43.1368476, upper bound: 43.1367260
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.39
Output dim: 8, lower bound: -43.1416049, upper bound: 43.1407731
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.39
Output dim: 8, lower bound: -43.1340917, upper bound: 43.1354146
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.39
Output dim: 8, lower bound: -43.1409992, upper bound: 43.1398193
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.39
Output dim: 8, lower bound: -43.1338473, upper bound: 43.1350177
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.39
Output dim: 8, lower bound: -43.1574921, upper bound: 43.1575417
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.39
Output dim: 8, lower bound: -43.1574921, upper bound: 43.1575417
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.39
Output dim: 8, lower bound: -43.1449620, upper bound: 43.1436103
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.39
Output dim: 8, lower bound: -43.1413022, upper bound: 43.1415268
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.39
Output dim: 8, lower bound: -43.1577081, upper bound: 43.1588368
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.39
Output dim: 8, lower bound: -43.1577081, upper bound: 43.1588368
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.39
Output dim: 8, lower bound: -43.1572100, upper bound: 43.1574639
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.39
Output dim: 8, lower bound: -43.1572100, upper bound: 43.1574636
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.39
Output dim: 8, lower bound: -43.1457409, upper bound: 43.1436490
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.39
Output dim: 8, lower bound: -43.1427877, upper bound: 43.1417305
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.39
Output dim: 8, lower bound: -43.1450311, upper bound: 43.1423399
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.39
Output dim: 8, lower bound: -43.1422602, upper bound: 43.1407977
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.39
Output dim: 8, lower bound: -43.1444121, upper bound: 43.1429397
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.39
Output dim: 8, lower bound: -43.1403407, upper bound: 43.1407971
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.39
Output dim: 8, lower bound: -43.1437519, upper bound: 43.1417729
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.39
Output dim: 8, lower bound: -43.1400550, upper bound: 43.1400550
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=51.530487060546875
rel_dist={8: [-43.17074367834749, 43.17074366907076]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 50

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1703288, upper bound: 43.1703862
time: 7.56 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1705811, upper bound: 43.1705811
time: 21.53 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 29.23 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 29.23
Output dim: 8, lower bound: -43.1703288, upper bound: 43.1703862
IS_A2, status: Status.UNKNOWN, split count: 1, time: 29.23
Output dim: 8, lower bound: -43.1705811, upper bound: 43.1705811

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -24.3706722, 19.5024204, -23.0061188, 18.4261284, -42.7967987, 42.5085373
1: -21.8586521, 17.4819832, -20.6564941, 16.5256901, -38.3843422, 38.1384735
2: -27.6241531, 17.3405914, -26.1055031, 16.3958035, -44.0199432, 43.4460945
3: -29.6621819, 14.9008579, -28.0469666, 14.0826626, -43.7448425, 42.9478226
4: -28.0579205, 19.9350491, -26.5498276, 18.8209209, -46.8788414, 46.4848785
5: -24.1463985, 18.8390179, -22.8218307, 17.8192692, -41.9656677, 41.6608505
6: -22.2339973, 22.0551300, -20.9910202, 20.8380260, -43.0720215, 43.0461502
7: -24.5400143, 23.1987343, -23.1822052, 21.9934196, -46.5334320, 46.3809319
8: -34.2982140, 16.5464706, -32.5316315, 15.5278721, -49.8260880, 49.0781021
9: -21.6667118, 22.0264530, -20.4470005, 20.7980270, -42.4647369, 42.4734535

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 183

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1678289, upper bound: 43.1679773
time: 6.42 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1678023, upper bound: 43.1679241
time: 7.38 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -24.2108173, 19.3813133, -24.3711872, 19.5088329, -43.7196503, 43.7524948
1: -21.7347298, 17.3723087, -21.8782272, 17.4851513, -39.2198792, 39.2505341
2: -27.4703083, 17.2466469, -27.6523323, 17.3596287, -44.8299370, 44.8989792
3: -29.5238304, 14.8024702, -29.7195702, 14.8975945, -44.4214211, 44.5220413
4: -27.9244385, 19.8066959, -28.1079559, 19.9377327, -47.8621712, 47.9146500
5: -24.0113335, 18.7387772, -24.1696205, 18.8616409, -42.8729744, 42.9083977
6: -22.1033802, 21.9187889, -22.2511330, 22.0631180, -44.1664963, 44.1699219
7: -24.3987465, 23.1103039, -24.5613556, 23.2590847, -47.6578293, 47.6716537
8: -34.1606636, 16.3779488, -34.3784752, 16.4904804, -50.6511421, 50.7564240
9: -21.5237293, 21.8889427, -21.6675072, 22.0340652, -43.5577927, 43.5564423

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 50

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1685739, upper bound: 43.1686044
time: 7.22 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1685525, upper bound: 43.1685525
time: 6.00 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 14.65 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 14.65
Output dim: 8, lower bound: -43.1678289, upper bound: 43.1679773
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 14.65
Output dim: 8, lower bound: -43.1678023, upper bound: 43.1679241
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 14.65
Output dim: 8, lower bound: -43.1685739, upper bound: 43.1686044
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 14.65
Output dim: 8, lower bound: -43.1685525, upper bound: 43.1685525

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -23.4324989, 18.7478447, -21.7678070, 17.4289112, -40.8614120, 40.5156517
1: -21.0115376, 16.8051910, -19.5338154, 15.6370087, -36.6485443, 36.3390045
2: -26.5604496, 16.6613979, -24.6989365, 15.5011234, -42.0615692, 41.3603287
3: -28.5227757, 14.3174648, -26.5420990, 13.3128862, -41.8356628, 40.8595619
4: -27.0068150, 19.1482334, -25.1600819, 17.7846947, -44.7915039, 44.3083076
5: -23.2335320, 18.1251392, -21.6096745, 16.8771915, -40.1107254, 39.7348099
6: -21.3578758, 21.2004433, -19.8358192, 19.7118702, -41.0697403, 41.0362625
7: -23.5822353, 22.3691845, -21.9191837, 20.8908615, -44.4730988, 44.2883682
8: -33.0436516, 15.7982969, -30.8634033, 14.5538778, -47.5975266, 46.6617012
9: -20.8275566, 21.1544037, -19.3400898, 19.6467686, -40.4743271, 40.4944878

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 183

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1625339, upper bound: 43.1622585
time: 8.52 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1619248, upper bound: 43.1620265
time: 7.39 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -22.8788910, 18.3015556, -25.5591698, 20.3891373, -43.2680283, 43.8607140
1: -20.5118790, 16.4068222, -22.9426403, 18.2711678, -38.7830467, 39.3494644
2: -25.9304543, 16.2622032, -29.0153389, 18.0738411, -44.0042915, 45.2775421
3: -27.8463936, 13.9737663, -31.1640491, 15.4553337, -43.3017197, 45.1378174
4: -26.3842678, 18.6879807, -29.5665379, 20.8030090, -47.1872635, 48.2545166
5: -22.6903286, 17.7045135, -25.3966846, 19.7524414, -42.4427719, 43.1011963
6: -20.8420830, 20.6952591, -23.2516727, 23.0856915, -43.9277725, 43.9469299
7: -23.0184383, 21.8744087, -25.7480621, 24.4886665, -47.5071030, 47.6224709
8: -32.2934685, 15.3708191, -36.0749512, 16.8949547, -49.1884232, 51.4457703
9: -20.3347740, 20.6337624, -22.6879692, 22.9714622, -43.3062210, 43.3217316

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 216

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1624171, upper bound: 43.1620830
time: 8.33 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1618654, upper bound: 43.1619014
time: 6.85 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -23.2540951, 18.6128654, -23.0501709, 18.4470749, -41.7011681, 41.6630363
1: -20.8714848, 16.6844940, -20.6853027, 16.5354042, -37.4068909, 37.3697968
2: -26.3834000, 16.5564442, -26.1513977, 16.4067364, -42.7901382, 42.7078362
3: -28.3614655, 14.2116938, -28.1151276, 14.0804310, -42.4418945, 42.3268204
4: -26.8489132, 19.0056896, -26.6232834, 18.8321495, -45.6810608, 45.6289711
5: -23.0786915, 18.0097523, -22.8799477, 17.8541889, -40.9328804, 40.8896980
6: -21.2119465, 21.0494518, -21.0200996, 20.8630581, -42.0750046, 42.0695496
7: -23.4211235, 22.2587681, -23.2123718, 22.0814648, -45.5025864, 45.4711380
8: -32.8750076, 15.6254930, -32.6000290, 15.4533205, -48.3283272, 48.2255173
9: -20.6681919, 21.0028000, -20.4865265, 20.8088264, -41.4770203, 41.4893265

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 50

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1626963, upper bound: 43.1624067
time: 9.64 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1620921, upper bound: 43.1621663
time: 7.76 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -22.6952019, 18.1626167, -26.9407997, 21.4820557, -44.1772575, 45.1034164
1: -20.3647499, 16.2818794, -24.1794300, 19.2393761, -39.6041260, 40.4613113
2: -25.7464561, 16.1534328, -30.5800495, 19.0471840, -44.7936401, 46.7334824
3: -27.6757622, 13.8653297, -32.8540993, 16.2778740, -43.9536362, 46.7194214
4: -26.2189007, 18.5397663, -31.1409283, 21.9280357, -48.1469345, 49.6806908
5: -22.5294628, 17.5840168, -26.7619076, 20.8035507, -43.3330154, 44.3459244
6: -20.6911221, 20.5398159, -24.5242462, 24.3232803, -45.0144043, 45.0640640
7: -22.8503208, 21.7579746, -27.1442833, 25.7704067, -48.6207275, 48.9022522
8: -32.1159019, 15.1968517, -37.9412003, 17.8575268, -49.9734268, 53.1380501
9: -20.1702003, 20.4783592, -23.9218655, 24.2154770, -44.3856773, 44.4002151

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1625860, upper bound: 43.1622318
time: 8.64 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1620377, upper bound: 43.1620377
time: 7.72 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 17.84 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 17.84
Output dim: 8, lower bound: -43.1625339, upper bound: 43.1622585
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 17.84
Output dim: 8, lower bound: -43.1619248, upper bound: 43.1620265
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 17.84
Output dim: 8, lower bound: -43.1624171, upper bound: 43.1620830
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 17.84
Output dim: 8, lower bound: -43.1618654, upper bound: 43.1619014
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 17.84
Output dim: 8, lower bound: -43.1626963, upper bound: 43.1624067
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 17.84
Output dim: 8, lower bound: -43.1620921, upper bound: 43.1621663
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 17.84
Output dim: 8, lower bound: -43.1625860, upper bound: 43.1622318
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 17.84
Output dim: 8, lower bound: -43.1620377, upper bound: 43.1620377

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -22.4373627, 17.9564323, -21.1553040, 16.9415131, -39.3788757, 39.1117363
1: -20.1149998, 16.1045589, -18.9799194, 15.2048817, -35.3198814, 35.0844803
2: -25.4197826, 15.9674072, -23.9964981, 15.0738392, -40.4936180, 39.9639015
3: -27.2943649, 13.7307081, -25.7836685, 12.9520321, -40.2463989, 39.5143776
4: -25.8562298, 18.3362656, -24.4486179, 17.2832031, -43.1394272, 42.7848816
5: -22.2396259, 17.3649178, -20.9972095, 16.4081783, -38.6478043, 38.3621254
6: -20.4415398, 20.2988186, -19.2712002, 19.1567440, -39.5982819, 39.5700188
7: -22.5658226, 21.4361095, -21.2903652, 20.3147430, -42.8805656, 42.7264633
8: -31.6719303, 15.1268787, -30.0170860, 14.1433935, -45.8153191, 45.1439667
9: -19.9375076, 20.2487183, -18.7904320, 19.0898476, -39.0273552, 39.0391502

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1572148, upper bound: 43.1570184
time: 7.29 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1571440, upper bound: 43.1568875
time: 7.72 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -29.0057335, 23.1738358, -20.3759079, 16.3201275, -45.3258591, 43.5497437
1: -26.0895195, 20.7024097, -18.2724609, 14.6523600, -40.7418785, 38.9748688
2: -32.9483109, 20.5367756, -23.0993614, 14.5288353, -47.4771461, 43.6361351
3: -35.3139572, 17.6004028, -24.8204670, 12.4890919, -47.8030472, 42.4208603
4: -33.4297676, 23.6182785, -23.5416985, 16.6426163, -50.0723839, 47.1599770
5: -28.7662487, 22.3630428, -20.2181206, 15.8089905, -44.5752411, 42.5811615
6: -26.4272728, 26.1685982, -18.5512238, 18.4455414, -44.8728027, 44.7198219
7: -29.2237797, 27.6214085, -20.4853191, 19.5810947, -48.8048744, 48.1067276
8: -40.6963844, 19.5018921, -28.9382381, 13.6146069, -54.3109894, 48.4401321
9: -25.7784729, 26.1688881, -18.0879059, 18.3775654, -44.1560364, 44.2567902

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 210

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1565506, upper bound: 43.1567271
time: 7.75 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1565044, upper bound: 43.1566587
time: 8.15 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -21.8947811, 17.5180225, -24.9353542, 19.8934975, -41.7882767, 42.4533730
1: -19.6242008, 15.7135620, -22.3812027, 17.8322353, -37.4564323, 38.0947571
2: -24.8007812, 15.5757475, -28.3022118, 17.6381950, -42.4389648, 43.8779602
3: -26.6301403, 13.3944645, -30.3931293, 15.0880499, -41.7181892, 43.7875900
4: -25.2455254, 17.8841324, -28.8463097, 20.2934647, -45.5389900, 46.7304420
5: -21.7071896, 16.9516830, -24.7740288, 19.2762680, -40.9834595, 41.7257118
6: -19.9356194, 19.8035202, -22.6766357, 22.5217934, -42.4574127, 42.4801559
7: -22.0115376, 20.9510651, -25.1104164, 23.9045620, -45.9160995, 46.0614815
8: -30.9349442, 14.7075577, -35.2168503, 16.4747372, -47.4096756, 49.9244080
9: -19.4541702, 19.7377510, -22.1301575, 22.4062939, -41.8604622, 41.8679085

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1570947, upper bound: 43.1568668
time: 8.83 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1570030, upper bound: 43.1566907
time: 38.02 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -28.4480133, 22.7254543, -24.1127892, 19.2395210, -47.6875343, 46.8382416
1: -25.5908108, 20.2993469, -21.6423759, 17.2543888, -42.8451996, 41.9417191
2: -32.3166962, 20.1335602, -27.3633041, 17.0615253, -49.3782196, 47.4968643
3: -34.6350403, 17.2495003, -29.3814583, 14.6023512, -49.2373886, 46.6309586
4: -32.8059044, 23.1571026, -27.8993549, 19.6206741, -52.4265785, 51.0564499
5: -28.2188244, 21.9386444, -23.9541149, 18.6493073, -46.8681335, 45.8927612
6: -25.9009304, 25.6589489, -21.9178543, 21.7766266, -47.6775436, 47.5767937
7: -28.6578827, 27.1250782, -24.2698689, 23.1376419, -51.7955246, 51.3949432
8: -39.9446678, 19.0682888, -34.0893936, 15.9130173, -55.8576851, 53.1576729
9: -25.2843513, 25.6392059, -21.3942490, 21.6597404, -46.9440918, 47.0334549

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 92

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1565076, upper bound: 43.1566336
time: 6.45 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1564486, upper bound: 43.1565266
time: 7.48 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -22.3365746, 17.8822231, -22.4412136, 17.9622765, -40.2988510, 40.3234253
1: -20.0442162, 16.0373669, -20.1369591, 16.1064167, -36.1506348, 36.1743240
2: -25.3314190, 15.9155712, -25.4534721, 15.9815645, -41.3129845, 41.3690414
3: -27.2272797, 13.6689014, -27.3624954, 13.7206059, -40.9478836, 41.0313950
4: -25.7878551, 18.2547188, -25.9196930, 18.3340836, -44.1219330, 44.1744118
5: -22.1624317, 17.3074856, -22.2720108, 17.3883438, -39.5507736, 39.5794945
6: -20.3665619, 20.2175121, -20.4589081, 20.3113136, -40.6778717, 40.6764221
7: -22.4830360, 21.3988514, -22.5899944, 21.5113468, -43.9943810, 43.9888458
8: -31.6083488, 15.0042038, -31.7604046, 15.0409889, -46.6493378, 46.7646103
9: -19.8468990, 20.1665344, -19.9416847, 20.2541542, -40.1010513, 40.1082191

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1575902, upper bound: 43.1573273
time: 8.26 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1576113, upper bound: 43.1572954
time: 7.01 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -27.8802986, 22.2901020, -21.6053753, 17.2959900, -45.1762886, 43.8954697
1: -25.0744896, 19.9117870, -19.3807602, 15.5161896, -40.5906754, 39.2925491
2: -31.6715298, 19.7712097, -24.4955215, 15.3951797, -47.0667114, 44.2667313
3: -33.9855881, 16.9441643, -26.3338699, 13.2248316, -47.2104187, 43.2780304
4: -32.1554108, 22.7173023, -24.9518318, 17.6465797, -49.8019829, 47.6691360
5: -27.6602707, 21.5175171, -21.4370270, 16.7486954, -44.4089584, 42.9545403
6: -25.4197578, 25.1660347, -19.6890144, 19.5493660, -44.9691238, 44.8550491
7: -28.0901108, 26.5903931, -21.7337990, 20.7272053, -48.8173027, 48.3241920
8: -39.1871643, 18.7456398, -30.6060162, 14.4706621, -53.6578255, 49.3516541
9: -24.7796288, 25.1610832, -19.1910725, 19.4906998, -44.2703209, 44.3521576

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 210

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1569437, upper bound: 43.1570547
time: 7.49 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1569175, upper bound: 43.1570310
time: 6.40 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -21.7936783, 17.4456120, -26.3268185, 20.9957504, -42.7894287, 43.7724304
1: -19.5506573, 15.6466513, -23.6289978, 18.8069420, -38.3575974, 39.2756500
2: -24.7137737, 15.5241823, -29.8793297, 18.6180687, -43.3318405, 45.4035110
3: -26.5608940, 13.3325958, -32.0986710, 15.9152422, -42.4761314, 45.4312592
4: -25.1744957, 17.8015690, -30.4349174, 21.4278164, -46.6023102, 48.2364883
5: -21.6281223, 16.8947525, -26.1516933, 20.3339024, -41.9620209, 43.0464401
6: -19.8608246, 19.7221756, -23.9583473, 23.7684631, -43.6292877, 43.6805229
7: -21.9272747, 20.9115715, -26.5167179, 25.1987476, -47.1260223, 47.4282913
8: -30.8725853, 14.5909824, -37.1017914, 17.4388256, -48.3114052, 51.6927719
9: -19.3626060, 19.6588192, -23.3719730, 23.6603851, -43.0229912, 43.0307922

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 216

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1574747, upper bound: 43.1571605
time: 8.66 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1574692, upper bound: 43.1570566
time: 8.80 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -27.3030434, 21.8108540, -25.4683380, 20.3124466, -47.6154861, 47.2791901
1: -24.5530186, 19.4854813, -22.8564339, 18.2029896, -42.7560081, 42.3419113
2: -31.0028419, 19.3472672, -28.8987961, 18.0158825, -49.0187225, 48.2460632
3: -33.2705040, 16.5761280, -31.0417557, 15.4074736, -48.6779785, 47.6178818
4: -31.4997005, 22.2335949, -29.4468079, 20.7249699, -52.2246704, 51.6804047
5: -27.0968056, 21.0656910, -25.2960396, 19.6789436, -46.7757454, 46.3617249
6: -24.8762207, 24.6357422, -23.1663170, 22.9902611, -47.8664780, 47.8020554
7: -27.5008163, 26.0701790, -25.6388626, 24.3980312, -51.8988419, 51.7090416
8: -38.3622475, 18.2748795, -35.9241905, 16.8514023, -55.2136345, 54.1990662
9: -24.2677650, 24.5921249, -22.6032906, 22.8804340, -47.1481972, 47.1954155

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1568926, upper bound: 43.1569412
time: 7.38 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1568606, upper bound: 43.1568606
time: 13.96 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 22.96 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.96
Output dim: 8, lower bound: -43.1572148, upper bound: 43.1570184
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.96
Output dim: 8, lower bound: -43.1571440, upper bound: 43.1568875
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.96
Output dim: 8, lower bound: -43.1565506, upper bound: 43.1567271
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.96
Output dim: 8, lower bound: -43.1565044, upper bound: 43.1566587
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.96
Output dim: 8, lower bound: -43.1570947, upper bound: 43.1568668
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.96
Output dim: 8, lower bound: -43.1570030, upper bound: 43.1566907
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.96
Output dim: 8, lower bound: -43.1565076, upper bound: 43.1566336
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.96
Output dim: 8, lower bound: -43.1564486, upper bound: 43.1565266
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.96
Output dim: 8, lower bound: -43.1575902, upper bound: 43.1573273
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.96
Output dim: 8, lower bound: -43.1576113, upper bound: 43.1572954
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.96
Output dim: 8, lower bound: -43.1569437, upper bound: 43.1570547
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.96
Output dim: 8, lower bound: -43.1569175, upper bound: 43.1570310
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.96
Output dim: 8, lower bound: -43.1574747, upper bound: 43.1571605
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.96
Output dim: 8, lower bound: -43.1574692, upper bound: 43.1570566
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.96
Output dim: 8, lower bound: -43.1568926, upper bound: 43.1569412
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.96
Output dim: 8, lower bound: -43.1568606, upper bound: 43.1568606

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -20.9385109, 16.7534962, -19.2884407, 15.4569855, -36.3954964, 36.0419388
1: -18.8002872, 15.0550070, -17.3319244, 13.9035053, -32.7037926, 32.3869324
2: -23.7299652, 14.8994236, -21.8899632, 13.7473850, -37.4773483, 36.7893867
3: -25.4988518, 12.7991333, -23.5482960, 11.7959118, -37.2947540, 36.3474274
4: -24.1853848, 17.0949631, -22.3559837, 15.7436562, -39.9290390, 39.4509468
5: -20.7836571, 16.2280369, -19.1810665, 14.9955521, -35.7792091, 35.4090958
6: -19.0142288, 18.9536572, -17.4915524, 17.4832363, -36.4974670, 36.4452057
7: -21.0453682, 20.1070328, -19.3828220, 18.6484489, -39.6938171, 39.4898529
8: -29.7018719, 13.9528494, -27.5620270, 12.7077274, -42.4095955, 41.5148735
9: -18.5803604, 18.8674831, -17.1087646, 17.3853569, -35.9657097, 35.9762459

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 210

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1376267, upper bound: 43.1372817
time: 7.91 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1350295, upper bound: 43.1350303
time: 8.50 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -20.6213284, 16.4994392, -22.2822495, 17.8211308, -38.4424553, 38.7816887
1: -18.5282555, 14.8377247, -20.1016273, 16.0299568, -34.5582123, 34.9393539
2: -23.3730965, 14.6676483, -25.3396187, 15.8023434, -39.1754379, 40.0072594
3: -25.1295547, 12.6014709, -27.2935886, 13.5370169, -38.6665726, 39.8950577
4: -23.8372459, 16.8292656, -25.8866520, 18.1468601, -41.9840927, 42.7159081
5: -20.4796219, 15.9914932, -22.2019768, 17.2928867, -37.7725067, 38.1934700
6: -18.7098103, 18.6729164, -20.2041283, 20.1949959, -38.9048080, 38.8770447
7: -20.7183285, 19.8337746, -22.4473915, 21.5371246, -42.2554359, 42.2811661
8: -29.2970886, 13.6867943, -31.7536011, 14.5707684, -43.8678513, 45.4403954
9: -18.2886372, 18.5708370, -19.7677116, 20.0635185, -38.3521576, 38.3385468

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 210

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1373885, upper bound: 43.1369987
time: 10.75 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1349083, upper bound: 43.1349156
time: 8.70 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -27.4847107, 21.9540825, -18.5779076, 14.8930254, -42.3777351, 40.5319901
1: -24.7573528, 19.6353111, -16.6845055, 13.3983135, -38.1556664, 36.3198166
2: -31.2429218, 19.4508743, -21.0705986, 13.2517185, -44.4946327, 40.5214729
3: -33.4967079, 16.6546478, -22.6695271, 11.3737841, -44.8704910, 39.3241730
4: -31.7403202, 22.3546715, -21.5224152, 15.1606236, -46.9009399, 43.8770790
5: -27.2888126, 21.2134895, -18.4704742, 14.4493275, -41.7381325, 39.6839600
6: -24.9731960, 24.8088112, -16.8371048, 16.8345108, -41.8076973, 41.6459122
7: -27.6969604, 26.2780724, -18.6466675, 17.9720783, -45.6690369, 44.9247398
8: -38.6911011, 18.3042812, -26.5735893, 12.2392502, -50.9303513, 44.8778610
9: -24.4107056, 24.7577209, -16.4696846, 16.7373791, -41.1480865, 41.2274017

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 161

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1371900, upper bound: 43.1369359
time: 9.48 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1335142, upper bound: 43.1341057
time: 8.27 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -27.1151447, 21.6524124, -21.5421410, 17.2341366, -44.3492737, 43.1945496
1: -24.4404144, 19.3775253, -19.4264183, 15.5048542, -39.9452667, 38.8039436
2: -30.8276157, 19.1789265, -24.4868584, 15.2840910, -46.1117058, 43.6657829
3: -33.0637817, 16.4166698, -26.3795319, 13.0980463, -46.1618195, 42.7962036
4: -31.3362255, 22.0443439, -25.0221596, 17.5386028, -48.8748245, 47.0665054
5: -26.9335232, 20.9339371, -21.4629784, 16.7230034, -43.6565247, 42.3969078
6: -24.6154442, 24.4784851, -19.5218010, 19.5203552, -44.1357956, 44.0002747
7: -27.3217773, 25.9614162, -21.6796017, 20.8359833, -48.1577606, 47.6410179
8: -38.2093124, 17.9863739, -30.7246761, 14.0770979, -52.2864037, 48.7110443
9: -24.0713787, 24.4048023, -19.1036339, 19.3884315, -43.4598083, 43.5084267

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 161

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1369522, upper bound: 43.1366852
time: 9.58 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1334635, upper bound: 43.1340104
time: 6.93 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -20.4571266, 16.3687191, -22.9698372, 18.3267612, -38.7838783, 39.3385544
1: -18.3620567, 14.7091866, -20.6590042, 16.4673290, -34.8293839, 35.3681908
2: -23.1818466, 14.5538158, -26.0995560, 16.2432785, -39.4251175, 40.6533737
3: -24.9100056, 12.5028563, -28.0456352, 13.8765259, -38.7865257, 40.5484886
4: -23.6399460, 16.6946430, -26.6585121, 18.6725998, -42.3125458, 43.3531532
5: -20.3101521, 15.8629084, -22.8616085, 17.7959538, -38.1061020, 38.7245140
6: -18.5664749, 18.5153580, -20.8064747, 20.7670555, -39.3335304, 39.3218307
7: -20.5497665, 19.6728745, -23.1263847, 22.1587029, -42.7084579, 42.7992592
8: -29.0523014, 13.5928211, -32.6524239, 14.9605951, -44.0128937, 46.2452431
9: -18.1516857, 18.4196472, -20.3627453, 20.6167984, -38.7684822, 38.7823906

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 210

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1370953, upper bound: 43.1366516
time: 8.53 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1347243, upper bound: 43.1346790
time: 6.44 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -20.0987244, 16.0824757, -25.8672123, 20.6150055, -40.7137299, 41.9496841
1: -18.0534935, 14.4628878, -23.3257942, 18.5181007, -36.5715942, 37.7886734
2: -22.7784252, 14.2934475, -29.4221268, 18.2324657, -41.0108795, 43.7155724
3: -24.4919205, 12.2792492, -31.6689701, 15.5524387, -40.0443573, 43.9482193
4: -23.2452202, 16.3954506, -30.0626869, 20.9955559, -44.2407761, 46.4581375
5: -19.9661980, 15.5949078, -25.7811203, 20.0108356, -39.9770355, 41.3760300
6: -18.2247314, 18.1976376, -23.4309959, 23.3822174, -41.6069450, 41.6286316
7: -20.1808853, 19.3619995, -26.0746136, 24.9482193, -45.1291046, 45.4366150
8: -28.5921078, 13.2972460, -36.6884499, 16.7645798, -45.3566818, 49.9856949
9: -17.8235626, 18.0868988, -22.9243851, 23.2044678, -41.0280228, 41.0112839

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 210

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1367294, upper bound: 43.1362965
time: 9.09 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1345184, upper bound: 43.1344619
time: 8.16 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -26.9656067, 21.5358067, -22.2111454, 17.7252769, -44.6908836, 43.7469521
1: -24.2898006, 19.2595997, -19.9695168, 15.9307384, -40.2205391, 39.2291183
2: -30.6528397, 19.0757351, -25.2286453, 15.7125225, -46.3653641, 44.3043785
3: -32.8611488, 16.3295555, -27.1088772, 13.4263029, -46.2874527, 43.4384308
4: -31.1561050, 21.9237404, -25.7761745, 18.0503120, -49.2064171, 47.6999092
5: -26.7792530, 20.8177891, -22.1025581, 17.2143135, -43.9935608, 42.9203491
6: -24.4851494, 24.3342381, -20.1061935, 20.0766087, -44.5617599, 44.4404259
7: -27.1679306, 25.8135643, -22.3426590, 21.4427643, -48.6106911, 48.1562233
8: -37.9868393, 17.9036350, -31.6036797, 14.4544458, -52.4412842, 49.5073166
9: -23.9493065, 24.2659359, -19.6843262, 19.9277191, -43.8770218, 43.9502487

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 161

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1367467, upper bound: 43.1364344
time: 8.25 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1334306, upper bound: 43.1338911
time: 8.78 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -26.5594215, 21.2043591, -25.0742226, 19.9846649, -46.5440865, 46.2785797
1: -23.9384556, 18.9757042, -22.6107216, 17.9605522, -41.8990097, 41.5864258
2: -30.1948357, 18.7779980, -28.5148315, 17.6783600, -47.8731918, 47.2928200
3: -32.3830185, 16.0688496, -30.6897888, 15.0863924, -47.4694099, 46.7586365
4: -30.7095585, 21.5819092, -29.1458092, 20.3475838, -51.0571365, 50.7277145
5: -26.3878441, 20.5100670, -24.9893036, 19.4034042, -45.7912483, 45.4993591
6: -24.0941277, 23.9706326, -22.6981506, 22.6659031, -46.7600250, 46.6687851
7: -26.7551556, 25.4626198, -25.2600002, 24.2053776, -50.9605331, 50.7226105
8: -37.4532928, 17.5587749, -35.5982895, 16.2323494, -53.6856422, 53.1570587
9: -23.5763607, 23.8800926, -22.2180901, 22.4860001, -46.0623627, 46.0981827

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 161

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1364084, upper bound: 43.1360587
time: 12.74 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1333497, upper bound: 43.1337577
time: 7.09 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -20.8911819, 16.7250900, -20.4989548, 16.4116974, -37.3028793, 37.2240448
1: -18.7750244, 15.0302296, -18.4293938, 14.7548409, -33.5298653, 33.4596252
2: -23.7063541, 14.8855658, -23.2694740, 14.5995665, -38.3059196, 38.1550369
3: -25.4984226, 12.7737970, -25.0398445, 12.5203619, -38.0187836, 37.8136406
4: -24.1768036, 17.0566521, -23.7519989, 16.7268105, -40.9036140, 40.8086472
5: -20.7567253, 16.2151470, -20.3809776, 15.9225521, -36.6792755, 36.5961227
6: -18.9894600, 18.9214058, -18.6107540, 18.5709934, -37.5604515, 37.5321579
7: -21.0172691, 20.1177025, -20.6177540, 19.7853794, -40.8026428, 40.7354584
8: -29.7119675, 13.8726616, -29.2128525, 13.5338745, -43.2458420, 43.0855141
9: -18.5388393, 18.8396072, -18.1888409, 18.4783554, -37.0171928, 37.0284500

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 210

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1424932, upper bound: 43.1418860
time: 8.20 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1412590, upper bound: 43.1410039
time: 8.41 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -20.5555038, 16.4580097, -23.7120285, 18.9526176, -39.5081177, 40.1700325
1: -18.4834175, 14.7985191, -21.3802109, 17.0260181, -35.5094376, 36.1787224
2: -23.3271904, 14.6413736, -26.9580059, 16.8103867, -40.1375771, 41.5993805
3: -25.1025391, 12.5649395, -29.0447521, 14.3808756, -39.4834137, 41.6096916
4: -23.8048134, 16.7774372, -27.5202560, 19.3073006, -43.1121140, 44.2976875
5: -20.4331932, 15.9631004, -23.6130981, 18.3742447, -38.8074379, 39.5761948
6: -18.6701050, 18.6235180, -21.5232162, 21.4708138, -40.1409187, 40.1467323
7: -20.6711922, 19.8247681, -23.8965397, 22.8649082, -43.5361023, 43.7213058
8: -29.2789021, 13.6031246, -33.6806221, 15.5602818, -44.8391800, 47.2837448
9: -18.2360649, 18.5295525, -21.0335579, 21.3474007, -39.5834656, 39.5631104

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 210

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1422699, upper bound: 43.1415437
time: 7.74 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1411240, upper bound: 43.1407889
time: 30.35 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -26.3834686, 21.0862026, -19.7132206, 15.7877388, -42.1712074, 40.7994194
1: -23.7632465, 18.8619556, -17.7136154, 14.1977835, -37.9610214, 36.5755692
2: -29.9918938, 18.7000332, -22.3639526, 14.0494032, -44.0412979, 41.0639763
3: -32.1986847, 16.0136642, -24.0690727, 12.0532112, -44.2518921, 40.0827370
4: -30.4927731, 21.4700146, -22.8347912, 16.0825958, -46.5753593, 44.3048058
5: -26.2071037, 20.3855133, -19.5954552, 15.3181019, -41.5251961, 39.9809685
6: -23.9913540, 23.8275452, -17.8846836, 17.8538570, -41.8451996, 41.7122269
7: -26.5873928, 25.2698135, -19.8033276, 19.0418739, -45.6292648, 45.0731430
8: -37.2083855, 17.5587234, -28.1224117, 13.0098190, -50.2182007, 45.6811333
9: -23.4320335, 23.7714539, -17.4839134, 17.7627144, -41.1947403, 41.2553596

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 210

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1419345, upper bound: 43.1415274
time: 7.95 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1401862, upper bound: 43.1404834
time: 13.70 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -26.0077477, 20.7811794, -22.8571510, 18.2763233, -44.2840729, 43.6383286
1: -23.4377518, 18.5996170, -20.6120300, 16.4231453, -39.8608971, 39.2116470
2: -29.5681572, 18.4244041, -25.9800758, 16.2114925, -45.7796478, 44.4044800
3: -31.7549934, 15.7736578, -27.9955730, 13.8783150, -45.6333084, 43.7692184
4: -30.0787258, 21.1548100, -26.5333481, 18.6082878, -48.6870117, 47.6881561
5: -25.8447227, 20.1001129, -22.7629509, 17.7218742, -43.5665970, 42.8630562
6: -23.6298294, 23.4914742, -20.7368565, 20.6978149, -44.3276405, 44.2283211
7: -26.2048531, 24.9445953, -23.0191536, 22.0655327, -48.2703857, 47.9637489
8: -36.7177277, 17.2431641, -32.5067673, 14.9834137, -51.7011414, 49.7499275
9: -23.0874825, 23.4165115, -20.2702904, 20.5725460, -43.6600227, 43.6867981

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 210

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1417285, upper bound: 43.1412264
time: 8.13 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1401234, upper bound: 43.1403541
time: 8.05 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -20.3893776, 16.3254528, -24.3158264, 19.3913956, -39.7807693, 40.6412811
1: -18.3160515, 14.6694164, -21.8710670, 17.4112606, -35.7273102, 36.5404816
2: -23.1351776, 14.5250492, -27.6290722, 17.1889896, -40.3241653, 42.1541214
3: -24.8817730, 12.4636784, -29.7013092, 14.6743279, -39.5560913, 42.1649818
4: -23.6066856, 16.6402168, -28.2070923, 19.7677422, -43.3744278, 44.8473091
5: -20.2617760, 15.8346357, -24.1948814, 18.8235474, -39.0853195, 40.0295181
6: -18.5222092, 18.4640656, -22.0455093, 21.9735336, -40.4957428, 40.5095749
7: -20.4987335, 19.6643524, -24.4944096, 23.4184551, -43.9171906, 44.1587601
8: -29.0337009, 13.5015640, -34.4860611, 15.8791466, -44.9128418, 47.9876175
9: -18.0942192, 18.3756180, -21.5629787, 21.8292732, -39.9234924, 39.9385948

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 210

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1420757, upper bound: 43.1413913
time: 8.99 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1409913, upper bound: 43.1406275
time: 15.36 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -20.0373745, 16.0463619, -27.3808365, 21.8171120, -41.8544846, 43.4272003
1: -18.0097752, 14.4266663, -24.6857166, 19.5756721, -37.5854492, 39.1123810
2: -22.7381363, 14.2695541, -31.1448288, 19.2980289, -42.0361633, 45.4143829
3: -24.4666138, 12.2435865, -33.5236092, 16.4495659, -40.9161758, 45.7671928
4: -23.2160950, 16.3483887, -31.7984505, 22.2253666, -45.4414597, 48.1468391
5: -19.9234390, 15.5704899, -27.2778587, 21.1636467, -41.0870819, 42.8483429
6: -18.1873837, 18.1513939, -24.8273697, 24.7366104, -42.9239883, 42.9787598
7: -20.1351681, 19.3569603, -27.6141796, 26.3594456, -46.4946060, 46.9711380
8: -28.5795822, 13.2213593, -38.7405663, 17.8141747, -46.3937492, 51.9619255
9: -17.7776642, 18.0518265, -24.2767563, 24.5674400, -42.3451042, 42.3285751

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 92

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1417236, upper bound: 43.1407818
time: 9.20 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1407640, upper bound: 43.1402535
time: 7.71 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 18.37 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.37
Output dim: 8, lower bound: -43.1376267, upper bound: 43.1372817
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.37
Output dim: 8, lower bound: -43.1350295, upper bound: 43.1350303
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.37
Output dim: 8, lower bound: -43.1373885, upper bound: 43.1369987
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.37
Output dim: 8, lower bound: -43.1349083, upper bound: 43.1349156
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.37
Output dim: 8, lower bound: -43.1371900, upper bound: 43.1369359
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.37
Output dim: 8, lower bound: -43.1335142, upper bound: 43.1341057
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.37
Output dim: 8, lower bound: -43.1369522, upper bound: 43.1366852
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.37
Output dim: 8, lower bound: -43.1334635, upper bound: 43.1340104
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.37
Output dim: 8, lower bound: -43.1370953, upper bound: 43.1366516
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.37
Output dim: 8, lower bound: -43.1347243, upper bound: 43.1346790
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.37
Output dim: 8, lower bound: -43.1367294, upper bound: 43.1362965
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.37
Output dim: 8, lower bound: -43.1345184, upper bound: 43.1344619
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.37
Output dim: 8, lower bound: -43.1367467, upper bound: 43.1364344
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.37
Output dim: 8, lower bound: -43.1334306, upper bound: 43.1338911
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.37
Output dim: 8, lower bound: -43.1364084, upper bound: 43.1360587
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.37
Output dim: 8, lower bound: -43.1333497, upper bound: 43.1337577
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.37
Output dim: 8, lower bound: -43.1424932, upper bound: 43.1418860
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.37
Output dim: 8, lower bound: -43.1412590, upper bound: 43.1410039
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.37
Output dim: 8, lower bound: -43.1422699, upper bound: 43.1415437
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.37
Output dim: 8, lower bound: -43.1411240, upper bound: 43.1407889
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.37
Output dim: 8, lower bound: -43.1419345, upper bound: 43.1415274
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.37
Output dim: 8, lower bound: -43.1401862, upper bound: 43.1404834
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.37
Output dim: 8, lower bound: -43.1417285, upper bound: 43.1412264
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.37
Output dim: 8, lower bound: -43.1401234, upper bound: 43.1403541
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.37
Output dim: 8, lower bound: -43.1420757, upper bound: 43.1413913
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.37
Output dim: 8, lower bound: -43.1409913, upper bound: 43.1406275
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.37
Output dim: 8, lower bound: -43.1417236, upper bound: 43.1407818
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.37
Output dim: 8, lower bound: -43.1407640, upper bound: 43.1402535
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 18.37
Output dim: 8, lower bound: -43.1568926, upper bound: 43.1569412
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 18.37
Output dim: 8, lower bound: -43.1568606, upper bound: 43.1568606
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=51.530487060546875
rel_dist={8: [-43.1705810903977, 43.17058109163787]}

## Binary Search with IS_dual_ind Result
status: None
Maximum delta epsilon: None
execution time: 1828.57 seconds
