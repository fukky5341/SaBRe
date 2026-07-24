## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2000 seconds
Threshold: 10.310653145
Search space: {k/256 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540)
1: (-4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358)
2: (-5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447)
3: (-6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520)
4: (-6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002)
5: (-5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936)
6: (-4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792)
7: (-5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823)
8: (-7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416)
9: (-4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449)

## BASE Result
execution time: IAR + LP analysis = 1.37 + 5.25 = 6.62 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -10.8533233, upper bound: 10.8533231


# Binary Search by BASE starts (time budget: 1993.38 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=12.009641647338867
rel_dist={8: [-10.853321756160986, 10.853321760348969]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=12.009641647338867
rel_dist={8: [-10.85331894689149, 10.853318913028158]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=12.009641647338867
rel_dist={8: [-10.853314892972527, 10.853314942404314]}

## Binary Search Result
Binary search time: 20.94 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 1972.44 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 92

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8517841, upper bound: 10.8514320
time: 2.70 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8518160, upper bound: 10.8518162
time: 5.21 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 8.06 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 8.06
Output dim: 8, lower bound: -10.8517841, upper bound: 10.8514320
IS_A2, status: Status.UNKNOWN, split count: 1, time: 8.06
Output dim: 8, lower bound: -10.8518160, upper bound: 10.8518162

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -2.9945498, 2.4854207, -5.1155667, 4.2306838, -7.2252336, 7.6009874
1: -2.4506655, 2.4260209, -4.4719973, 3.9877017, -6.4383669, 6.8980169
2: -3.2870402, 2.6394088, -5.8428183, 4.1089778, -7.3960166, 8.4822273
3: -3.5422220, 2.3601861, -6.3578134, 3.6902223, -7.2324443, 8.7179995
4: -3.3720222, 2.6054800, -5.9891391, 4.3749437, -7.7469659, 8.5946188
5: -2.8525395, 2.5022659, -5.0547872, 4.1931419, -7.0456815, 7.5570531
6: -2.8254833, 2.7629449, -4.8366127, 4.7296124, -7.5550957, 7.5995579
7: -2.9726825, 2.8755846, -5.1773725, 5.0368652, -8.0095482, 8.0529556
8: -4.6693630, 3.0776982, -7.8759146, 4.0384431, -8.7078056, 10.9536133
9: -2.5680308, 2.8494966, -4.5277948, 4.8068671, -7.3748970, 7.3772917

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8513517, upper bound: 10.8513518
time: 6.26 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8513517, upper bound: 10.8513594
time: 4.23 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -3.6680012, 3.0478530, -4.8507833, 4.0169296, -7.6849294, 7.8986363
1: -3.1057496, 2.9268103, -4.2268887, 3.7935634, -6.8993120, 7.1536989
2: -4.1053696, 3.1162860, -5.5311880, 3.9265852, -8.0319548, 8.6474743
3: -4.4386749, 2.7713633, -6.0121965, 3.5228319, -7.9615068, 8.7835569
4: -4.2074580, 3.1692934, -5.6721082, 4.1548419, -8.3622999, 8.8414021
5: -3.5509248, 3.0351212, -4.7857285, 3.9820225, -7.5329466, 7.8208485
6: -3.4747064, 3.3891344, -4.5897646, 4.4844198, -7.9591255, 7.9788990
7: -3.6690035, 3.5624459, -4.9027133, 4.7716417, -8.4406452, 8.4651594
8: -5.7219377, 3.3660746, -7.4882498, 3.8859463, -9.6078835, 10.8543224
9: -3.1934023, 3.4740181, -4.2854271, 4.5641932, -7.7575941, 7.7594442

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8518160, upper bound: 10.8518160
time: 2.50 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8518159, upper bound: 10.8518159
time: 2.42 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 6.33 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 6.33
Output dim: 8, lower bound: -10.8513517, upper bound: 10.8513518
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 6.33
Output dim: 8, lower bound: -10.8513517, upper bound: 10.8513594
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 6.33
Output dim: 8, lower bound: -10.8518160, upper bound: 10.8518160
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 6.33
Output dim: 8, lower bound: -10.8518159, upper bound: 10.8518159

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -2.9945498, 2.4854207, -2.9945498, 2.4854207, -5.4799705, 5.4799705
1: -2.4506655, 2.4260209, -2.4506655, 2.4260209, -4.8766861, 4.8766861
2: -3.2870402, 2.6394088, -3.2870402, 2.6394088, -5.9264488, 5.9264488
3: -3.5422220, 2.3601861, -3.5422220, 2.3601861, -5.9024081, 5.9024081
4: -3.3720222, 2.6054800, -3.3720222, 2.6054800, -5.9775019, 5.9775019
5: -2.8525395, 2.5022659, -2.8525395, 2.5022659, -5.3548055, 5.3548055
6: -2.8254833, 2.7629449, -2.8254833, 2.7629449, -5.5884285, 5.5884285
7: -2.9726825, 2.8755846, -2.9726825, 2.8755846, -5.8482671, 5.8482671
8: -4.6693630, 3.0776982, -4.6693630, 3.0776982, -7.7470613, 7.7470613
9: -2.5680308, 2.8494966, -2.5680308, 2.8494966, -5.4175272, 5.4175272

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8498914, upper bound: 10.8500985
time: 2.96 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8498953, upper bound: 10.8499696
time: 2.34 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -2.9945498, 2.4854207, -3.6680012, 3.0478530, -6.0424027, 6.1534214
1: -2.4506655, 2.4260209, -3.1057496, 2.9268103, -5.3774757, 5.5317698
2: -3.2870402, 2.6394088, -4.1053696, 3.1162860, -6.4033260, 6.7447786
3: -3.5422220, 2.3601861, -4.4386749, 2.7713633, -6.3135853, 6.7988610
4: -3.3720222, 2.6054800, -4.2074580, 3.1692934, -6.5413156, 6.8129377
5: -2.8525395, 2.5022659, -3.5509248, 3.0351212, -5.8876600, 6.0531907
6: -2.8254833, 2.7629449, -3.4747064, 3.3891344, -6.2146177, 6.2376509
7: -2.9726825, 2.8755846, -3.6690035, 3.5624459, -6.5351286, 6.5445881
8: -4.6693630, 3.0776982, -5.7219377, 3.3660746, -8.0354366, 8.7996359
9: -2.5680308, 2.8494966, -3.1934023, 3.4740181, -6.0420485, 6.0428991

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8498914, upper bound: 10.8500998
time: 3.25 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8498953, upper bound: 10.8499713
time: 4.59 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -3.5636892, 2.9592881, -3.8933744, 3.2337098, -6.7973990, 6.8526626
1: -3.0062733, 2.8517087, -3.3363652, 3.1137857, -6.1200590, 6.1880741
2: -3.9803500, 3.0446031, -4.4048572, 3.2788126, -7.2591629, 7.4494600
3: -4.2996650, 2.7120852, -4.7824364, 2.9859266, -7.2855916, 7.4945216
4: -4.0799570, 3.0834501, -4.5315251, 3.3776383, -7.4575953, 7.6149750
5: -3.4419408, 2.9548407, -3.7999735, 3.2426651, -6.6846056, 6.7548141
6: -3.3747091, 3.2941816, -3.7001369, 3.6212091, -6.9959183, 6.9943185
7: -3.5621767, 3.4570878, -3.9319315, 3.8218703, -7.3840470, 7.3890190
8: -5.5647755, 3.3167012, -6.0726881, 3.3668399, -8.9316158, 9.3893890
9: -3.0963645, 3.3779497, -3.4134443, 3.6960492, -6.7924137, 6.7913942

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8503118, upper bound: 10.8504700
time: 4.17 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8503134, upper bound: 10.8503141
time: 2.01 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -3.6586008, 3.0397885, -4.5213280, 3.7506666, -7.4092665, 7.5611162
1: -3.0967705, 2.9200139, -3.9277043, 3.5620401, -6.6588106, 6.8477182
2: -4.0941067, 3.1097631, -5.1526623, 3.7063074, -7.8004141, 8.2624254
3: -4.4261923, 2.7659798, -5.5974798, 3.3347614, -7.7609539, 8.3634586
4: -4.1959701, 3.1615443, -5.2926188, 3.8902135, -8.0861826, 8.4541626
5: -3.5409925, 3.0278649, -4.4566908, 3.7280974, -7.2690897, 7.4845548
6: -3.4656460, 3.3805823, -4.2881799, 4.1914425, -7.6570883, 7.6687622
7: -3.6593840, 3.5529578, -4.5747447, 4.4519367, -8.1113205, 8.1277027
8: -5.7077637, 3.3614755, -7.0174932, 3.6781023, -9.3858662, 10.3789682
9: -3.1846619, 3.4652820, -3.9907618, 4.2711110, -7.4557729, 7.4560437

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8503115, upper bound: 10.8504703
time: 4.29 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8503131, upper bound: 10.8503142
time: 2.99 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 8.66 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 8.66
Output dim: 8, lower bound: -10.8498914, upper bound: 10.8500985
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 8.66
Output dim: 8, lower bound: -10.8498953, upper bound: 10.8499696
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 8.66
Output dim: 8, lower bound: -10.8498914, upper bound: 10.8500998
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 8.66
Output dim: 8, lower bound: -10.8498953, upper bound: 10.8499713
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 8.66
Output dim: 8, lower bound: -10.8503118, upper bound: 10.8504700
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 8.66
Output dim: 8, lower bound: -10.8503134, upper bound: 10.8503141
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 8.66
Output dim: 8, lower bound: -10.8503115, upper bound: 10.8504703
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 8.66
Output dim: 8, lower bound: -10.8503131, upper bound: 10.8503142

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -1.1463287, 1.1388307, -2.9359334, 2.4369869, -3.5833156, 4.0747643
1: -0.9501979, 1.0848776, -2.3943515, 2.3831568, -3.3333547, 3.4792290
2: -1.0575454, 1.3447318, -3.2147088, 2.5966134, -3.6541588, 4.5594406
3: -1.0804440, 1.2116272, -3.4645827, 2.3234270, -3.4038711, 4.6762099
4: -1.2093067, 1.1103208, -3.2999482, 2.5566926, -3.7659993, 4.4102688
5: -1.0625710, 1.1400380, -2.7925649, 2.4565690, -3.5191400, 3.9326029
6: -1.1217198, 1.1095903, -2.7702298, 2.7087331, -3.8304529, 3.8798201
7: -1.1063398, 1.1148367, -2.9130421, 2.8164728, -3.9228125, 4.0278788
8: -1.6310238, 2.6237946, -4.5786653, 3.0566869, -4.6877108, 7.2024598
9: -1.0470525, 1.2589920, -2.5136828, 2.7966509, -3.8437033, 3.7726748

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8448135, upper bound: 10.8459641
time: 2.13 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8485760, upper bound: 10.8484098
time: 2.54 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -1.9651489, 1.6936014, -2.7269721, 2.2694843, -4.2346334, 4.4205732
1: -1.5800816, 1.6665426, -2.2029314, 2.2298827, -3.8099642, 3.8694739
2: -1.9933929, 1.9163849, -2.9587407, 2.4510560, -4.4444489, 4.8751259
3: -2.1288476, 1.6861715, -3.1870484, 2.1940756, -4.3229232, 4.8732200
4: -2.1741326, 1.7406292, -3.0586734, 2.3816986, -4.5558310, 4.7993026
5: -1.8066512, 1.7302947, -2.5766160, 2.2985775, -4.1052289, 4.3069105
6: -1.8680632, 1.8223088, -2.5742521, 2.5181243, -4.3861876, 4.3965607
7: -1.9071271, 1.8472733, -2.7012956, 2.6038642, -4.5109911, 4.5485687
8: -3.0221694, 2.7996442, -4.2545695, 2.9881923, -6.0103617, 7.0542135
9: -1.6512964, 1.9428821, -2.3220158, 2.6130495, -4.2643461, 4.2648978

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8450940, upper bound: 10.8463113
time: 2.11 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8486080, upper bound: 10.8486082
time: 4.32 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -1.1463287, 1.1388307, -3.6075156, 2.9971814, -4.1435099, 4.7463465
1: -0.9501979, 1.0848776, -3.0482330, 2.8824544, -3.8326523, 4.1331105
2: -1.0575454, 1.3447318, -4.0311747, 3.0715322, -4.1290779, 5.3759065
3: -1.0804440, 1.2116272, -4.3590574, 2.7328594, -3.8133035, 5.5706844
4: -1.2093067, 1.1103208, -4.1329613, 3.1187801, -4.3280869, 5.2432823
5: -1.0625710, 1.1400380, -3.4878426, 2.9871020, -4.0496731, 4.6278806
6: -1.1217198, 1.1095903, -3.4160366, 3.3334279, -4.4551477, 4.5256271
7: -1.1063398, 1.1148367, -3.6061249, 3.5019808, -4.6083207, 4.7209616
8: -1.6310238, 2.6237946, -5.6292839, 3.3407855, -4.9718094, 8.2530785
9: -1.0470525, 1.2589920, -3.1373594, 3.4181702, -4.4652228, 4.3963513

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8502737, upper bound: 10.8500998
time: 3.38 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8502736, upper bound: 10.8500998
time: 2.55 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -1.9651489, 1.6936014, -3.3892169, 2.8172297, -4.7823787, 5.0828180
1: -1.5800816, 1.6665426, -2.8383417, 2.7217436, -4.3018250, 4.5048842
2: -1.9933929, 1.9163849, -3.7647855, 2.9155529, -4.9089460, 5.6811705
3: -2.1288476, 1.6861715, -4.0701461, 2.5958488, -4.7246962, 5.7563176
4: -2.1741326, 1.7406292, -3.8618994, 2.9362867, -5.1104193, 5.6025286
5: -1.8066512, 1.7302947, -3.2647469, 2.8136322, -4.6202836, 4.9950418
6: -1.8680632, 1.8223088, -3.2058873, 3.1316013, -4.9996643, 5.0281963
7: -1.9071271, 1.8472733, -3.3786545, 3.2812319, -5.1883593, 5.2259278
8: -3.0221694, 2.7996442, -5.2939162, 3.2567909, -6.2789602, 8.0935602
9: -1.6512964, 1.9428821, -2.9347641, 3.2179065, -4.8692026, 4.8776464

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8502750, upper bound: 10.8499712
time: 2.44 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8502750, upper bound: 10.8499711
time: 2.10 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -1.6261789, 1.4446988, -3.8321962, 3.1824815, -4.8086605, 5.2768950
1: -1.3204114, 1.4341936, -3.2782249, 3.0689805, -4.3893919, 4.7124186
2: -1.5655643, 1.6549833, -4.3298740, 3.2337184, -4.7992826, 5.9848576
3: -1.6808292, 1.4865319, -4.7021408, 2.9470301, -4.6278591, 6.1886725
4: -1.7803187, 1.4773070, -4.4563322, 3.3265715, -5.1068902, 5.9336390
5: -1.5006416, 1.4857213, -3.7361603, 3.1941988, -4.6948404, 5.2218819
6: -1.5490335, 1.5222368, -3.6409931, 3.5648689, -5.1139026, 5.1632299
7: -1.5688956, 1.5355225, -3.8684356, 3.7606359, -5.3295317, 5.4039583
8: -2.4572721, 2.7478786, -5.9789147, 3.3372796, -5.7945518, 8.7267933
9: -1.3914628, 1.6519135, -3.3568506, 3.6395884, -5.0310512, 5.0087643

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8461066, upper bound: 10.8470127
time: 2.20 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8489629, upper bound: 10.8488470
time: 2.69 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -2.4667835, 2.0668497, -3.6266627, 3.0095863, -5.4763699, 5.6935124
1: -1.9951704, 2.0402424, -3.0804183, 2.9166152, -4.9117856, 5.1206608
2: -2.6282482, 2.2499514, -4.0791473, 3.0865517, -5.7147999, 6.3290987
3: -2.8241405, 1.9974011, -4.4311175, 2.8175533, -5.6416941, 6.4285188
4: -2.7634697, 2.1578138, -4.2012234, 3.1541829, -5.9176526, 6.3590374
5: -2.3108869, 2.1063786, -3.5197911, 3.0307648, -5.3416519, 5.6261697
6: -2.3251629, 2.2830393, -3.4429653, 3.3743141, -5.6994772, 5.7260046
7: -2.4331374, 2.3469868, -3.6540649, 3.5527799, -5.9859171, 6.0010519
8: -3.8660214, 2.9625652, -5.6623545, 3.2442102, -7.1102314, 8.6249199
9: -2.0952880, 2.3902946, -3.1665297, 3.4496765, -5.5449648, 5.5568242

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8461991, upper bound: 10.8472733
time: 4.55 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8489862, upper bound: 10.8489862
time: 2.87 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -1.7012315, 1.4988158, -4.4585571, 3.7001510, -5.4013824, 5.9573727
1: -1.3792017, 1.4881947, -3.8702679, 3.5168669, -4.8960686, 5.3584623
2: -1.6563637, 1.7029923, -5.0780187, 3.6615257, -5.3178892, 6.7810111
3: -1.7834704, 1.5273528, -5.5176258, 3.2960734, -5.0795441, 7.0449786
4: -1.8701279, 1.5366575, -5.2182417, 3.8394215, -5.7095494, 6.7548990
5: -1.5707138, 1.5399677, -4.3934340, 3.6784828, -5.2491965, 5.9334016
6: -1.6170659, 1.5895599, -4.2298737, 4.1344719, -5.7515378, 5.8194337
7: -1.6437995, 1.6063063, -4.5104876, 4.3906422, -6.0344419, 6.1167936
8: -2.5865841, 2.7680106, -6.9249077, 3.6445231, -6.2311072, 9.6929188
9: -1.4471902, 1.7179668, -3.9341905, 4.2146368, -5.6618271, 5.6521573

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8502529, upper bound: 10.8498238
time: 3.82 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8502529, upper bound: 10.8502524
time: 3.56 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -2.5532863, 2.1374898, -4.2390113, 3.5187421, -6.0720282, 6.3765011
1: -2.0654128, 2.1013300, -3.6617169, 3.3546057, -5.4200182, 5.7630472
2: -2.7338185, 2.3062928, -4.8120432, 3.5060019, -6.2398205, 7.1183357
3: -2.9392681, 2.0467706, -5.2302971, 3.1595144, -6.0987825, 7.2770677
4: -2.8623862, 2.2294815, -4.9487953, 3.6568284, -6.5192146, 7.1782770
5: -2.4035511, 2.1671262, -4.1647549, 3.5041392, -5.9076900, 6.3318810
6: -2.4025979, 2.3607879, -4.0216103, 3.9311466, -6.3337445, 6.3823981
7: -2.5204720, 2.4331856, -4.2828045, 4.1690998, -6.6895719, 6.7159901
8: -4.0017319, 2.9927511, -6.5919776, 3.5343218, -7.5360537, 9.5847282
9: -2.1714344, 2.4668164, -3.7313299, 4.0116029, -6.1830373, 6.1981463

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8499025, upper bound: 10.8502749
time: 3.17 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8499025, upper bound: 10.8502749
time: 2.77 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 7.33 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 7.33
Output dim: 8, lower bound: -10.8448135, upper bound: 10.8459641
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 7.33
Output dim: 8, lower bound: -10.8485760, upper bound: 10.8484098
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 7.33
Output dim: 8, lower bound: -10.8450940, upper bound: 10.8463113
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 7.33
Output dim: 8, lower bound: -10.8486080, upper bound: 10.8486082
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 7.33
Output dim: 8, lower bound: -10.8502737, upper bound: 10.8500998
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 7.33
Output dim: 8, lower bound: -10.8502736, upper bound: 10.8500998
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 7.33
Output dim: 8, lower bound: -10.8502750, upper bound: 10.8499712
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 7.33
Output dim: 8, lower bound: -10.8502750, upper bound: 10.8499711
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 7.33
Output dim: 8, lower bound: -10.8461066, upper bound: 10.8470127
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 7.33
Output dim: 8, lower bound: -10.8489629, upper bound: 10.8488470
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 7.33
Output dim: 8, lower bound: -10.8461991, upper bound: 10.8472733
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 7.33
Output dim: 8, lower bound: -10.8489862, upper bound: 10.8489862
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 7.33
Output dim: 8, lower bound: -10.8502529, upper bound: 10.8498238
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 7.33
Output dim: 8, lower bound: -10.8502529, upper bound: 10.8502524
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 7.33
Output dim: 8, lower bound: -10.8499025, upper bound: 10.8502749
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 7.33
Output dim: 8, lower bound: -10.8499025, upper bound: 10.8502749

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.9181588, 1.0023355, -0.5878258, 0.7754006, -1.6935594, 1.5901613
1: -0.7853019, 0.9227080, -0.5120600, 0.6335638, -1.4188657, 1.4347681
2: -0.8301572, 1.1799483, -0.5597510, 0.8805442, -1.7107013, 1.7396994
3: -0.8154722, 1.0816091, -0.4261194, 0.7950267, -1.6104989, 1.5077286
4: -0.9508305, 0.9370227, -0.6182709, 0.6434394, -1.5942699, 1.5552936
5: -0.8568289, 0.9762193, -0.5767512, 0.6985697, -1.5553986, 1.5529705
6: -0.9082052, 0.9222766, -0.5775142, 0.6464877, -1.5546929, 1.4997907
7: -0.9005802, 0.9199703, -0.6030902, 0.6146899, -1.5152700, 1.5230606
8: -1.2342744, 2.5890667, -0.6026796, 2.5124264, -3.7467008, 3.1917462
9: -0.8951771, 1.0775614, -0.6915110, 0.7876092, -1.6827862, 1.7690723

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8448135, upper bound: 10.8459641
time: 2.86 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8448135, upper bound: 10.8459641
time: 2.50 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -1.1089281, 1.1171485, -1.7902300, 1.5642440, -2.6731720, 2.9073787
1: -0.9234869, 1.0591009, -1.4399142, 1.5293304, -2.4528172, 2.4990151
2: -1.0214140, 1.3196528, -1.7840327, 1.8094037, -2.8308177, 3.1036854
3: -1.0372087, 1.1916637, -1.8807935, 1.5737431, -2.6109519, 3.0724573
4: -1.1655810, 1.0823485, -1.9664638, 1.5990214, -2.7646024, 3.0488124
5: -1.0288326, 1.1131084, -1.6407146, 1.6005707, -2.6294031, 2.7538230
6: -1.0883856, 1.0789829, -1.7037392, 1.6607120, -2.7490976, 2.7827220
7: -1.0714797, 1.0840160, -1.7294710, 1.6802700, -2.7517495, 2.8134871
8: -1.5679581, 2.6191223, -2.7172871, 2.7470040, -4.3149624, 5.3364096
9: -1.0218763, 1.2295468, -1.5135455, 1.7844603, -2.8063366, 2.7430923

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8468861, upper bound: 10.8464194
time: 2.09 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8468861, upper bound: 10.8464193
time: 2.21 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -1.6506455, 1.4727306, -0.5062460, 0.7087560, -2.3594015, 1.9789766
1: -1.3291407, 1.4339296, -0.4469584, 0.5586731, -1.8878138, 1.8808880
2: -1.6115177, 1.7014270, -0.5244555, 0.7842703, -2.3957880, 2.2258825
3: -1.6777989, 1.4672542, -0.3421311, 0.7294438, -2.4072428, 1.8093853
4: -1.8084658, 1.4872353, -0.5439377, 0.5692471, -2.3777130, 2.0311730
5: -1.5147009, 1.4963413, -0.5057573, 0.6328741, -2.1475749, 2.0020986
6: -1.5747716, 1.5373733, -0.4996258, 0.5811428, -2.1559145, 2.0369992
7: -1.5904909, 1.5526869, -0.5305357, 0.5392081, -2.1296990, 2.0832226
8: -2.4934487, 2.7348378, -0.4527779, 2.4955411, -4.9889898, 3.1876156
9: -1.4112105, 1.6637282, -0.6548167, 0.7116795, -2.1228900, 2.3185449

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8450940, upper bound: 10.8463113
time: 2.19 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8450940, upper bound: 10.8463113
time: 1.84 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -1.9274917, 1.6669704, -1.6143622, 1.4419580, -3.3694496, 3.2813325
1: -1.5502402, 1.6382838, -1.3016973, 1.4000299, -2.9502702, 2.9399810
2: -1.9475894, 1.8909440, -1.5708747, 1.6957514, -3.6433408, 3.4618187
3: -2.0755079, 1.6619997, -1.6347383, 1.4671333, -3.5426412, 3.2967381
4: -2.1295292, 1.7104421, -1.7573085, 1.4595085, -3.5890379, 3.4677505
5: -1.7715122, 1.7023313, -1.4770014, 1.4731447, -3.2446568, 3.1793327
6: -1.8332955, 1.7878597, -1.5444671, 1.5034473, -3.3367429, 3.3323269
7: -1.8674258, 1.8115458, -1.5531086, 1.5141580, -3.3815837, 3.3646545
8: -2.9587073, 2.7923102, -2.4204597, 2.7151217, -5.6738291, 5.2127700
9: -1.6210880, 1.9095658, -1.3825499, 1.6310054, -3.2520933, 3.2921157

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8486080, upper bound: 10.8486083
time: 3.97 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8486080, upper bound: 10.8486079
time: 2.29 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -1.0719728, 1.0925107, -2.7644606, 2.3014555, -3.3734283, 3.8569713
1: -0.8971055, 1.0351620, -2.2410917, 2.2823696, -3.1794751, 3.2762537
2: -0.9850948, 1.2969804, -3.0205889, 2.4945168, -3.4796116, 4.3175693
3: -0.9948242, 1.1736166, -3.2572219, 2.2631302, -3.2579544, 4.4308386
4: -1.1234049, 1.0542817, -3.1099172, 2.4248657, -3.5482707, 4.1641989
5: -0.9949080, 1.0875671, -2.6197200, 2.3413708, -3.3362789, 3.7072871
6: -1.0552293, 1.0481771, -2.6196885, 2.5639405, -3.6191697, 3.6678658
7: -1.0369927, 1.0539336, -2.7450228, 2.6508985, -3.6878910, 3.7989564
8: -1.5034766, 2.6043007, -4.3321776, 3.0142336, -4.5177102, 6.9364786
9: -0.9970530, 1.1996217, -2.3637977, 2.6598098, -3.6568627, 3.5634193

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8468509, upper bound: 10.8462607
time: 2.12 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8488905, upper bound: 10.8483956
time: 2.88 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -1.1397295, 1.1346816, -3.3235030, 2.7575383, -3.8972678, 4.4581847
1: -0.9454300, 1.0802951, -2.7765627, 2.6770041, -3.6224341, 3.8568578
2: -1.0510892, 1.3404247, -3.6902697, 2.8743911, -3.9254804, 5.0306945
3: -1.0728019, 1.2081779, -3.9847913, 2.5697119, -3.6425138, 5.1929693
4: -1.2016361, 1.1053023, -3.7859116, 2.8845837, -4.0862198, 4.8912139
5: -1.0565734, 1.1353018, -3.1963296, 2.7671986, -3.8237720, 4.3316317
6: -1.1157839, 1.1040953, -3.1439161, 3.0731502, -4.1889343, 4.2480116
7: -1.1001567, 1.1094491, -3.3159833, 3.2153602, -4.3155169, 4.4254322
8: -1.6195865, 2.6219907, -5.1940784, 3.2101922, -4.8297787, 7.8160691
9: -1.0425315, 1.2536941, -2.8741908, 3.1573048, -4.1998363, 4.1278849

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8468509, upper bound: 10.8462607
time: 2.22 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8488905, upper bound: 10.8483956
time: 2.17 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -1.8828317, 1.6329288, -2.5627952, 2.1426468, -4.0254784, 4.1957240
1: -1.5145121, 1.6068411, -2.0747862, 2.1351552, -3.6496673, 3.6816273
2: -1.8939302, 1.8630209, -2.7713833, 2.3566985, -4.2506285, 4.6344042
3: -2.0142226, 1.6389668, -2.9891107, 2.1343460, -4.1485686, 4.6280775
4: -2.0758691, 1.6749401, -2.8789651, 2.2539721, -4.3298411, 4.5539055
5: -1.7289028, 1.6701242, -2.4080691, 2.1937485, -3.9226513, 4.0781932
6: -1.7918645, 1.7473032, -2.4298172, 2.3813434, -4.1732078, 4.1771202
7: -1.8210597, 1.7697040, -2.5383379, 2.4477301, -4.2687898, 4.3080420
8: -2.8806787, 2.7748322, -4.0152445, 2.9532025, -5.8338814, 6.7900767
9: -1.5862136, 1.8696995, -2.1867085, 2.4839084, -4.0701218, 4.0564079

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8459066, upper bound: 10.8448464
time: 1.49 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8489285, upper bound: 10.8485977
time: 2.05 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -1.9578618, 1.6881763, -3.1085410, 2.5798502, -4.5377121, 4.7967172
1: -1.5742772, 1.6612045, -2.5688577, 2.5182109, -4.0924883, 4.2300620
2: -1.9845445, 1.9116075, -3.4269943, 2.7203400, -4.7048845, 5.3386021
3: -2.1187029, 1.6819651, -3.6999130, 2.4354820, -4.5541849, 5.3818779
4: -2.1654356, 1.7348034, -3.5184467, 2.7036831, -4.8691187, 5.2532501
5: -1.7997651, 1.7249398, -2.9756784, 2.5970242, -4.3967896, 4.7006183
6: -1.8613185, 1.8156052, -2.9361157, 2.8734560, -4.7347746, 4.7517209
7: -1.8994396, 1.8403778, -3.0927870, 2.9969969, -4.8964367, 4.9331646
8: -3.0095928, 2.7973683, -4.8606062, 3.1348724, -6.1444654, 7.6579742
9: -1.6454034, 1.9363954, -2.6750872, 2.9616559, -4.6070595, 4.6114826

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8502749, upper bound: 10.8499711
time: 2.70 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8502749, upper bound: 10.8499713
time: 2.53 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -1.3677335, 1.2736822, -1.0979109, 1.0934272, -2.4611607, 2.3715930
1: -1.1195953, 1.2450639, -0.9127841, 1.0435810, -2.1631763, 2.1578479
2: -1.2734703, 1.4768113, -0.9851624, 1.2707365, -2.5442066, 2.4619737
3: -1.3442304, 1.3361281, -1.0364298, 1.1718731, -2.5161035, 2.3725579
4: -1.4723403, 1.2793055, -1.1655684, 1.0673933, -2.5397336, 2.4448738
5: -1.2636328, 1.2978883, -1.0173315, 1.0980639, -2.3616967, 2.3152199
6: -1.3165915, 1.2958851, -1.0496970, 1.0650580, -2.3816495, 2.3455820
7: -1.3167102, 1.3021507, -1.0687828, 1.0783980, -2.3951082, 2.3709335
8: -2.0164511, 2.7048945, -1.5155067, 2.5159533, -4.5324044, 4.2204013
9: -1.2014222, 1.4343657, -1.0012952, 1.2070904, -2.4085126, 2.4356608

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8454210, upper bound: 10.8458737
time: 2.41 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8454210, upper bound: 10.8470127
time: 2.60 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -1.5868207, 1.4174043, -2.5966496, 2.1585758, -3.7453966, 4.0140538
1: -1.2896252, 1.4049867, -2.1120830, 2.1464112, -3.4360363, 3.5170698
2: -1.5173934, 1.6288576, -2.8143430, 2.3446460, -3.8620393, 4.4432006
3: -1.6267543, 1.4640787, -3.0527534, 2.1355460, -3.7623003, 4.5168324
4: -1.7332771, 1.4462975, -2.9394550, 2.2805462, -4.0138235, 4.3857527
5: -1.4640822, 1.4569957, -2.4537578, 2.2213931, -3.6854753, 3.9107535
6: -1.5133796, 1.4871062, -2.4424789, 2.4157839, -3.9291635, 3.9295850
7: -1.5294094, 1.4981563, -2.5893219, 2.5065994, -4.0360088, 4.0874782
8: -2.3901722, 2.7416868, -4.0603104, 2.8734493, -5.2636213, 6.8019972
9: -1.3622926, 1.6174881, -2.2142806, 2.5142438, -3.8765364, 3.8317688

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8475545, upper bound: 10.8470437
time: 2.05 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8475545, upper bound: 10.8470437
time: 2.40 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -2.1387625, 1.8181485, -0.9340247, 0.9944582, -3.1332207, 2.7521732
1: -1.7246728, 1.7995710, -0.7952818, 0.9275866, -2.6522593, 2.5948529
2: -2.2100644, 2.0209846, -0.8229578, 1.1561425, -3.3662069, 2.8439424
3: -2.3652773, 1.7633109, -0.8428412, 1.0846390, -3.4499164, 2.6061521
4: -2.3906076, 1.8800204, -0.9737825, 0.9441838, -3.3347914, 2.8538029
5: -1.9732547, 1.8595989, -0.8685680, 0.9805970, -2.9538517, 2.7281668
6: -2.0201747, 1.9830225, -0.8996621, 0.9299837, -2.9501586, 2.8826847
7: -2.0937581, 2.0217538, -0.9173845, 0.9379766, -3.0317347, 2.9391384
8: -3.3309836, 2.8745315, -1.2290571, 2.4945145, -5.8254981, 4.1035886
9: -1.8062415, 2.0980067, -0.8926961, 1.0754068, -2.8816483, 2.9907029

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8461991, upper bound: 10.8472733
time: 2.46 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8461991, upper bound: 10.8472733
time: 3.19 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -2.4272156, 2.0371432, -2.4267495, 2.0198803, -4.4470959, 4.4638929
1: -1.9633653, 2.0116210, -1.9718368, 2.0182757, -3.9816411, 3.9834578
2: -2.5794144, 2.2233710, -2.5947704, 2.2246590, -4.8040733, 4.8181415
3: -2.7712159, 1.9714339, -2.8192763, 2.0245657, -4.7957816, 4.7907104
4: -2.7191224, 2.1247978, -2.7369566, 2.1327066, -4.8518291, 4.8617544
5: -2.2694693, 2.0777559, -2.2742004, 2.0928924, -4.3623619, 4.3519564
6: -2.2897308, 2.2476118, -2.2811627, 2.2566981, -4.5464287, 4.5287743
7: -2.3930507, 2.3078229, -2.4087429, 2.3337183, -4.7267690, 4.7165661
8: -3.8033760, 2.9524498, -3.7808492, 2.8222334, -6.6256094, 6.7332993
9: -2.0613606, 2.3557463, -2.0605507, 2.3612871, -4.4226475, 4.4162970

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8472733, upper bound: 10.8461990
time: 1.98 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8472733, upper bound: 10.8461989
time: 2.05 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -1.7012315, 1.4988158, -2.4537344, 2.0413580, -3.7425895, 3.9525502
1: -1.3792017, 1.4881947, -2.0024812, 2.0433486, -3.4225502, 3.4906759
2: -1.6563637, 1.7029923, -2.6022239, 2.1899247, -3.8462884, 4.3052163
3: -1.7834704, 1.5273528, -2.8723474, 2.0187831, -3.8022535, 4.3997002
4: -1.8701279, 1.5366575, -2.7795520, 2.1584210, -4.0285492, 4.3162093
5: -1.5707138, 1.5399677, -2.3218193, 2.1091714, -3.6798851, 3.8617868
6: -1.6170659, 1.5895599, -2.2985623, 2.2840483, -3.9011142, 3.8881221
7: -1.6437995, 1.6063063, -2.4423211, 2.3698483, -4.0136480, 4.0486274
8: -2.5865841, 2.7680106, -3.8316989, 2.8167729, -5.4033570, 6.5997095
9: -1.4471902, 1.7179668, -2.0896592, 2.3859878, -3.8331780, 3.8076260

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8475311, upper bound: 10.8470437
time: 2.75 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8487578, upper bound: 10.8488471
time: 6.10 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -1.7012315, 1.4988158, -3.4058876, 2.8222871, -4.5235186, 4.9047031
1: -1.3792017, 1.4881947, -2.8615646, 2.7321920, -4.1113939, 4.3497591
2: -1.6563637, 1.7029923, -3.7859368, 2.8935270, -4.5498905, 5.4889293
3: -1.7834704, 1.5273528, -4.1148424, 2.5997860, -4.3832564, 5.6421952
4: -1.8701279, 1.5366575, -3.9149845, 2.9522479, -4.8223758, 5.4516420
5: -1.5707138, 1.5399677, -3.2967706, 2.8345456, -4.4052591, 4.8367381
6: -1.6170659, 1.5895599, -3.2124498, 3.1543715, -4.7714376, 4.8020096
7: -1.6437995, 1.6063063, -3.4143915, 3.3287570, -4.9725566, 5.0206976
8: -2.5865841, 2.7680106, -5.3182182, 3.1655350, -5.7521191, 8.0862293
9: -1.4471902, 1.7179668, -2.9500194, 3.2389216, -4.6861119, 4.6679859

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8475311, upper bound: 10.8470437
time: 2.00 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8487578, upper bound: 10.8488471
time: 3.07 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -2.5532863, 2.1374898, -2.4728999, 2.0647445, -4.6180305, 4.6103897
1: -2.0654128, 2.1013300, -2.0001447, 2.0473304, -4.1127434, 4.1014748
2: -2.7338185, 2.3062928, -2.6507483, 2.2812443, -5.0150628, 4.9570408
3: -2.9392681, 2.0467706, -2.8546143, 2.0454047, -4.9846725, 4.9013848
4: -2.8623862, 2.2294815, -2.7708211, 2.1708169, -5.0332031, 5.0003023
5: -2.4035511, 2.1671262, -2.3094630, 2.1154160, -4.5189672, 4.4765892
6: -2.4025979, 2.3607879, -2.3405354, 2.2895494, -4.6921473, 4.7013235
7: -2.5204720, 2.4331856, -2.4443541, 2.3522027, -4.8726749, 4.8775396
8: -4.0017319, 2.9927511, -3.8514762, 2.8919880, -6.8937197, 6.8442273
9: -2.1714344, 2.4668164, -2.0990751, 2.3915987, -4.5630331, 4.5658913

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8499030, upper bound: 10.8502750
time: 17.14 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8499030, upper bound: 10.8502750
time: 2.14 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -2.5532863, 2.1374898, -3.1085410, 2.5798502, -5.1331367, 5.2460308
1: -2.0654128, 2.1013300, -2.5688577, 2.5182109, -4.5836239, 4.6701880
2: -2.7338185, 2.3062928, -3.4269943, 2.7203400, -5.4541588, 5.7332869
3: -2.9392681, 2.0467706, -3.6999130, 2.4354820, -5.3747501, 5.7466836
4: -2.8623862, 2.2294815, -3.5184467, 2.7036831, -5.5660696, 5.7479281
5: -2.4035511, 2.1671262, -2.9756784, 2.5970242, -5.0005751, 5.1428046
6: -2.4025979, 2.3607879, -2.9361157, 2.8734560, -5.2760539, 5.2969036
7: -2.5204720, 2.4331856, -3.0927870, 2.9969969, -5.5174689, 5.5259724
8: -4.0017319, 2.9927511, -4.8606062, 3.1348724, -7.1366043, 7.8533573
9: -2.1714344, 2.4668164, -2.6750872, 2.9616559, -5.1330900, 5.1419039

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8499030, upper bound: 10.8503132
time: 1.86 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8499030, upper bound: 10.8503141
time: 2.71 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 6.03 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.03
Output dim: 8, lower bound: -10.8448135, upper bound: 10.8459641
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.03
Output dim: 8, lower bound: -10.8448135, upper bound: 10.8459641
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.03
Output dim: 8, lower bound: -10.8468861, upper bound: 10.8464194
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.03
Output dim: 8, lower bound: -10.8468861, upper bound: 10.8464193
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.03
Output dim: 8, lower bound: -10.8450940, upper bound: 10.8463113
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.03
Output dim: 8, lower bound: -10.8450940, upper bound: 10.8463113
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.03
Output dim: 8, lower bound: -10.8486080, upper bound: 10.8486083
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.03
Output dim: 8, lower bound: -10.8486080, upper bound: 10.8486079
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.03
Output dim: 8, lower bound: -10.8468509, upper bound: 10.8462607
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.03
Output dim: 8, lower bound: -10.8488905, upper bound: 10.8483956
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.03
Output dim: 8, lower bound: -10.8468509, upper bound: 10.8462607
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.03
Output dim: 8, lower bound: -10.8488905, upper bound: 10.8483956
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.03
Output dim: 8, lower bound: -10.8459066, upper bound: 10.8448464
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.03
Output dim: 8, lower bound: -10.8489285, upper bound: 10.8485977
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.03
Output dim: 8, lower bound: -10.8502749, upper bound: 10.8499711
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.03
Output dim: 8, lower bound: -10.8502749, upper bound: 10.8499713
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.03
Output dim: 8, lower bound: -10.8454210, upper bound: 10.8458737
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.03
Output dim: 8, lower bound: -10.8454210, upper bound: 10.8470127
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.03
Output dim: 8, lower bound: -10.8475545, upper bound: 10.8470437
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.03
Output dim: 8, lower bound: -10.8475545, upper bound: 10.8470437
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.03
Output dim: 8, lower bound: -10.8461991, upper bound: 10.8472733
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.03
Output dim: 8, lower bound: -10.8461991, upper bound: 10.8472733
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.03
Output dim: 8, lower bound: -10.8472733, upper bound: 10.8461990
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.03
Output dim: 8, lower bound: -10.8472733, upper bound: 10.8461989
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.03
Output dim: 8, lower bound: -10.8475311, upper bound: 10.8470437
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.03
Output dim: 8, lower bound: -10.8487578, upper bound: 10.8488471
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.03
Output dim: 8, lower bound: -10.8475311, upper bound: 10.8470437
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.03
Output dim: 8, lower bound: -10.8487578, upper bound: 10.8488471
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.03
Output dim: 8, lower bound: -10.8499030, upper bound: 10.8502750
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.03
Output dim: 8, lower bound: -10.8499030, upper bound: 10.8502750
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.03
Output dim: 8, lower bound: -10.8499030, upper bound: 10.8503132
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.03
Output dim: 8, lower bound: -10.8499030, upper bound: 10.8503141

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.5030130, 0.6738580, -0.5527236, 0.7441187, -1.2471317, 1.2265816
1: -0.4618849, 0.5729823, -0.4836035, 0.6034496, -1.0653346, 1.0565858
2: -0.5341837, 0.7911453, -0.5424044, 0.8420688, -1.3762525, 1.3335497
3: -0.3394054, 0.7994530, -0.3881693, 0.7685074, -1.1079128, 1.1876223
4: -0.5410276, 0.5724103, -0.5869045, 0.6095464, -1.1505740, 1.1593149
5: -0.4987844, 0.6418643, -0.5476568, 0.6694705, -1.1682549, 1.1895211
6: -0.5178655, 0.5697587, -0.5453771, 0.6180836, -1.1359491, 1.1151358
7: -0.5264295, 0.5282149, -0.5702510, 0.5818527, -1.1082821, 1.0984659
8: -0.4561309, 2.4688973, -0.5365293, 2.4949899, -2.9511209, 3.0054266
9: -0.6557375, 0.7080813, -0.6733946, 0.7546420, -1.4103795, 1.3814759

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8445362, upper bound: 10.8458287
time: 1.82 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8445362, upper bound: 10.8459641
time: 2.56 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.7561775, 0.8877184, -0.5846403, 0.7724472, -1.5286248, 1.4723587
1: -0.6655762, 0.8008986, -0.5093484, 0.6308017, -1.2963779, 1.3102469
2: -0.6844483, 1.0620499, -0.5580912, 0.8769798, -1.5614281, 1.6201410
3: -0.6218578, 0.9901599, -0.4226653, 0.7925181, -1.4143759, 1.4128252
4: -0.7793264, 0.8067955, -0.6153806, 0.6402822, -1.4196086, 1.4221761
5: -0.7249805, 0.8542902, -0.5741050, 0.6958625, -1.4208431, 1.4283953
6: -0.7602545, 0.7828797, -0.5745282, 0.6438760, -1.4041305, 1.3574078
7: -0.7599026, 0.7770091, -0.6001158, 0.6115033, -1.3714058, 1.3771250
8: -0.9422535, 2.5381758, -0.5963911, 2.5108347, -3.4530883, 3.1345668
9: -0.7969270, 0.9404777, -0.6898326, 0.7846042, -1.5815312, 1.6303103

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8445362, upper bound: 10.8458287
time: 2.27 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8445362, upper bound: 10.8459641
time: 2.22 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.2649682, 0.4365217, -1.7902300, 1.5642440, -1.8292122, 2.2267518
1: -0.2110203, 0.3204181, -1.4399142, 1.5293304, -1.7403507, 1.7603323
2: -0.4011096, 0.3798940, -1.7840327, 1.8094037, -2.2105134, 2.1639266
3: -0.1496294, 0.4142324, -1.8807935, 1.5737431, -1.7233725, 2.2950258
4: -0.3361644, 0.2973544, -1.9664638, 1.5990214, -1.9351858, 2.2638183
5: -0.2678746, 0.3659123, -1.6407146, 1.6005707, -1.8684453, 2.0066271
6: -0.2380511, 0.3307576, -1.7037392, 1.6607120, -1.8987632, 2.0344968
7: -0.3203646, 0.2548853, -1.7294710, 1.6802700, -2.0006347, 1.9843562
8: 0.1495964, 2.3928282, -2.7172871, 2.7470040, -2.5974076, 5.1101151
9: -0.5390543, 0.4187671, -1.5135455, 1.7844603, -2.3235145, 1.9323126

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8438410, upper bound: 10.8464190
time: 1.82 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8438410, upper bound: 10.8464193
time: 4.12 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.4962973, 0.6877433, -1.7902300, 1.5642440, -2.0605414, 2.4779735
1: -0.4526193, 0.5614443, -1.4399142, 1.5293304, -1.9819497, 2.0013585
2: -0.5372895, 0.7767532, -1.7840327, 1.8094037, -2.3466930, 2.5607858
3: -0.3340729, 0.7750189, -1.8807935, 1.5737431, -1.9078161, 2.6558123
4: -0.5360340, 0.5677570, -1.9664638, 1.5990214, -2.1350555, 2.5342207
5: -0.4944470, 0.6357657, -1.6407146, 1.6005707, -2.0950177, 2.2764804
6: -0.5096372, 0.5704117, -1.7037392, 1.6607120, -2.1703491, 2.2741508
7: -0.5219655, 0.5243276, -1.7294710, 1.6802700, -2.2022355, 2.2537985
8: -0.4517331, 2.5235822, -2.7172871, 2.7470040, -3.1987371, 5.2408695
9: -0.6627966, 0.7065476, -1.5135455, 1.7844603, -2.4472568, 2.2200930

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8438410, upper bound: 10.8446888
time: 8.65 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8438410, upper bound: 10.8484082
time: 2.24 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.9170872, 1.0028160, -0.4803811, 0.6803163, -1.5974035, 1.4831971
1: -0.7836363, 0.9258493, -0.4244257, 0.5304357, -1.3140720, 1.3502750
2: -0.8399833, 1.2375007, -0.5114486, 0.7504748, -1.5904582, 1.7489493
3: -0.7972943, 1.0843289, -0.3159341, 0.7070236, -1.5043178, 1.4002630
4: -0.9476143, 0.9264617, -0.5193745, 0.5445960, -1.4922103, 1.4458362
5: -0.8505856, 0.9842122, -0.4780510, 0.6123558, -1.4629414, 1.4622631
6: -0.9166536, 0.9250562, -0.4688029, 0.5573616, -1.4740152, 1.3938591
7: -0.8893870, 0.9190227, -0.5080119, 0.5131106, -1.4024976, 1.4270346
8: -1.2514068, 2.5834684, -0.3983576, 2.4784038, -3.7298107, 2.9818261
9: -0.8991263, 1.0776665, -0.6409518, 0.6819326, -1.5810590, 1.7186184

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8428635, upper bound: 10.8376437
time: 1.80 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8343190, upper bound: 10.7893897
time: 2.04 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -1.4255254, 1.3141093, -0.5037905, 0.7061282, -2.1316538, 1.8178998
1: -1.1527662, 1.2726383, -0.4447974, 0.5560591, -1.7088253, 1.7174357
2: -1.3500524, 1.5599290, -0.5232111, 0.7811768, -2.1312292, 2.0831401
3: -1.3980827, 1.3414884, -0.3397104, 0.7273751, -2.1254578, 1.6811988
4: -1.5418810, 1.3104488, -0.5415894, 0.5670004, -2.1088815, 1.8520381
5: -1.3055727, 1.3389995, -0.5031861, 0.6309971, -1.9365698, 1.8421856
6: -1.3718890, 1.3404925, -0.4967811, 0.5787770, -1.9506660, 1.8372736
7: -1.3670820, 1.3534386, -0.5284524, 0.5368227, -1.9039047, 1.8818910
8: -2.1072142, 2.6731100, -0.4477358, 2.4939806, -4.6011949, 3.1208458
9: -1.2451329, 1.4755951, -0.6535512, 0.7089368, -1.9540697, 2.1291463

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8428591, upper bound: 10.8376156
time: 2.35 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8343190, upper bound: 10.7893897
time: 1.78 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -1.1539570, 1.1424278, -1.5330487, 1.3839092, -2.5378661, 2.6754766
1: -0.9534629, 1.0952193, -1.2376535, 1.3426378, -2.2961006, 2.3328729
2: -1.0730567, 1.4079773, -1.4728265, 1.6453143, -2.7183709, 2.8808038
3: -1.0641934, 1.2433804, -1.5225674, 1.4214885, -2.4856820, 2.7659478
4: -1.2052786, 1.1110660, -1.6608790, 1.3947498, -2.6000285, 2.7719450
5: -1.0594730, 1.1520875, -1.4006095, 1.4153426, -2.4748156, 2.5526969
6: -1.1409607, 1.1168473, -1.4704392, 1.4315405, -2.5725012, 2.5872865
7: -1.0980096, 1.1153560, -1.4716007, 1.4386718, -2.5366814, 2.5869565
8: -1.6536237, 2.6234021, -2.2811031, 2.6923523, -4.3459759, 4.9045053
9: -1.0586052, 1.2595332, -1.3222384, 1.5597086, -2.6183138, 2.5817716

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8483508, upper bound: 10.8485760
time: 3.23 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8483508, upper bound: 10.8485760
time: 14.97 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -1.6998211, 1.4999235, -1.6070671, 1.4367237, -3.1365447, 3.1069906
1: -1.3697739, 1.4731708, -1.2959552, 1.3947247, -2.7644987, 2.7691259
2: -1.6721557, 1.7450354, -1.5620586, 1.6911850, -3.3633409, 3.3070941
3: -1.7607398, 1.5318984, -1.6247263, 1.4629782, -3.2237182, 3.1566248
4: -1.8578347, 1.5297571, -1.7486813, 1.4536905, -3.3115253, 3.2784386
5: -1.5576010, 1.5371618, -1.4701506, 1.4679368, -3.0255377, 3.0073123
6: -1.6244454, 1.5821499, -1.5378141, 1.4969579, -3.1214032, 3.1199641
7: -1.6376832, 1.5973866, -1.5458288, 1.5073903, -3.1450734, 3.1432154
8: -2.5665276, 2.7248259, -2.4078951, 2.7130075, -5.2795353, 5.1327209
9: -1.4484569, 1.7079360, -1.3771007, 1.6245775, -3.0730343, 3.0850368

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8486080, upper bound: 10.8486080
time: 2.21 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8486080, upper bound: 10.8486079
time: 2.13 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.2575211, 0.4196585, -2.4291546, 2.0317860, -2.2893071, 2.8488131
1: -0.2017032, 0.3106887, -1.9634320, 2.0339324, -2.2356355, 2.2741206
2: -0.3961727, 0.3633478, -2.5970578, 2.2509356, -2.6471083, 2.9604056
3: -0.1444370, 0.3990124, -2.7956610, 2.0243235, -2.1687605, 3.1946733
4: -0.3266458, 0.2886288, -2.7249064, 2.1345065, -2.4611523, 3.0135353
5: -0.2599434, 0.3539122, -2.2676177, 2.0909963, -2.3509398, 2.6215301
6: -0.2294266, 0.3194167, -2.2966218, 2.2551649, -2.4845915, 2.6160386
7: -0.3114450, 0.2442848, -2.3987522, 2.3159165, -2.6273615, 2.6430371
8: 0.1779630, 2.3788168, -3.7967238, 2.9071598, -2.7291968, 6.1755409
9: -0.5330266, 0.4048324, -2.0650737, 2.3610132, -2.8940396, 2.4699061

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8468464, upper bound: 10.8462604
time: 2.29 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8468464, upper bound: 10.8462607
time: 4.25 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.4738432, 0.6637069, -2.7218738, 2.2674029, -2.7412462, 3.3855805
1: -0.4332883, 0.5387607, -2.2026882, 2.2508273, -2.6841156, 2.7414489
2: -0.5249767, 0.7536155, -2.9680748, 2.4644575, -2.9894342, 3.7216902
3: -0.3150466, 0.7536184, -3.2001812, 2.2345617, -2.5496082, 3.9537997
4: -0.5192333, 0.5445958, -3.0616360, 2.3884969, -2.9077301, 3.6062317
5: -0.4727543, 0.6171192, -2.5752068, 2.3094544, -2.7822087, 3.1923261
6: -0.4832434, 0.5509640, -2.5786648, 2.5248592, -3.0081027, 3.1296287
7: -0.5013899, 0.5042734, -2.7017934, 2.6075444, -3.1089344, 3.2060668
8: -0.4092767, 2.5069275, -4.2655973, 3.0004594, -3.4097362, 6.7725248
9: -0.6495362, 0.6848310, -2.3262219, 2.6222911, -3.2718272, 3.0110528

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8486966, upper bound: 10.8483946
time: 4.44 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8486966, upper bound: 10.8483954
time: 2.89 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.2642660, 0.4349587, -2.9722500, 2.4638629, -2.7281289, 3.4072087
1: -0.2101599, 0.3195322, -2.4399574, 2.4151521, -2.6253121, 2.7594895
2: -0.4006401, 0.3783835, -3.2573504, 2.6112697, -3.0119100, 3.6357338
3: -0.1491099, 0.4128262, -3.5089936, 2.3263562, -2.4754660, 3.9218199
4: -0.3352927, 0.2965561, -3.3540835, 2.5857449, -2.9210377, 3.6506395
5: -0.2671497, 0.3648174, -2.8366113, 2.4864933, -2.7536430, 3.2014287
6: -0.2372482, 0.3297240, -2.7971582, 2.7452431, -2.9824913, 3.1268823
7: -0.3195517, 0.2539203, -2.9549828, 2.8622367, -3.1817884, 3.2089031
8: 0.1522293, 2.3914852, -4.6495085, 3.0799739, -2.9277446, 7.0409937
9: -0.5384822, 0.4174856, -2.5472462, 2.8352566, -3.3737388, 2.9647317

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8468464, upper bound: 10.8462604
time: 8.66 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8468464, upper bound: 10.8462607
time: 2.16 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.4943049, 0.6853105, -3.2790587, 2.7206635, -3.2149684, 3.9643693
1: -0.4506135, 0.5593430, -2.7336693, 2.6439238, -3.0945373, 3.2930121
2: -0.5361349, 0.7744700, -3.6358156, 2.8420601, -3.3781950, 4.4102855
3: -0.3321071, 0.7730864, -3.9254324, 2.5405338, -2.8726408, 4.6985188
4: -0.5345002, 0.5656791, -3.7307603, 2.8470125, -3.3815126, 4.2964392
5: -0.4922767, 0.6340967, -3.1508033, 2.7318053, -3.2240820, 3.7849002
6: -0.5071580, 0.5686703, -3.1005118, 3.0317624, -3.5389204, 3.6691821
7: -0.5200371, 0.5223972, -3.2698607, 3.1704054, -3.6904426, 3.7922580
8: -0.4476144, 2.5220680, -5.1251860, 3.1934137, -3.6410282, 7.6472540
9: -0.6616191, 0.7042855, -2.8329263, 3.1165817, -3.7782009, 3.5372119

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8486967, upper bound: 10.8483947
time: 2.63 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8486967, upper bound: 10.8483955
time: 3.73 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.3169972, 0.5268811, -2.2417250, 1.8800881, -2.1970854, 2.7686062
1: -0.2687016, 0.3807785, -1.8063819, 1.8945791, -2.1632807, 2.1871605
2: -0.4377053, 0.5163553, -2.3548377, 2.1188188, -2.5565240, 2.8711929
3: -0.1915580, 0.4967713, -2.5336936, 1.9026011, -2.0941591, 3.0304649
4: -0.3947687, 0.3559047, -2.5002556, 1.9738164, -2.3685851, 2.8561602
5: -0.3189638, 0.4429646, -2.0674281, 1.9489492, -2.2679129, 2.5103927
6: -0.2955890, 0.3955885, -2.1204534, 2.0809491, -2.3765380, 2.5160420
7: -0.3710942, 0.3298102, -2.1986904, 2.1252694, -2.4963636, 2.5285006
8: -0.0447130, 2.4464045, -3.4842675, 2.8575621, -2.9022751, 5.9306717
9: -0.5681500, 0.5083965, -1.8931810, 2.1933672, -2.7615173, 2.4015775

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8421558, upper bound: 10.8428727
time: 1.66 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8421558, upper bound: 10.8448464
time: 1.87 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.9094036, 1.0141072, -2.5225005, 2.1098797, -3.0192833, 3.5366077
1: -0.7756825, 0.9150221, -2.0413618, 2.1047401, -2.8804226, 2.9563839
2: -0.8323802, 1.2212851, -2.7197406, 2.3279500, -3.1603303, 3.9410257
3: -0.7899684, 1.0634568, -2.9332042, 2.1069090, -2.8968773, 3.9966609
4: -0.9405079, 0.9217929, -2.8312867, 2.2188098, -3.1593177, 3.7530794
5: -0.8443104, 0.9769226, -2.3654323, 2.1633441, -3.0076547, 3.3423548
6: -0.9074525, 0.9220220, -2.3918166, 2.3435955, -3.2510481, 3.3138385
7: -0.8840309, 0.9130273, -2.4955566, 2.4071541, -3.2911849, 3.4085839
8: -1.2448273, 2.6311197, -3.9494162, 2.9412932, -4.1861205, 6.5805359
9: -0.8968158, 1.0759728, -2.1502872, 2.4476986, -3.3445144, 3.2262599

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8458597, upper bound: 10.8463310
time: 2.90 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8458597, upper bound: 10.8463310
time: 2.95 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -1.1878470, 1.1620940, -3.1085410, 2.5798502, -3.7676973, 4.2706351
1: -0.9776375, 1.1192989, -2.5688577, 2.5182109, -3.4958484, 3.6881566
2: -1.1056038, 1.4304458, -3.4269943, 2.7203400, -3.8259439, 4.8574400
3: -1.1022657, 1.2642536, -3.6999130, 2.4354820, -3.5377479, 4.9641666
4: -1.2443480, 1.1364033, -3.5184467, 2.7036831, -3.9480312, 4.6548500
5: -1.0894648, 1.1762786, -2.9756784, 2.5970242, -3.6864891, 4.1519570
6: -1.1714327, 1.1446024, -2.9361157, 2.8734560, -4.0448885, 4.0807180
7: -1.1296177, 1.1423111, -3.0927870, 2.9969969, -4.1266146, 4.2350979
8: -1.7103063, 2.6281688, -4.8606062, 3.1348724, -4.8451786, 7.4887753
9: -1.0821081, 1.2856536, -2.6750872, 2.9616559, -4.0437641, 3.9607408

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8458597, upper bound: 10.8463310
time: 3.75 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8489285, upper bound: 10.8485979
time: 13.29 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -1.7374430, 1.5260094, -3.1085410, 2.5798502, -4.3172932, 4.6345506
1: -1.3993523, 1.5011034, -2.5688577, 2.5182109, -3.9175632, 4.0699611
2: -1.7178857, 1.7695702, -3.4269943, 2.7203400, -4.4382257, 5.1965647
3: -1.8137107, 1.5556287, -3.6999130, 2.4354820, -4.2491927, 5.2555418
4: -1.9023755, 1.5597622, -3.5184467, 2.7036831, -4.6060586, 5.0782089
5: -1.5925713, 1.5645006, -2.9756784, 2.5970242, -4.1895952, 4.5401793
6: -1.6587183, 1.6159592, -2.9361157, 2.8734560, -4.5321741, 4.5520749
7: -1.6752318, 1.6330019, -3.0927870, 2.9969969, -4.6722288, 4.7257891
8: -2.6300566, 2.7313948, -4.8606062, 3.1348724, -5.7649288, 7.5920010
9: -1.4765049, 1.7409468, -2.6750872, 2.9616559, -4.4381609, 4.4160337

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8458597, upper bound: 10.8463310
time: 4.39 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8489285, upper bound: 10.8485977
time: 2.99 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.3234944, 0.5258511, -1.0979109, 1.0934272, -1.4169216, 1.6237620
1: -0.2821373, 0.3912882, -0.9127841, 1.0435810, -1.3257183, 1.3040723
2: -0.4434229, 0.4876702, -0.9851624, 1.2707365, -1.7141594, 1.4728326
3: -0.1950404, 0.5329831, -1.0364298, 1.1718731, -1.3669136, 1.5694129
4: -0.3996407, 0.3672229, -1.1655684, 1.0673933, -1.4670340, 1.5327913
5: -0.3290858, 0.4537344, -1.0173315, 1.0980639, -1.4271498, 1.4710659
6: -0.3037047, 0.4039551, -1.0496970, 1.0650580, -1.3687627, 1.4536521
7: -0.3785180, 0.3386521, -1.0687828, 1.0783980, -1.4569160, 1.4074349
8: -0.0277395, 2.4787157, -1.5155067, 2.5159533, -2.5436928, 3.9942224
9: -0.5798796, 0.5102180, -1.0012952, 1.2070904, -1.7869700, 1.5115132

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 92

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8420793, upper bound: 10.8319649
time: 1.91 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8420793, upper bound: 10.8458737
time: 2.23 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.6921540, 0.8546086, -1.0979109, 1.0934272, -1.7855811, 1.9525194
1: -0.6203120, 0.7469473, -0.9127841, 1.0435810, -1.6638930, 1.6597314
2: -0.6455306, 0.9957517, -0.9851624, 1.2707365, -1.9162670, 1.9809141
3: -0.5479308, 0.9487576, -1.0364298, 1.1718731, -1.7198039, 1.9851874
4: -0.7168100, 0.7541748, -1.1655684, 1.0673933, -1.7842033, 1.9197431
5: -0.6766599, 0.7997634, -1.0173315, 1.0980639, -1.7747238, 1.8170949
6: -0.7078207, 0.7338922, -1.0496970, 1.0650580, -1.7728786, 1.7835892
7: -0.7013257, 0.7139036, -1.0687828, 1.0783980, -1.7797236, 1.7826865
8: -0.8375096, 2.6201806, -1.5155067, 2.5159533, -3.3534629, 4.1356874
9: -0.7656823, 0.8946528, -1.0012952, 1.2070904, -1.9727727, 1.8959479

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 92

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8420793, upper bound: 10.8342182
time: 1.71 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8420793, upper bound: 10.8470110
time: 1.88 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.3234944, 0.5258511, -2.5966496, 2.1585758, -2.4820702, 3.1225009
1: -0.2821373, 0.3912882, -2.1120830, 2.1464112, -2.4285486, 2.5033712
2: -0.4434229, 0.4876702, -2.8143430, 2.3446460, -2.7880688, 3.3020132
3: -0.1950404, 0.5329831, -3.0527534, 2.1355460, -2.3305864, 3.5857365
4: -0.3996407, 0.3672229, -2.9394550, 2.2805462, -2.6801867, 3.3066781
5: -0.3290858, 0.4537344, -2.4537578, 2.2213931, -2.5504789, 2.9074922
6: -0.3037047, 0.4039551, -2.4424789, 2.4157839, -2.7194886, 2.8464341
7: -0.3785180, 0.3386521, -2.5893219, 2.5065994, -2.8851173, 2.9279740
8: -0.0277395, 2.4787157, -4.0603104, 2.8734493, -2.9011889, 6.5390263
9: -0.5798796, 0.5102180, -2.2142806, 2.5142438, -3.0941234, 2.7244987

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8420793, upper bound: 10.8466105
time: 1.92 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8420793, upper bound: 10.8458737
time: 3.41 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.6921540, 0.8546086, -2.5966496, 2.1585758, -2.8507297, 3.4512582
1: -0.6203120, 0.7469473, -2.1120830, 2.1464112, -2.7667232, 2.8590302
2: -0.6455306, 0.9957517, -2.8143430, 2.3446460, -2.9901767, 3.8100948
3: -0.5479308, 0.9487576, -3.0527534, 2.1355460, -2.6834769, 4.0015111
4: -0.7168100, 0.7541748, -2.9394550, 2.2805462, -2.9973562, 3.6936297
5: -0.6766599, 0.7997634, -2.4537578, 2.2213931, -2.8980529, 3.2535212
6: -0.7078207, 0.7338922, -2.4424789, 2.4157839, -3.1236045, 3.1763711
7: -0.7013257, 0.7139036, -2.5893219, 2.5065994, -3.2079251, 3.3032255
8: -0.8375096, 2.6201806, -4.0603104, 2.8734493, -3.7109590, 6.6804910
9: -0.7656823, 0.8946528, -2.2142806, 2.5142438, -3.2799263, 3.1089334

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8420793, upper bound: 10.8319649
time: 2.08 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8420793, upper bound: 10.8488442
time: 3.12 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -1.5578934, 1.3989991, -0.9340247, 0.9944582, -2.5523515, 2.3330238
1: -1.2618370, 1.3827186, -0.7952818, 0.9275866, -2.1894236, 2.1780005
2: -1.5017576, 1.6508577, -0.8229578, 1.1561425, -2.6579001, 2.4738154
3: -1.5624310, 1.4399718, -0.8428412, 1.0846390, -2.6470699, 2.2828131
4: -1.6965367, 1.4165334, -0.9737825, 0.9441838, -2.6407204, 2.3903160
5: -1.4286734, 1.4371059, -0.8685680, 0.9805970, -2.4092703, 2.3056738
6: -1.4922395, 1.4588895, -0.8996621, 0.9299837, -2.4222231, 2.3585515
7: -1.4966750, 1.4694895, -0.9173845, 0.9379766, -2.4346516, 2.3868740
8: -2.3378210, 2.7279415, -1.2290571, 2.4945145, -4.8323355, 3.9569986
9: -1.3448328, 1.5885056, -0.8926961, 1.0754068, -2.4202394, 2.4812016

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 92

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8432181, upper bound: 10.8346958
time: 2.08 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8432181, upper bound: 10.8472705
time: 1.85 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -1.9858575, 1.7051694, -0.9340247, 0.9944582, -2.9803157, 2.6391940
1: -1.5983520, 1.6872733, -0.7952818, 0.9275866, -2.5259385, 2.4825549
2: -2.0188963, 1.9212224, -0.8229578, 1.1561425, -3.1750388, 2.7441802
3: -2.1541657, 1.6733425, -0.8428412, 1.0846390, -3.2388048, 2.5161836
4: -2.2103035, 1.7545812, -0.9737825, 0.9441838, -3.1544874, 2.7283638
5: -1.8300750, 1.7449013, -0.8685680, 0.9805970, -2.8106720, 2.6134694
6: -1.8789757, 1.8425492, -0.8996621, 0.9299837, -2.8089595, 2.7422113
7: -1.9337867, 1.8736417, -0.9173845, 0.9379766, -2.8717632, 2.7910261
8: -3.0687926, 2.8252709, -1.2290571, 2.4945145, -5.5633068, 4.0543280
9: -1.6697447, 1.9629823, -0.8926961, 1.0754068, -2.7451515, 2.8556786

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 92

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8432181, upper bound: 10.8346958
time: 2.09 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8432181, upper bound: 10.8472705
time: 2.09 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.4275523, 0.6417869, -2.4267495, 2.0198803, -2.4474325, 3.0685363
1: -0.3806729, 0.4716321, -1.9718368, 2.0182757, -2.3989487, 2.4434688
2: -0.4971472, 0.6613703, -2.5947704, 2.2246590, -2.7218060, 3.2561407
3: -0.2692680, 0.6543707, -2.8192763, 2.0245657, -2.2938337, 3.4736471
4: -0.4778625, 0.4907953, -2.7369566, 2.1327066, -2.6105692, 3.2277520
5: -0.4241970, 0.5680022, -2.2742004, 2.0928924, -2.5170894, 2.8422027
6: -0.4117333, 0.5161842, -2.2811627, 2.2566981, -2.6684313, 2.7973471
7: -0.4615071, 0.4616155, -2.4087429, 2.3337183, -2.7952254, 2.8703585
8: -0.2946862, 2.5462341, -3.7808492, 2.8222334, -3.1169195, 6.3270836
9: -0.6312777, 0.6300566, -2.0605507, 2.3612871, -2.9925647, 2.6906073

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8421195, upper bound: 10.8458597
time: 1.86 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8421195, upper bound: 10.8461989
time: 2.41 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -1.4150589, 1.3209132, -2.4267495, 2.0198803, -3.4349391, 3.7476625
1: -1.1497474, 1.2640007, -1.9718368, 2.0182757, -3.1680231, 3.2358375
2: -1.3367178, 1.5492901, -2.5947704, 2.2246590, -3.5613768, 4.1440606
3: -1.3863975, 1.3404584, -2.8192763, 2.0245657, -3.4109631, 4.1597347
4: -1.5242915, 1.3071234, -2.7369566, 2.1327066, -3.6569982, 4.0440798
5: -1.2997463, 1.3329661, -2.2742004, 2.0928924, -3.3926387, 3.6071665
6: -1.3660018, 1.3360178, -2.2811627, 2.2566981, -3.6227000, 3.6171806
7: -1.3540499, 1.3401821, -2.4087429, 2.3337183, -3.6877682, 3.7489250
8: -2.1067986, 2.7567573, -3.7808492, 2.8222334, -4.9290323, 6.5376062
9: -1.2408161, 1.4741616, -2.0605507, 2.3612871, -3.6021032, 3.5347123

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8421195, upper bound: 10.8489283
time: 2.12 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8421195, upper bound: 10.8489861
time: 2.19 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.3370145, 0.5414453, -2.1601596, 1.8124235, -2.1494379, 2.7016048
1: -0.2955642, 0.4011110, -1.7585046, 1.8220448, -2.1176090, 2.1596155
2: -0.4501714, 0.5061964, -2.2173052, 1.9818923, -2.4320638, 2.7235017
3: -0.2037310, 0.5499586, -2.4596715, 1.8083159, -2.0120468, 3.0096302
4: -0.4088288, 0.3796670, -2.4384351, 1.9059786, -2.3148074, 2.8181021
5: -0.3407745, 0.4696174, -2.0149121, 1.8837928, -2.2245672, 2.4845295
6: -0.3152050, 0.4196144, -2.0220389, 2.0094342, -2.3246393, 2.4416533
7: -0.3888828, 0.3539717, -2.1331329, 2.0722373, -2.4611201, 2.4871047
8: -0.0584546, 2.4912453, -3.3433690, 2.7329535, -2.7914081, 5.8346143
9: -0.5861121, 0.5254810, -1.8261173, 2.1203527, -2.7064648, 2.3515983

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8469938, upper bound: 10.8467808
time: 2.96 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8469938, upper bound: 10.8471087
time: 2.48 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.7270169, 0.8864459, -2.4097519, 2.0082545, -2.7352715, 3.2961979
1: -0.6505847, 0.7777827, -1.9671695, 2.0113833, -2.6619680, 2.7449522
2: -0.6715032, 1.0305731, -2.5468855, 2.1607265, -2.8322296, 3.5774586
3: -0.5888312, 0.9761503, -2.8130541, 1.9895557, -2.5783868, 3.7892044
4: -0.7519388, 0.7874410, -2.7296867, 2.1221461, -2.8740849, 3.5171278
5: -0.7072436, 0.8324199, -2.2761159, 2.0770681, -2.7843118, 3.1085358
6: -0.7420080, 0.7646109, -2.2589874, 2.2444642, -2.9864721, 3.0235982
7: -0.7354411, 0.7506222, -2.3975391, 2.3261404, -3.0615816, 3.1481614
8: -0.9097078, 2.6354394, -3.7616780, 2.8045831, -3.7142909, 6.3971176
9: -0.7886221, 0.9274369, -2.0514262, 2.3473296, -3.1359518, 2.9788632

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8482874, upper bound: 10.8487944
time: 2.16 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8482874, upper bound: 10.8488571
time: 2.54 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.3370145, 0.5414453, -3.0386350, 2.5135355, -2.8505499, 3.5800803
1: -0.2955642, 0.4011110, -2.5092409, 2.4583368, -2.7539010, 2.9103518
2: -0.4501714, 0.5061964, -3.3332374, 2.6187592, -3.0689306, 3.8394337
3: -0.2037310, 0.5499586, -3.6138206, 2.3476174, -2.5513484, 4.1637793
4: -0.4088288, 0.3796670, -3.4612849, 2.6390440, -3.0478728, 3.8409519
5: -0.3407745, 0.4696174, -2.9203377, 2.5409262, -2.8817008, 3.3899550
6: -0.3152050, 0.4196144, -2.8485799, 2.8103848, -3.1255898, 3.2681942
7: -0.3888828, 0.3539717, -3.0364423, 2.9579546, -3.3468375, 3.3904140
8: -0.0584546, 2.4912453, -4.7519040, 3.0232227, -3.0816774, 7.2431493
9: -0.5861121, 0.5254810, -2.6067936, 2.9018126, -3.4879246, 3.1322746

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8467314, upper bound: 10.8466105
time: 2.00 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8467314, upper bound: 10.8470437
time: 2.17 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.7270169, 0.8864459, -3.3645344, 2.7877715, -3.5147884, 4.2509804
1: -0.6505847, 0.7777827, -2.8215401, 2.7013073, -3.3518920, 3.5993228
2: -0.6715032, 1.0305731, -3.7351890, 2.8634391, -3.5349422, 4.7657623
3: -0.5888312, 0.9761503, -4.0592470, 2.5731251, -3.1619563, 5.0353975
4: -0.7519388, 0.7874410, -3.8636527, 2.9173336, -3.6692724, 4.6510935
5: -0.7072436, 0.8324199, -3.2543433, 2.8017199, -3.5089636, 4.0867634
6: -0.7420080, 0.7646109, -3.1720915, 3.1157739, -3.8577819, 3.9367023
7: -0.7354411, 0.7506222, -3.3717337, 3.2867351, -4.0221763, 4.1223559
8: -0.9097078, 2.6354394, -5.2545137, 3.1491246, -4.0588322, 7.8899531
9: -0.7886221, 0.9274369, -2.9112771, 3.2012157, -3.9898379, 3.8387141

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8484940, upper bound: 10.8487763
time: 2.86 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8484940, upper bound: 10.8488471
time: 9.43 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -1.8649862, 1.6123476, -2.4728999, 2.0647445, -3.9297307, 4.0852475
1: -1.5037171, 1.6116217, -2.0001447, 2.0473304, -3.5510473, 3.6117663
2: -1.8773242, 1.8609242, -2.6507483, 2.2812443, -4.1585684, 4.5116725
3: -1.9987080, 1.6535308, -2.8546143, 2.0454047, -4.0441127, 4.5081453
4: -2.0529852, 1.6643445, -2.7708211, 2.1708169, -4.2238021, 4.4351654
5: -1.7111468, 1.6649787, -2.3094630, 2.1154160, -3.8265629, 3.9744418
6: -1.7759657, 1.7359195, -2.3405354, 2.2895494, -4.0655150, 4.0764551
7: -1.8016112, 1.7584584, -2.4443541, 2.3522027, -4.1538138, 4.2028122
8: -2.8555021, 2.7882047, -3.8514762, 2.8919880, -5.7474899, 6.6396809
9: -1.5731103, 1.8614441, -2.0990751, 2.3915987, -3.9647090, 3.9605193

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8448464, upper bound: 10.8459066
time: 1.73 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8485977, upper bound: 10.8489285
time: 2.96 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -2.3064961, 1.9402096, -2.4728999, 2.0647445, -4.3712406, 4.4131098
1: -1.8644428, 1.9259710, -2.0001447, 2.0473304, -3.9117732, 3.9261158
2: -2.4301052, 2.1443124, -2.6507483, 2.2812443, -4.7113495, 4.7950606
3: -2.6115966, 1.9018739, -2.8546143, 2.0454047, -4.6570015, 4.7564883
4: -2.5799150, 2.0248418, -2.7708211, 2.1708169, -4.7507319, 4.7956629
5: -2.1414781, 1.9897954, -2.3094630, 2.1154160, -4.2568941, 4.2992582
6: -2.1803813, 2.1379731, -2.3405354, 2.2895494, -4.4699306, 4.4785085
7: -2.2687497, 2.1878359, -2.4443541, 2.3522027, -4.6209526, 4.6321898
8: -3.6028047, 2.9067030, -3.8514762, 2.8919880, -6.4947929, 6.7581792
9: -1.9549139, 2.2488258, -2.0990751, 2.3915987, -4.3465128, 4.3479009

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8448464, upper bound: 10.8459066
time: 1.94 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8485977, upper bound: 10.8489284
time: 4.92 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -1.8649862, 1.6123476, -3.1085410, 2.5798502, -4.4448366, 4.7208886
1: -1.5037171, 1.6116217, -2.5688577, 2.5182109, -4.0219278, 4.1804795
2: -1.8773242, 1.8609242, -3.4269943, 2.7203400, -4.5976644, 5.2879186
3: -1.9987080, 1.6535308, -3.6999130, 2.4354820, -4.4341898, 5.3534441
4: -2.0529852, 1.6643445, -3.5184467, 2.7036831, -4.7566681, 5.1827912
5: -1.7111468, 1.6649787, -2.9756784, 2.5970242, -4.3081713, 4.6406574
6: -1.7759657, 1.7359195, -2.9361157, 2.8734560, -4.6494217, 4.6720352
7: -1.8016112, 1.7584584, -3.0927870, 2.9969969, -4.7986078, 4.8512454
8: -2.8555021, 2.7882047, -4.8606062, 3.1348724, -5.9903746, 7.6488109
9: -1.5731103, 1.8614441, -2.6750872, 2.9616559, -4.5347662, 4.5365314

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8461986, upper bound: 10.8472705
time: 2.50 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8489862, upper bound: 10.8489862
time: 1.88 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -2.3064961, 1.9402096, -3.1085410, 2.5798502, -4.8863463, 5.0487509
1: -1.8644428, 1.9259710, -2.5688577, 2.5182109, -4.3826537, 4.4948287
2: -2.4301052, 2.1443124, -3.4269943, 2.7203400, -5.1504450, 5.5713067
3: -2.6115966, 1.9018739, -3.6999130, 2.4354820, -5.0470786, 5.6017871
4: -2.5799150, 2.0248418, -3.5184467, 2.7036831, -5.2835979, 5.5432882
5: -2.1414781, 1.9897954, -2.9756784, 2.5970242, -4.7385025, 4.9654741
6: -2.1803813, 2.1379731, -2.9361157, 2.8734560, -5.0538373, 5.0740891
7: -2.2687497, 2.1878359, -3.0927870, 2.9969969, -5.2657466, 5.2806230
8: -3.6028047, 2.9067030, -4.8606062, 3.1348724, -6.7376771, 7.7673092
9: -1.9549139, 2.2488258, -2.6750872, 2.9616559, -4.9165697, 4.9239130

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8461986, upper bound: 10.8472705
time: 3.91 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8489862, upper bound: 10.8489862
time: 1.92 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 7.30 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8445362, upper bound: 10.8458287
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8445362, upper bound: 10.8459641
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8445362, upper bound: 10.8458287
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8445362, upper bound: 10.8459641
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8438410, upper bound: 10.8464190
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8438410, upper bound: 10.8464193
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8438410, upper bound: 10.8446888
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8438410, upper bound: 10.8484082
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8428635, upper bound: 10.8376437
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8343190, upper bound: 10.7893897
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8428591, upper bound: 10.8376156
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8343190, upper bound: 10.7893897
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8483508, upper bound: 10.8485760
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8483508, upper bound: 10.8485760
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8486080, upper bound: 10.8486080
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8486080, upper bound: 10.8486079
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8468464, upper bound: 10.8462604
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8468464, upper bound: 10.8462607
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8486966, upper bound: 10.8483946
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8486966, upper bound: 10.8483954
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8468464, upper bound: 10.8462604
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8468464, upper bound: 10.8462607
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8486967, upper bound: 10.8483947
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8486967, upper bound: 10.8483955
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8421558, upper bound: 10.8428727
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8421558, upper bound: 10.8448464
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8458597, upper bound: 10.8463310
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8458597, upper bound: 10.8463310
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8458597, upper bound: 10.8463310
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8489285, upper bound: 10.8485979
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8458597, upper bound: 10.8463310
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8489285, upper bound: 10.8485977
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8420793, upper bound: 10.8319649
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8420793, upper bound: 10.8458737
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8420793, upper bound: 10.8342182
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8420793, upper bound: 10.8470110
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8420793, upper bound: 10.8466105
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8420793, upper bound: 10.8458737
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8420793, upper bound: 10.8319649
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8420793, upper bound: 10.8488442
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8432181, upper bound: 10.8346958
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8432181, upper bound: 10.8472705
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8432181, upper bound: 10.8346958
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8432181, upper bound: 10.8472705
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8421195, upper bound: 10.8458597
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8421195, upper bound: 10.8461989
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8421195, upper bound: 10.8489283
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8421195, upper bound: 10.8489861
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8469938, upper bound: 10.8467808
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8469938, upper bound: 10.8471087
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8482874, upper bound: 10.8487944
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8482874, upper bound: 10.8488571
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8467314, upper bound: 10.8466105
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8467314, upper bound: 10.8470437
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8484940, upper bound: 10.8487763
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8484940, upper bound: 10.8488471
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8448464, upper bound: 10.8459066
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8485977, upper bound: 10.8489285
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8448464, upper bound: 10.8459066
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8485977, upper bound: 10.8489284
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8461986, upper bound: 10.8472705
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8489862, upper bound: 10.8489862
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8461986, upper bound: 10.8472705
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.30
Output dim: 8, lower bound: -10.8489862, upper bound: 10.8489862

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.5030130, 0.6738580, -0.2575211, 0.4196585, -0.9226716, 0.9313792
1: -0.4618849, 0.5729823, -0.2017032, 0.3106887, -0.7725736, 0.7746855
2: -0.5341837, 0.7911453, -0.3961727, 0.3633478, -0.8975315, 1.1873180
3: -0.3394054, 0.7994530, -0.1444370, 0.3990124, -0.7384177, 0.9438899
4: -0.5410276, 0.5724103, -0.3266458, 0.2886288, -0.8296564, 0.8990562
5: -0.4987844, 0.6418643, -0.2599434, 0.3539122, -0.8526966, 0.9018077
6: -0.5178655, 0.5697587, -0.2294266, 0.3194167, -0.8372822, 0.7991852
7: -0.5264295, 0.5282149, -0.3114450, 0.2442848, -0.7707143, 0.8396599
8: -0.4561309, 2.4688973, 0.1779630, 2.3788168, -2.8349478, 2.2909343
9: -0.6557375, 0.7080813, -0.5330266, 0.4048324, -1.0605700, 1.2411079

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8332557, upper bound: 10.7832568
time: 2.17 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8329046, upper bound: 10.7823821
time: 3.25 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.5030130, 0.6738580, -0.3151550, 0.5234368, -1.0264498, 0.9890131
1: -0.4618849, 0.5729823, -0.2665668, 0.3788257, -0.8407106, 0.8395491
2: -0.5341837, 0.7911453, -0.4368146, 0.5135902, -1.0477740, 1.2279599
3: -0.3394054, 0.7994530, -0.1898467, 0.4942394, -0.8336447, 0.9892997
4: -0.5410276, 0.5724103, -0.3929656, 0.3535688, -0.8945964, 0.9653760
5: -0.4987844, 0.6418643, -0.3171511, 0.4397398, -0.9385241, 0.9590154
6: -0.5178655, 0.5697587, -0.2934890, 0.3933810, -0.9112465, 0.8632476
7: -0.5264295, 0.5282149, -0.3693697, 0.3265849, -0.8530144, 0.8975847
8: -0.4561309, 2.4688973, -0.0397803, 2.4460194, -2.9021504, 2.5086777
9: -0.6557375, 0.7080813, -0.5674683, 0.5056182, -1.1613557, 1.2755497

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8332557, upper bound: 10.7892604
time: 2.09 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8329046, upper bound: 10.7874221
time: 3.14 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.7561775, 0.8877184, -0.2642660, 0.4349587, -1.1911361, 1.1519845
1: -0.6655762, 0.8008986, -0.2101599, 0.3195322, -0.9851084, 1.0110584
2: -0.6844483, 1.0620499, -0.4006401, 0.3783835, -1.0628318, 1.4626900
3: -0.6218578, 0.9901599, -0.1491099, 0.4128262, -1.0346839, 1.1392698
4: -0.7793264, 0.8067955, -0.3352927, 0.2965561, -1.0758826, 1.1420882
5: -0.7249805, 0.8542902, -0.2671497, 0.3648174, -1.0897980, 1.1214399
6: -0.7602545, 0.7828797, -0.2372482, 0.3297240, -1.0899785, 1.0201278
7: -0.7599026, 0.7770091, -0.3195517, 0.2539203, -1.0138228, 1.0965608
8: -0.9422535, 2.5381758, 0.1522293, 2.3914852, -3.3337388, 2.3859465
9: -0.7969270, 0.9404777, -0.5384822, 0.4174856, -1.2144126, 1.4789599

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8332557, upper bound: 10.7832568
time: 2.17 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8329041, upper bound: 10.7823784
time: 2.55 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.7561775, 0.8877184, -0.3252241, 0.5387346, -1.2949121, 1.2129426
1: -0.6655762, 0.8008986, -0.2777374, 0.3883377, -1.0539140, 1.0786359
2: -0.6844483, 1.0620499, -0.4437530, 0.5286155, -1.2130637, 1.5058029
3: -0.6218578, 0.9901599, -0.1982427, 0.5082079, -1.1300657, 1.1884027
4: -0.7793264, 0.8067955, -0.4019126, 0.3652118, -1.1445382, 1.2087080
5: -0.7249805, 0.8542902, -0.3259960, 0.4551302, -1.1801107, 1.1802862
6: -0.7602545, 0.7828797, -0.3045951, 0.4036756, -1.1639301, 1.0874748
7: -0.7599026, 0.7770091, -0.3775733, 0.3415163, -1.1014190, 1.1545824
8: -0.9422535, 2.5381758, -0.0659280, 2.4597492, -3.4020028, 2.6041038
9: -0.7969270, 0.9404777, -0.5740118, 0.5194492, -1.3163762, 1.5144894

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8332557, upper bound: 10.7892604
time: 2.17 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8329041, upper bound: 10.7873947
time: 3.51 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.2649682, 0.4365217, -0.4962973, 0.6877433, -0.9527115, 0.9328191
1: -0.2110203, 0.3204181, -0.4526193, 0.5614443, -0.7724646, 0.7730373
2: -0.4011096, 0.3798940, -0.5372895, 0.7767532, -1.1778628, 0.9171835
3: -0.1496294, 0.4142324, -0.3340729, 0.7750189, -0.9246483, 0.7483053
4: -0.3361644, 0.2973544, -0.5360340, 0.5677570, -0.9039213, 0.8333884
5: -0.2678746, 0.3659123, -0.4944470, 0.6357657, -0.9036404, 0.8603593
6: -0.2380511, 0.3307576, -0.5096372, 0.5704117, -0.8084629, 0.8403948
7: -0.3203646, 0.2548853, -0.5219655, 0.5243276, -0.8446922, 0.7768508
8: 0.1495964, 2.3928282, -0.4517331, 2.5235822, -2.3739858, 2.8445613
9: -0.5390543, 0.4187671, -0.6627966, 0.7065476, -1.2456019, 1.0815637

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 218

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8450793, upper bound: 10.8450742
time: 2.02 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8119703, upper bound: 10.8385244
time: 2.11 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.2649682, 0.4365217, -0.9760764, 1.0569754, -1.3219435, 1.4125981
1: -0.2110203, 0.3204181, -0.8229526, 0.9603311, -1.1713514, 1.1433706
2: -0.4011096, 0.3798940, -0.8983625, 1.2647631, -1.6658727, 1.2782565
3: -0.1496294, 0.4142324, -0.8695257, 1.0978384, -1.2474678, 1.2837580
4: -0.3361644, 0.2973544, -1.0113614, 0.9743018, -1.3104662, 1.3087158
5: -0.2678746, 0.3659123, -0.9018834, 1.0239484, -1.2918230, 1.2677957
6: -0.2380511, 0.3307576, -0.9707808, 0.9743237, -1.2123749, 1.3015385
7: -0.3203646, 0.2548853, -0.9428638, 0.9696788, -1.2900434, 1.1977491
8: 0.1495964, 2.3928282, -1.3572345, 2.6502719, -2.5006754, 3.7500627
9: -0.5390543, 0.4187671, -0.9401914, 1.1280645, -1.6671188, 1.3589585

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8450793, upper bound: 10.8451052
time: 1.99 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8119703, upper bound: 10.8385248
time: 1.94 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.4962973, 0.6877433, -0.4962973, 0.6877433, -1.1840407, 1.1840407
1: -0.4526193, 0.5614443, -0.4526193, 0.5614443, -1.0140636, 1.0140636
2: -0.5372895, 0.7767532, -0.5372895, 0.7767532, -1.3140427, 1.3140427
3: -0.3340729, 0.7750189, -0.3340729, 0.7750189, -1.1090918, 1.1090918
4: -0.5360340, 0.5677570, -0.5360340, 0.5677570, -1.1037910, 1.1037910
5: -0.4944470, 0.6357657, -0.4944470, 0.6357657, -1.1302127, 1.1302127
6: -0.5096372, 0.5704117, -0.5096372, 0.5704117, -1.0800489, 1.0800489
7: -0.5219655, 0.5243276, -0.5219655, 0.5243276, -1.0462930, 1.0462930
8: -0.4517331, 2.5235822, -0.4517331, 2.5235822, -2.9753153, 2.9753153
9: -0.6627966, 0.7065476, -0.6627966, 0.7065476, -1.3693441, 1.3693441

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8473589, upper bound: 10.8480176
time: 9.98 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8473275, upper bound: 10.8474108
time: 4.32 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.4962973, 0.6877433, -0.9760764, 1.0569754, -1.5532727, 1.6638197
1: -0.4526193, 0.5614443, -0.8229526, 0.9603311, -1.4129504, 1.3843968
2: -0.5372895, 0.7767532, -0.8983625, 1.2647631, -1.8020526, 1.6751157
3: -0.3340729, 0.7750189, -0.8695257, 1.0978384, -1.4319113, 1.6445446
4: -0.5360340, 0.5677570, -1.0113614, 0.9743018, -1.5103359, 1.5791183
5: -0.4944470, 0.6357657, -0.9018834, 1.0239484, -1.5183954, 1.5376492
6: -0.5096372, 0.5704117, -0.9707808, 0.9743237, -1.4839610, 1.5411925
7: -0.5219655, 0.5243276, -0.9428638, 0.9696788, -1.4916443, 1.4671915
8: -0.4517331, 2.5235822, -1.3572345, 2.6502719, -3.1020050, 3.8808167
9: -0.6627966, 0.7065476, -0.9401914, 1.1280645, -1.7908611, 1.6467390

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8473589, upper bound: 10.8481160
time: 2.96 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8473275, upper bound: 10.8474197
time: 3.75 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.9170872, 1.0028160, -0.3641338, 0.5569353, -1.4740225, 1.3669498
1: -0.7836363, 0.9258493, -0.3211140, 0.4235117, -1.2071481, 1.2469633
2: -0.8399833, 1.2375007, -0.4535171, 0.6156034, -1.4555868, 1.6910179
3: -0.7972943, 1.0843289, -0.2266936, 0.5746298, -1.3719240, 1.3110225
4: -0.9476143, 0.9264617, -0.4324371, 0.4061092, -1.3537235, 1.3588988
5: -0.8505856, 0.9842122, -0.3598576, 0.5033844, -1.3539701, 1.3440697
6: -0.9166536, 0.9250562, -0.3392354, 0.4456309, -1.3622845, 1.2642915
7: -0.8893870, 0.9190227, -0.4094425, 0.3898941, -1.2792811, 1.3284652
8: -1.2514068, 2.5834684, -0.1583158, 2.3895183, -3.6409249, 2.7417841
9: -0.8991263, 1.0776665, -0.5754827, 0.5625396, -1.4616659, 1.6531492

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7837721, upper bound: 10.8215915
time: 1.71 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7837721, upper bound: 10.8376437
time: 1.78 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.8352762, 0.9479830, -0.8845224, 1.1286987, -1.9639750, 1.8325055
1: -0.7237486, 0.8664033, -0.7753875, 0.9744759, -1.6982245, 1.6417909
2: -0.7602773, 1.1817780, -0.6842888, 1.2980093, -2.0582867, 1.8660667
3: -0.7000723, 1.0392785, -0.7112259, 1.0455887, -1.7456610, 1.7505045
4: -0.8585853, 0.8631239, -0.9035873, 0.9061563, -1.7647417, 1.7667112
5: -0.7834181, 0.9220630, -0.9099830, 0.9296093, -1.7130274, 1.8320460
6: -0.8405120, 0.8572311, -0.9333003, 0.9471288, -1.7876408, 1.7905314
7: -0.8194289, 0.8469766, -0.8461662, 0.9366345, -1.7560635, 1.6931429
8: -1.1141547, 2.5580266, -1.2255497, 2.4771893, -3.5913439, 3.7835763
9: -0.8489318, 1.0121214, -0.8685841, 1.1237754, -1.9727072, 1.8807056

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7783306, upper bound: 10.7783306
time: 1.52 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7783306, upper bound: 10.7893897
time: 1.52 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -1.4255254, 1.3141093, -0.3775752, 0.5749890, -2.0005145, 1.6916845
1: -1.1527662, 1.2726383, -0.3345771, 0.4356995, -1.5884657, 1.6072154
2: -1.3500524, 1.5599290, -0.4614054, 0.6345583, -1.9846107, 2.0213344
3: -1.3980827, 1.3414884, -0.2364069, 0.5929499, -1.9910326, 1.5778953
4: -1.5418810, 1.3104488, -0.4433555, 0.4241220, -1.9660029, 1.7538042
5: -1.3055727, 1.3389995, -0.3731883, 0.5192621, -1.8248348, 1.7121878
6: -1.3718890, 1.3404925, -0.3546815, 0.4616614, -1.8335505, 1.6951740
7: -1.3670820, 1.3534386, -0.4216528, 0.4073547, -1.7744367, 1.7750914
8: -2.1072142, 2.6731100, -0.1926321, 2.4019377, -4.5091519, 2.8657422
9: -1.2451329, 1.4755951, -0.5841808, 0.5791788, -1.8243117, 2.0597758

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7837721, upper bound: 10.8215915
time: 2.07 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7837721, upper bound: 10.8376156
time: 1.83 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -1.3201374, 1.2447258, -0.9241397, 1.1578227, -2.4779601, 2.1688654
1: -1.0705479, 1.1983912, -0.8113010, 1.0171117, -2.0876596, 2.0096922
2: -1.2332364, 1.4943407, -0.7341272, 1.3299041, -2.5631404, 2.2284679
3: -1.2715995, 1.2850757, -0.7479866, 1.0694591, -2.3410587, 2.0330622
4: -1.4159781, 1.2283026, -0.9494711, 0.9330984, -2.3490765, 2.1777737
5: -1.2087183, 1.2648835, -0.9558842, 0.9490806, -2.1577988, 2.2207677
6: -1.2785902, 1.2506757, -0.9658098, 1.0079092, -2.2864995, 2.2164855
7: -1.2619482, 1.2626672, -0.8852416, 0.9639153, -2.2258635, 2.1479087
8: -1.9281013, 2.6441171, -1.2907498, 2.4898496, -4.4179506, 3.9348669
9: -1.1710417, 1.3899025, -0.8852949, 1.1647358, -2.3357775, 2.2751975

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 218

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7783306, upper bound: 10.7783306
time: 2.59 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7783306, upper bound: 10.7893897
time: 1.67 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -1.1539570, 1.1424278, -0.4738432, 0.6637069, -1.8176639, 1.6162710
1: -0.9534629, 1.0952193, -0.4332883, 0.5387607, -1.4922235, 1.5285076
2: -1.0730567, 1.4079773, -0.5249767, 0.7536155, -1.8266722, 1.9329541
3: -1.0641934, 1.2433804, -0.3150466, 0.7536184, -1.8178117, 1.5584271
4: -1.2052786, 1.1110660, -0.5192333, 0.5445958, -1.7498745, 1.6302993
5: -1.0594730, 1.1520875, -0.4727543, 0.6171192, -1.6765922, 1.6248417
6: -1.1409607, 1.1168473, -0.4832434, 0.5509640, -1.6919247, 1.6000906
7: -1.0980096, 1.1153560, -0.5013899, 0.5042734, -1.6022830, 1.6167458
8: -1.6536237, 2.6234021, -0.4092767, 2.5069275, -4.1605511, 3.0326788
9: -1.0586052, 1.2595332, -0.6495362, 0.6848310, -1.7434361, 1.9090693

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8435258, upper bound: 10.8448134
time: 1.79 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8435258, upper bound: 10.8485760
time: 1.92 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -1.1539570, 1.1424278, -0.9094036, 1.0141072, -2.1680641, 2.0518312
1: -0.9534629, 1.0952193, -0.7756825, 0.9150221, -1.8684850, 1.8709018
2: -1.0730567, 1.4079773, -0.8323802, 1.2212851, -2.2943418, 2.2403574
3: -1.0641934, 1.2433804, -0.7899684, 1.0634568, -2.1276503, 2.0333488
4: -1.2052786, 1.1110660, -0.9405079, 0.9217929, -2.1270714, 2.0515738
5: -1.0594730, 1.1520875, -0.8443104, 0.9769226, -2.0363955, 1.9963979
6: -1.1409607, 1.1168473, -0.9074525, 0.9220220, -2.0629826, 2.0242996
7: -1.0980096, 1.1153560, -0.8840309, 0.9130273, -2.0110369, 1.9993869
8: -1.6536237, 2.6234021, -1.2448273, 2.6311197, -4.2847433, 3.8682294
9: -1.0586052, 1.2595332, -0.8968158, 1.0759728, -2.1345780, 2.1563489

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8435258, upper bound: 10.8450267
time: 1.95 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8435258, upper bound: 10.8486080
time: 2.22 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -1.6998211, 1.4999235, -0.8701518, 0.9640027, -2.6638238, 2.3700752
1: -1.3697739, 1.4731708, -0.7507637, 0.8928276, -2.2626014, 2.2239344
2: -1.6721557, 1.7450354, -0.7920432, 1.2218990, -2.8940549, 2.5370786
3: -1.7607398, 1.5318984, -0.7323985, 1.0737739, -2.8345137, 2.2642968
4: -1.8578347, 1.5297571, -0.8905865, 0.8910397, -2.7488744, 2.4203436
5: -1.5576010, 1.5371618, -0.8110431, 0.9510492, -2.5086503, 2.3482051
6: -1.6244454, 1.5821499, -0.8720698, 0.8862948, -2.5107403, 2.4542196
7: -1.6376832, 1.5973866, -0.8468562, 0.8726479, -2.5103312, 2.4442430
8: -2.5665276, 2.7248259, -1.1762111, 2.5650687, -5.1315966, 3.9010370
9: -1.4484569, 1.7079360, -0.8687177, 1.0376856, -2.4861426, 2.5766537

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8439516, upper bound: 10.8450940
time: 1.72 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8439516, upper bound: 10.8439516
time: 1.89 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -1.6998211, 1.4999235, -1.3883845, 1.2826481, -2.9824691, 2.8883080
1: -1.3697739, 1.4731708, -1.1242496, 1.2427707, -2.6125445, 2.5974202
2: -1.6721557, 1.7450354, -1.3088988, 1.5543575, -3.2265134, 3.0539341
3: -1.7607398, 1.5318984, -1.3433409, 1.3403285, -3.1010683, 2.8752394
4: -1.8578347, 1.5297571, -1.4892002, 1.2814103, -3.1392450, 3.0189574
5: -1.5576010, 1.5371618, -1.2672858, 1.3136685, -2.8712695, 2.8044477
6: -1.6244454, 1.5821499, -1.3400819, 1.3062483, -2.9306936, 2.9222317
7: -1.6376832, 1.5973866, -1.3278298, 1.3123034, -2.9499865, 2.9252164
8: -2.5665276, 2.7248259, -2.0351620, 2.6530733, -5.2196007, 4.7599878
9: -1.4484569, 1.7079360, -1.2165411, 1.4407532, -2.8892102, 2.9244771

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8439516, upper bound: 10.8450940
time: 1.93 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8439516, upper bound: 10.8486078
time: 4.04 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.2575211, 0.4196585, -0.8030262, 0.9194499, -1.1769711, 1.2226847
1: -0.2017032, 0.3106887, -0.7108005, 0.8483313, -1.0500344, 1.0214891
2: -0.3961727, 0.3633478, -0.7292405, 1.1010319, -1.4972045, 1.0925882
3: -0.1444370, 0.3990124, -0.6778525, 1.0508845, -1.1953214, 1.0768650
4: -0.3266458, 0.2886288, -0.8251313, 0.8517629, -1.1784087, 1.1137601
5: -0.2599434, 0.3539122, -0.7676656, 0.8958097, -1.1557530, 1.1215779
6: -0.2294266, 0.3194167, -0.8092924, 0.8278895, -1.0573161, 1.1287091
7: -0.3114450, 0.2442848, -0.8009923, 0.8188201, -1.1302650, 1.0452771
8: 0.1779630, 2.3788168, -1.0371348, 2.5907433, -2.4127803, 3.4159517
9: -0.5330266, 0.4048324, -0.8295887, 0.9865087, -1.5195353, 1.2344211

Time for backsubstitution: 1.37 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=12.009641647338867
rel_dist={8: [-10.853321756160986, 10.853321760348969]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 92

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8517581, upper bound: 10.8514231
time: 3.81 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8518155, upper bound: 10.8518158
time: 8.03 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 11.97 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 11.97
Output dim: 8, lower bound: -10.8517581, upper bound: 10.8514231
IS_A2, status: Status.UNKNOWN, split count: 1, time: 11.97
Output dim: 8, lower bound: -10.8518155, upper bound: 10.8518158

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -2.9945498, 2.4854207, -4.2330718, 3.5171497, -6.5116997, 6.7184916
1: -2.4506655, 2.4260209, -3.6493349, 3.3437438, -5.7944093, 6.0753555
2: -3.2870402, 2.6394088, -4.7990913, 3.5056651, -6.7927041, 7.4385004
3: -3.5422220, 2.3601861, -5.2073903, 3.1388066, -6.6810284, 7.5675764
4: -3.3720222, 2.6054800, -4.9245014, 3.6462295, -7.0182514, 7.5299807
5: -2.8525395, 2.5022659, -4.1511145, 3.4900503, -6.3425889, 6.6533804
6: -2.8254833, 2.7629449, -4.0150690, 3.9169633, -6.7424469, 6.7780137
7: -2.9726825, 2.8755846, -4.2652841, 4.1514683, -7.1241508, 7.1408687
8: -4.6693630, 3.0776982, -6.5759234, 3.5634408, -8.2328033, 9.6536217
9: -2.5680308, 2.8494966, -3.7233696, 4.0015116, -6.5695424, 6.5728664

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8503477, upper bound: 10.8499491
time: 2.64 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8502464, upper bound: 10.8499610
time: 3.10 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -3.6680012, 3.0478530, -4.2152534, 3.5034325, -7.1714334, 7.2631063
1: -3.1057496, 2.9268103, -3.6337061, 3.3329256, -6.4386749, 6.5605164
2: -4.1053696, 3.1162860, -4.7789278, 3.4932246, -7.5985942, 7.8952122
3: -4.4386749, 2.7713633, -5.1904926, 3.1330976, -7.5717726, 7.9618549
4: -4.2074580, 3.1692934, -4.9043150, 3.6331391, -7.8405972, 8.0736084
5: -3.5509248, 3.0351212, -4.1341681, 3.4774535, -7.0283756, 7.1692882
6: -3.4747064, 3.3891344, -3.9999735, 3.9023623, -7.3770685, 7.3891077
7: -3.6690035, 3.5624459, -4.2484536, 4.1346374, -7.8036404, 7.8108997
8: -5.7219377, 3.3660746, -6.5484333, 3.5591998, -9.2811375, 9.9145050
9: -3.1934023, 3.4740181, -3.7089088, 3.9868102, -7.1802106, 7.1829252

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8518155, upper bound: 10.8518158
time: 2.98 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8518157, upper bound: 10.8518151
time: 3.83 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 8.22 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 8.22
Output dim: 8, lower bound: -10.8503477, upper bound: 10.8499491
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 8.22
Output dim: 8, lower bound: -10.8502464, upper bound: 10.8499610
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 8.22
Output dim: 8, lower bound: -10.8518155, upper bound: 10.8518158
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 8.22
Output dim: 8, lower bound: -10.8518157, upper bound: 10.8518151

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -2.2002499, 1.8494973, -2.1959443, 1.8494157, -4.0496655, 4.0454416
1: -1.7733535, 1.8472070, -1.7841898, 1.8482751, -3.6216285, 3.6313968
2: -2.2915225, 2.0668805, -2.2679930, 2.0180104, -4.3095331, 4.3348732
3: -2.4736643, 1.8531780, -2.5018129, 1.8370330, -4.3106976, 4.3549910
4: -2.4504862, 1.9372973, -2.4706442, 1.9357620, -4.3862481, 4.4079418
5: -2.0246263, 1.9095783, -2.0459325, 1.9088012, -3.9334273, 3.9555109
6: -2.0795374, 2.0387869, -2.0647454, 2.0414233, -4.1209607, 4.1035323
7: -2.1562178, 2.0801592, -2.1630886, 2.0999374, -4.2561550, 4.2432480
8: -3.4075100, 2.8226676, -3.4037845, 2.7859077, -6.1934175, 6.2264519
9: -1.8519459, 2.1501296, -1.8584915, 2.1539917, -4.0059376, 4.0086212

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8424213, upper bound: 10.8434255
time: 2.41 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8486822, upper bound: 10.8485066
time: 16.56 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -2.2280517, 1.8731461, -3.1581256, 2.6235197, -4.8515711, 5.0312719
1: -1.7951037, 1.8628588, -2.6157122, 2.5433664, -4.3384700, 4.4785709
2: -2.3322520, 2.1031368, -3.4782815, 2.7169085, -5.0491605, 5.5814180
3: -2.5081353, 1.8787794, -3.7717891, 2.4278023, -4.9359379, 5.6505685
4: -2.4774613, 1.9590179, -3.5940578, 2.7389915, -5.2164526, 5.5530758
5: -2.0479903, 1.9299963, -3.0376110, 2.6296663, -4.6776567, 4.9676075
6: -2.1109776, 2.0624204, -2.9737740, 2.9179006, -5.0288782, 5.0361943
7: -2.1822233, 2.1005418, -3.1487684, 3.0676262, -5.2498493, 5.2493105
8: -3.4519701, 2.8444986, -4.9294882, 3.1114371, -6.5634069, 7.7739868
9: -1.8751402, 2.1744518, -2.7186060, 3.0086436, -4.8837838, 4.8930578

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8445317, upper bound: 10.8438977
time: 3.51 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8488833, upper bound: 10.8485824
time: 2.94 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -3.3688412, 2.7963219, -3.3286576, 2.7630975, -6.1319389, 6.1249795
1: -2.8201103, 2.7117250, -2.7905605, 2.6969743, -5.5170846, 5.5022855
2: -3.7471359, 2.9100559, -3.7167680, 2.8840227, -6.6311588, 6.6268239
3: -4.0435085, 2.6006446, -4.0307322, 2.6342831, -6.6777916, 6.6313767
4: -3.8418293, 2.9227676, -3.8224938, 2.9042172, -6.7460465, 6.7452612
5: -3.2423074, 2.8038058, -3.2072144, 2.7919540, -6.0342617, 6.0110202
6: -3.1885350, 3.1158218, -3.1588957, 3.0946722, -6.2832069, 6.2747173
7: -3.3627625, 3.2607331, -3.3400431, 3.2417228, -6.6044855, 6.6007762
8: -5.2658396, 3.2297688, -5.1990910, 3.1431789, -8.4090185, 8.4288597
9: -2.9166121, 3.1996279, -2.8903782, 3.1734743, -6.0900865, 6.0900059

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8503068, upper bound: 10.8504488
time: 4.06 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8503132, upper bound: 10.8503132
time: 3.80 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -3.5490742, 2.9464583, -3.9284887, 3.2581851, -6.8072596, 6.8749471
1: -2.9921305, 2.8407619, -3.3609047, 3.1240208, -6.1161513, 6.2016668
2: -3.9625013, 3.0341249, -4.4342556, 3.2951584, -7.2576599, 7.4683805
3: -4.2803764, 2.7034256, -4.8113265, 2.9673696, -7.2477460, 7.5147524
4: -4.0620694, 3.0713229, -4.5561419, 3.3961427, -7.4582119, 7.6274648
5: -3.4269443, 2.9435132, -3.8322906, 3.2551899, -6.6821342, 6.7758036
6: -3.3605032, 3.2806571, -3.7256153, 3.6402977, -7.0008011, 7.0062723
7: -3.5474536, 3.4421592, -3.9574881, 3.8445921, -7.3920460, 7.3996472
8: -5.5421901, 3.3089881, -6.1150780, 3.4012477, -8.9434376, 9.4240665
9: -3.0825191, 3.3642206, -3.4407995, 3.7187929, -6.8013120, 6.8050203

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8503068, upper bound: 10.8504489
time: 2.51 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8503127, upper bound: 10.8503136
time: 2.49 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 6.39 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 6.39
Output dim: 8, lower bound: -10.8424213, upper bound: 10.8434255
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 6.39
Output dim: 8, lower bound: -10.8486822, upper bound: 10.8485066
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 6.39
Output dim: 8, lower bound: -10.8445317, upper bound: 10.8438977
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 6.39
Output dim: 8, lower bound: -10.8488833, upper bound: 10.8485824
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 6.39
Output dim: 8, lower bound: -10.8503068, upper bound: 10.8504488
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 6.39
Output dim: 8, lower bound: -10.8503132, upper bound: 10.8503132
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 6.39
Output dim: 8, lower bound: -10.8503068, upper bound: 10.8504489
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 6.39
Output dim: 8, lower bound: -10.8503127, upper bound: 10.8503136

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.4043604, 0.6074980, -1.2354671, 1.1876043, -1.5919647, 1.8429651
1: -0.3585974, 0.4519699, -1.0204153, 1.1403959, -1.4989933, 1.4723852
2: -0.4743955, 0.6384621, -1.1290644, 1.3493106, -1.8237062, 1.7675265
3: -0.2517788, 0.6248695, -1.2007763, 1.2570050, -1.5087838, 1.8256458
4: -0.4609523, 0.4615960, -1.3226705, 1.1801713, -1.6411235, 1.7842665
5: -0.4011335, 0.5442472, -1.1476113, 1.1961589, -1.5972924, 1.6918585
6: -0.3823066, 0.4923779, -1.1841390, 1.1812449, -1.5635514, 1.6765169
7: -0.4422995, 0.4368832, -1.2010869, 1.1929245, -1.6352240, 1.6379700
8: -0.2330061, 2.4567294, -1.7685511, 2.5993152, -2.8323212, 4.2252808
9: -0.6012856, 0.6008418, -1.1002686, 1.3244699, -1.9257555, 1.7011104

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8424213, upper bound: 10.8434255
time: 2.06 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8424213, upper bound: 10.8434255
time: 2.09 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -1.1180259, 1.1264412, -1.7272825, 1.5123899, -2.6304159, 2.8537238
1: -0.9244901, 1.0602522, -1.4019494, 1.4976541, -2.4221442, 2.4622016
2: -1.0329329, 1.3534248, -1.6805153, 1.7028482, -2.7357812, 3.0339401
3: -1.0377450, 1.1797954, -1.8439873, 1.5485891, -2.5863342, 3.0237827
4: -1.1730082, 1.0821362, -1.9126120, 1.5553848, -2.7283931, 2.9947481
5: -1.0305799, 1.1206187, -1.5983318, 1.5595558, -2.5901356, 2.7189505
6: -1.0965043, 1.0856285, -1.6343329, 1.6135590, -2.7100635, 2.7199614
7: -1.0725073, 1.0902597, -1.6767253, 1.6405287, -2.7130361, 2.7669849
8: -1.5849450, 2.6341076, -2.6162119, 2.6901593, -4.2751045, 5.2503195
9: -1.0289613, 1.2346197, -1.4623364, 1.7395637, -2.7685251, 2.6969562

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8486822, upper bound: 10.8485066
time: 2.39 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8486822, upper bound: 10.8485065
time: 5.00 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.3856851, 0.5910726, -1.9633067, 1.6912036, -2.0768886, 2.5543792
1: -0.3405835, 0.4391911, -1.5852257, 1.6591485, -1.9997320, 2.0244169
2: -0.4695427, 0.6303051, -1.9753512, 1.8627406, -2.3322835, 2.6056564
3: -0.2402608, 0.5979753, -2.1439354, 1.6410222, -1.8812829, 2.7419107
4: -0.4480854, 0.4362407, -2.2015350, 1.7349483, -2.1830337, 2.6377757
5: -0.3813739, 0.5261358, -1.8145806, 1.7253901, -2.1067638, 2.3407164
6: -0.3630418, 0.4719749, -1.8448713, 1.8254797, -2.1885216, 2.3168461
7: -0.4273925, 0.4155157, -1.9200816, 1.8643911, -2.2917836, 2.3355973
8: -0.2082924, 2.4583650, -3.0216718, 2.7501736, -2.9584661, 5.4800367
9: -0.5943607, 0.5857191, -1.6483594, 1.9395258, -2.5338864, 2.2340784

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8445317, upper bound: 10.8438977
time: 2.91 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8445317, upper bound: 10.8438977
time: 2.16 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -1.1789749, 1.1638298, -2.6801507, 2.2314696, -3.4104445, 3.8439806
1: -0.9667869, 1.0998735, -2.1717014, 2.1877542, -3.1545410, 3.2715750
2: -1.0958297, 1.4128406, -2.8918145, 2.3735299, -3.4693596, 4.3046551
3: -1.1019998, 1.2109493, -3.1258428, 2.1217892, -3.2237890, 4.3367920
4: -1.2424214, 1.1246908, -3.0224693, 2.3357372, -3.5781586, 4.1471601
5: -1.0822704, 1.1657479, -2.5426488, 2.2626519, -3.3449223, 3.7083967
6: -1.1547188, 1.1350241, -2.5126836, 2.4778674, -3.6325860, 3.6477077
7: -1.1257610, 1.1389929, -2.6603332, 2.5797882, -3.7055492, 3.7993259
8: -1.6918225, 2.6506441, -4.1909132, 2.9516060, -4.6434288, 6.8415575
9: -1.0720564, 1.2819602, -2.2832646, 2.5779598, -3.6500163, 3.5652249

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 218

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8488833, upper bound: 10.8485824
time: 7.60 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8488825, upper bound: 10.8485825
time: 3.31 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -1.4660666, 1.3306402, -2.4945865, 2.0806525, -3.5467191, 3.8252268
1: -1.1943853, 1.3196143, -2.0271344, 2.0855772, -3.2799625, 3.3467488
2: -1.3775412, 1.5532612, -2.6780229, 2.2756550, -3.6531963, 4.2312841
3: -1.4619946, 1.3992000, -2.9258416, 2.1042838, -3.5662785, 4.3250418
4: -1.5892677, 1.3516893, -2.8155012, 2.1995566, -3.7888243, 4.1671906
5: -1.3517020, 1.3700330, -2.3485756, 2.1487880, -3.5004900, 3.7186086
6: -1.4060488, 1.3793380, -2.3565466, 2.3250558, -3.7311046, 3.7358847
7: -1.4082381, 1.3890175, -2.4774411, 2.3985724, -3.8068104, 3.8664584
8: -2.1806726, 2.7064040, -3.8938568, 2.8483226, -5.0289955, 6.6002607
9: -1.2739632, 1.5111173, -2.1251023, 2.4251099, -3.6990731, 3.6362195

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8457065, upper bound: 10.8463458
time: 2.96 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8489356, upper bound: 10.8488348
time: 3.52 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -2.2922516, 1.9298626, -2.5683627, 2.1405647, -4.4328165, 4.4982252
1: -1.8519363, 1.9171284, -2.0859141, 2.1358392, -3.9877756, 4.0030427
2: -2.4126348, 2.1361301, -2.7798667, 2.3453226, -4.7579575, 4.9159966
3: -2.5911894, 1.8943267, -3.0205579, 2.1531053, -4.7442946, 4.9148846
4: -2.5626817, 2.0129530, -2.8987851, 2.2618291, -4.8245106, 4.9117384
5: -2.1262560, 1.9796728, -2.4200633, 2.2035143, -4.3297701, 4.3997359
6: -2.1675062, 2.1254070, -2.4291143, 2.3918743, -4.5593805, 4.5545216
7: -2.2530234, 2.1735835, -2.5539036, 2.4690247, -4.7220478, 4.7274871
8: -3.5801113, 2.9053893, -4.0128927, 2.8802731, -6.4603844, 6.9182820
9: -1.9425740, 2.2366657, -2.1901360, 2.4907141, -4.4332881, 4.4268017

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8459213, upper bound: 10.8465889
time: 9.63 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8489859, upper bound: 10.8489859
time: 6.53 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -1.6158303, 1.4366711, -3.0788708, 2.5503294, -4.1661596, 4.5155420
1: -1.3121722, 1.4261323, -2.5512300, 2.5004582, -3.8126304, 3.9773622
2: -1.5528413, 1.6479323, -3.3890772, 2.6688333, -4.2216744, 5.0370092
3: -1.6668420, 1.4803703, -3.6905553, 2.4291778, -4.0960197, 5.1709256
4: -1.7682292, 1.4690592, -3.5102134, 2.6858397, -4.4540691, 4.9792728
5: -1.4909390, 1.4780560, -2.9581332, 2.5828500, -4.0737891, 4.4361892
6: -1.5394132, 1.5126630, -2.9026892, 2.8553209, -4.3947344, 4.4153523
7: -1.5587735, 1.5258867, -3.0783267, 2.9921374, -4.5509109, 4.6042132
8: -2.4388802, 2.7437100, -4.8086381, 3.0426402, -5.4815207, 7.5523481
9: -1.3834261, 1.6425891, -2.6507969, 2.9415884, -4.3250146, 4.2933860

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8457065, upper bound: 10.8463458
time: 2.75 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8489356, upper bound: 10.8488349
time: 3.28 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -2.4541075, 2.0567029, -3.1128478, 2.5793397, -5.0334473, 5.1695509
1: -1.9851649, 2.0307779, -2.5765743, 2.5216517, -4.5068169, 4.6073523
2: -2.6125124, 2.2412968, -3.4355431, 2.7081873, -5.3206997, 5.6768398
3: -2.8077796, 1.9896080, -3.7323115, 2.4569068, -5.2646866, 5.7219195
4: -2.7491951, 2.1473477, -3.5443130, 2.7126527, -5.4618478, 5.6916609
5: -2.2975371, 2.0971487, -2.9878120, 2.6087832, -4.9063206, 5.0849609
6: -2.3137426, 2.2715616, -2.9393582, 2.8845601, -5.1983027, 5.2109199
7: -2.4204097, 2.3344600, -3.1098335, 3.0186715, -5.4390812, 5.4442935
8: -3.8452392, 2.9574025, -4.8598680, 3.0699134, -6.9151525, 7.8172703
9: -2.0841813, 2.3789923, -2.6825035, 2.9702644, -5.0544457, 5.0614958

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8459213, upper bound: 10.8465889
time: 2.32 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8489859, upper bound: 10.8489859
time: 3.79 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 7.55 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 7.55
Output dim: 8, lower bound: -10.8424213, upper bound: 10.8434255
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 7.55
Output dim: 8, lower bound: -10.8424213, upper bound: 10.8434255
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 7.55
Output dim: 8, lower bound: -10.8486822, upper bound: 10.8485066
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 7.55
Output dim: 8, lower bound: -10.8486822, upper bound: 10.8485065
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 7.55
Output dim: 8, lower bound: -10.8445317, upper bound: 10.8438977
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 7.55
Output dim: 8, lower bound: -10.8445317, upper bound: 10.8438977
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 7.55
Output dim: 8, lower bound: -10.8488833, upper bound: 10.8485824
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 7.55
Output dim: 8, lower bound: -10.8488825, upper bound: 10.8485825
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 7.55
Output dim: 8, lower bound: -10.8457065, upper bound: 10.8463458
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 7.55
Output dim: 8, lower bound: -10.8489356, upper bound: 10.8488348
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 7.55
Output dim: 8, lower bound: -10.8459213, upper bound: 10.8465889
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 7.55
Output dim: 8, lower bound: -10.8489859, upper bound: 10.8489859
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 7.55
Output dim: 8, lower bound: -10.8457065, upper bound: 10.8463458
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 7.55
Output dim: 8, lower bound: -10.8489356, upper bound: 10.8488349
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 7.55
Output dim: 8, lower bound: -10.8459213, upper bound: 10.8465889
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 7.55
Output dim: 8, lower bound: -10.8489859, upper bound: 10.8489859

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.3579350, 0.5515869, -0.6427993, 0.7935848, -1.1515198, 1.1943861
1: -0.3141608, 0.4159026, -0.5836922, 0.7056606, -1.0198214, 0.9995947
2: -0.4479609, 0.5775763, -0.5916769, 0.9196732, -1.3676341, 1.1692531
3: -0.2188401, 0.5660433, -0.4959909, 0.9330486, -1.1518887, 1.0620341
4: -0.4263678, 0.4005649, -0.6691750, 0.7098185, -1.1361864, 1.0697398
5: -0.3559325, 0.4934884, -0.6347641, 0.7563028, -1.1122353, 1.1282525
6: -0.3316242, 0.4413666, -0.6491286, 0.6853328, -1.0169570, 1.0904952
7: -0.4042892, 0.3808291, -0.6589234, 0.6687571, -1.0730462, 1.0397525
8: -0.1215185, 2.4122949, -0.7070176, 2.4705968, -2.5921154, 3.1193125
9: -0.5719382, 0.5488146, -0.7179431, 0.8389457, -1.4108839, 1.2667577

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 218

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8037597, upper bound: 10.8340135
time: 2.43 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7756909, upper bound: 10.8295545
time: 1.69 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.3842893, 0.5848309, -1.0370927, 1.0584066, -1.4426959, 1.6219236
1: -0.3400367, 0.4364212, -0.8725663, 1.0042590, -1.3442957, 1.3089875
2: -0.4631833, 0.6139858, -0.9323015, 1.2177321, -1.6809154, 1.5462873
3: -0.2370567, 0.6010800, -0.9663225, 1.1552762, -1.3923329, 1.5674025
4: -0.4464871, 0.4362318, -1.0857419, 1.0302024, -1.4766896, 1.5219737
5: -0.3807752, 0.5240986, -0.9678369, 1.0516133, -1.4323885, 1.4919355
6: -0.3596447, 0.4724005, -1.0046282, 1.0131295, -1.3727741, 1.4770287
7: -0.4266691, 0.4143503, -1.0115947, 1.0261889, -1.4528580, 1.4259450
8: -0.1878986, 2.4383802, -1.4165478, 2.5469623, -2.7348609, 3.8549280
9: -0.5893306, 0.5785383, -0.9658356, 1.1627080, -1.7520386, 1.5443740

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8037597, upper bound: 10.8340135
time: 2.65 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7756909, upper bound: 10.8295545
time: 2.76 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.9146944, 0.9973972, -0.9900287, 1.0262423, -1.9409366, 1.9874259
1: -0.7801512, 0.9220710, -0.8501695, 0.9882913, -1.7684425, 1.7722405
2: -0.8305516, 1.2203692, -0.8944461, 1.2269726, -2.0575242, 2.1148152
3: -0.7997999, 1.0736260, -0.9015329, 1.1724048, -1.9722047, 1.9751589
4: -0.9459305, 0.9256461, -1.0256798, 0.9955359, -1.9414663, 1.9513259
5: -0.8480436, 0.9795129, -0.9234488, 1.0309503, -1.8789940, 1.9029617
6: -0.9067557, 0.9218251, -0.9763470, 0.9809557, -1.8877114, 1.8981720
7: -0.8901040, 0.9158691, -0.9632375, 0.9811059, -1.8712099, 1.8791065
8: -1.2376401, 2.5790095, -1.3488300, 2.5406675, -3.7783077, 3.9278395
9: -0.8940834, 1.0734980, -0.9395475, 1.1308941, -2.0249774, 2.0130455

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 92

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8478681, upper bound: 10.8479126
time: 3.43 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8477468, upper bound: 10.8475180
time: 3.12 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -1.0372276, 1.0753977, -1.5077183, 1.3510473, -2.3882749, 2.5831161
1: -0.8667643, 1.0060343, -1.2295378, 1.3428172, -2.2095814, 2.2355721
2: -0.9525806, 1.3011525, -1.4149162, 1.5615077, -2.5140882, 2.7160687
3: -0.9445344, 1.1371448, -1.5413941, 1.4286326, -2.3731670, 2.6785388
4: -1.0802345, 1.0207359, -1.6487494, 1.3825060, -2.4627404, 2.6694851
5: -0.9561310, 1.0653993, -1.3922055, 1.4002357, -2.3563666, 2.4576049
6: -1.0227489, 1.0189084, -1.4355237, 1.4157407, -2.4384897, 2.4544320
7: -0.9984612, 1.0226597, -1.4560542, 1.4344676, -2.4329288, 2.4787140
8: -1.4457150, 2.6120191, -2.2360191, 2.6287494, -4.0744643, 4.8480382
9: -0.9747210, 1.1700163, -1.2980397, 1.5450824, -2.5198035, 2.4680560

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8478681, upper bound: 10.8479125
time: 2.81 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8477468, upper bound: 10.8475180
time: 3.21 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.3426904, 0.5365787, -1.2368059, 1.1900363, -1.5327266, 1.7733846
1: -0.2979554, 0.4031311, -1.0171199, 1.1470044, -1.4449598, 1.4202510
2: -0.4447793, 0.5708351, -1.1403388, 1.4009376, -1.8457168, 1.7111739
3: -0.2099150, 0.5406694, -1.1902831, 1.2585814, -1.4684963, 1.7309525
4: -0.4146934, 0.3815612, -1.3248882, 1.1722932, -1.5869865, 1.7064495
5: -0.3401187, 0.4758513, -1.1407405, 1.2053987, -1.5455174, 1.6165918
6: -0.3183238, 0.4213136, -1.1957824, 1.1847206, -1.5030444, 1.6170959
7: -0.3903328, 0.3615949, -1.1934103, 1.1953206, -1.5856534, 1.5550051
8: -0.1010466, 2.4146442, -1.7772872, 2.5895033, -2.6905499, 4.1919317
9: -0.5666758, 0.5350150, -1.1073307, 1.3254876, -1.8921634, 1.6423457

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8094147, upper bound: 10.8362946
time: 2.97 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7817172, upper bound: 10.8317979
time: 3.21 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.3682125, 0.5683104, -1.7362462, 1.5238633, -1.8920758, 2.3045566
1: -0.3232426, 0.4233845, -1.4003532, 1.4933151, -1.8165576, 1.8237377
2: -0.4587233, 0.6063846, -1.6975253, 1.7141773, -2.1729007, 2.3039098
3: -0.2272792, 0.5744941, -1.8310280, 1.5153861, -1.7426653, 2.4055221
4: -0.4339822, 0.4127258, -1.9292423, 1.5541646, -1.9881468, 2.3419681
5: -0.3637002, 0.5057818, -1.6010900, 1.5591528, -1.9228530, 2.1068716
6: -0.3427901, 0.4517881, -1.6368273, 1.6176456, -1.9604357, 2.0886154
7: -0.4117745, 0.3933746, -1.6862175, 1.6482459, -2.0600204, 2.0795922
8: -0.1636088, 2.4401746, -2.6282306, 2.6848936, -2.8485024, 5.0684052
9: -0.5827428, 0.5639079, -1.4698381, 1.7370129, -2.3197556, 2.0337460

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8094147, upper bound: 10.8362946
time: 2.63 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7817172, upper bound: 10.8317979
time: 3.53 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.9684784, 1.0319247, -1.9023263, 1.6345156, -2.6029940, 2.9342511
1: -0.8171946, 0.9586614, -1.5347983, 1.6278535, -2.4450481, 2.4934597
2: -0.8871958, 1.2770023, -1.9159312, 1.8681345, -2.7553303, 3.1929336
3: -0.8569835, 1.1007984, -2.0699129, 1.6739291, -2.5309126, 3.1707113
4: -1.0025612, 0.9638014, -2.1109936, 1.6911463, -2.6937075, 3.0747950
5: -0.8919826, 1.0204520, -1.7497840, 1.6919507, -2.5839334, 2.7702360
6: -0.9608990, 0.9647748, -1.8021601, 1.7698720, -2.7307711, 2.7669349
7: -0.9337769, 0.9621022, -1.8481581, 1.8033831, -2.7371600, 2.8102603
8: -1.3365197, 2.5948565, -2.9056053, 2.7149687, -4.0514884, 5.5004616
9: -0.9310005, 1.1155852, -1.5978429, 1.8925118, -2.8235123, 2.7134280

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8481978, upper bound: 10.8481799
time: 2.45 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8481303, upper bound: 10.8476902
time: 2.47 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -1.0931306, 1.1099932, -2.4216881, 2.0273552, -3.1204858, 3.5316813
1: -0.9051741, 1.0429332, -1.9645312, 2.0035610, -2.9087353, 3.0074644
2: -1.0116602, 1.3582058, -2.5774493, 2.2062850, -3.2179451, 3.9356551
3: -1.0026276, 1.1655509, -2.7872708, 1.9718709, -2.9744985, 3.9528217
4: -1.1430602, 1.0599892, -2.7309875, 2.1220613, -3.2651215, 3.7909768
5: -1.0041709, 1.1058916, -2.2719655, 2.0781503, -3.0823212, 3.3778572
6: -1.0771387, 1.0645186, -2.2770209, 2.2462416, -3.3233802, 3.3415394
7: -1.0446502, 1.0692073, -2.4002943, 2.3223333, -3.3669834, 3.4695015
8: -1.5467176, 2.6279294, -3.7824099, 2.8569694, -4.4036870, 6.4103394
9: -1.0130436, 1.2137399, -2.0586729, 2.3515041, -3.3645477, 3.2724128

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8481978, upper bound: 10.8481796
time: 5.33 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8481302, upper bound: 10.8476904
time: 2.81 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.6792385, 0.8392280, -0.4719619, 0.6476282, -1.3268666, 1.3111899
1: -0.6068542, 0.7312909, -0.4322266, 0.5303035, -1.1371577, 1.1635175
2: -0.6275367, 0.9632816, -0.4932367, 0.7079936, -1.3355303, 1.4565182
3: -0.5380987, 0.9323756, -0.3088719, 0.7531818, -1.2912805, 1.2412474
4: -0.7077425, 0.7432685, -0.5083756, 0.5426600, -1.2504025, 1.2516441
5: -0.6659219, 0.7875402, -0.4691519, 0.6092744, -1.2751963, 1.2566922
6: -0.6862788, 0.7175530, -0.4604885, 0.5465732, -1.2328520, 1.1780415
7: -0.6931224, 0.7079447, -0.5029935, 0.5027412, -1.1958635, 1.2109382
8: -0.7972792, 2.5795188, -0.3542492, 2.4153016, -3.2125807, 2.9337680
9: -0.7530746, 0.8765873, -0.6221384, 0.6653582, -1.4184328, 1.4987257

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8407394, upper bound: 10.8239898
time: 2.01 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8404607, upper bound: 10.8229561
time: 2.13 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -1.0345616, 1.0706791, -1.4052455, 1.2792685, -2.3138301, 2.4759245
1: -0.8745953, 1.0141129, -1.1497573, 1.2722820, -2.1468773, 2.1638703
2: -0.9465399, 1.2643640, -1.3115633, 1.5333282, -2.4798679, 2.5759273
3: -0.9537855, 1.1642642, -1.3863803, 1.3818971, -2.3356826, 2.5506444
4: -1.0760713, 1.0300677, -1.5194933, 1.2998794, -2.3759508, 2.5495610
5: -0.9626806, 1.0615180, -1.2914971, 1.3297122, -2.2923927, 2.3530149
6: -1.0212004, 1.0191028, -1.3491627, 1.3248078, -2.3460083, 2.3682656
7: -1.0026731, 1.0198544, -1.3494828, 1.3376110, -2.3402841, 2.3693371
8: -1.4450341, 2.6490076, -2.0605826, 2.6076179, -4.0526519, 4.7095900
9: -0.9741148, 1.1736236, -1.2241430, 1.4573127, -2.4314275, 2.3977666

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8483221, upper bound: 10.8485347
time: 2.31 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8483099, upper bound: 10.8481744
time: 2.79 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -1.2070260, 1.1883464, -0.4581404, 0.6328316, -1.8398576, 1.6464868
1: -0.9903581, 1.1203762, -0.4178665, 0.5154772, -1.5058353, 1.5382427
2: -1.1170475, 1.3852674, -0.4899753, 0.7057483, -1.8227959, 1.8752427
3: -1.1516836, 1.2174633, -0.2964560, 0.7334941, -1.8851776, 1.5139192
4: -1.2838676, 1.1513475, -0.4984052, 0.5259996, -1.8098671, 1.6497527
5: -1.1141745, 1.1821733, -0.4523298, 0.5975058, -1.7116803, 1.6345030
6: -1.1717650, 1.1617544, -0.4431474, 0.5337437, -1.7055087, 1.6049018
7: -1.1624470, 1.1669390, -0.4892989, 0.4894153, -1.6518623, 1.6562378
8: -1.7440765, 2.6920638, -0.3366221, 2.4169292, -4.1610060, 3.0286858
9: -1.0915645, 1.3082197, -0.6159739, 0.6535441, -1.7451086, 1.9241936

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8413274, upper bound: 10.8258480
time: 3.99 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8412345, upper bound: 10.8256224
time: 2.21 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -1.8532994, 1.6134348, -1.5194244, 1.3573529, -3.2106524, 3.1328592
1: -1.4939909, 1.5882030, -1.2384129, 1.3503897, -2.8443806, 2.8266158
2: -1.8598261, 1.8402951, -1.4479218, 1.6249522, -3.4847784, 3.2882168
3: -1.9708375, 1.6062328, -1.5398408, 1.4563192, -3.4271567, 3.1460736
4: -2.0474870, 1.6506445, -1.6509537, 1.3879820, -3.4354692, 3.3015981
5: -1.7050924, 1.6493380, -1.3914838, 1.4151889, -3.1202812, 3.0408218
6: -1.7614129, 1.7226501, -1.4534187, 1.4258064, -3.1872191, 3.1760688
7: -1.7928543, 1.7455165, -1.4616591, 1.4379797, -3.2308340, 3.2071757
8: -2.8436840, 2.8095455, -2.2512667, 2.6323557, -5.4760399, 5.0608120
9: -1.5655336, 1.8480453, -1.3077682, 1.5559430, -3.1214767, 3.1558137

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8483825, upper bound: 10.8487566
time: 2.59 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8483771, upper bound: 10.8483771
time: 2.58 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.7635387, 0.9058533, -0.6758897, 0.8345136, -1.5980523, 1.5817430
1: -0.6710826, 0.8020365, -0.6007124, 0.7196300, -1.3907125, 1.4027489
2: -0.6932665, 1.0360627, -0.6030082, 0.9455654, -1.6388319, 1.6390710
3: -0.6374953, 0.9901159, -0.5366870, 0.9124663, -1.5499617, 1.5268030
4: -0.7888569, 0.8181093, -0.7047047, 0.7383931, -1.5272501, 1.5228140
5: -0.7331417, 0.8598328, -0.6578687, 0.7853696, -1.5185113, 1.5177015
6: -0.7609805, 0.7901736, -0.6631619, 0.7113469, -1.4723274, 1.4533355
7: -0.7712575, 0.7872962, -0.6947887, 0.7113934, -1.4826509, 1.4820849
8: -0.9575533, 2.6085939, -0.7691084, 2.4826326, -3.4401860, 3.3777022
9: -0.8055106, 0.9498602, -0.7360113, 0.8625315, -1.6680422, 1.6858714

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8407394, upper bound: 10.8239898
time: 2.24 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8404607, upper bound: 10.8229561
time: 3.84 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -1.1692578, 1.1554668, -1.9133472, 1.6365958, -2.8058536, 3.0688140
1: -0.9723853, 1.1047943, -1.5462160, 1.6303854, -2.6027708, 2.6510103
2: -1.0786647, 1.3524439, -1.9219112, 1.8567576, -2.9354224, 3.2743552
3: -1.1103325, 1.2336708, -2.0869658, 1.6543432, -2.7646756, 3.3206367
4: -1.2343962, 1.1319830, -2.1329875, 1.6984929, -2.9328890, 3.2649705
5: -1.0861676, 1.1575463, -1.7638556, 1.6962368, -2.7824044, 2.9214020
6: -1.1422145, 1.1314278, -1.8026075, 1.7777807, -2.9199953, 2.9340353
7: -1.1290026, 1.1324182, -1.8656248, 1.8165724, -2.9455750, 2.9980431
8: -1.6791956, 2.6813638, -2.9255247, 2.7120457, -4.3912411, 5.6068888
9: -1.0646073, 1.2815963, -1.6030964, 1.9002303, -2.9648376, 2.8846927

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8483221, upper bound: 10.8485341
time: 3.99 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8483099, upper bound: 10.8481744
time: 2.36 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -1.3518485, 1.2789779, -0.6400032, 0.8054264, -2.1572747, 1.9189811
1: -1.0999537, 1.2194175, -0.5688674, 0.6889983, -1.7889520, 1.7882849
2: -1.2611611, 1.4767500, -0.5791275, 0.9205493, -2.1817102, 2.0558774
3: -1.3220115, 1.2919657, -0.4914818, 0.8787404, -2.2007518, 1.7834475
4: -1.4555404, 1.2600650, -0.6659518, 0.7015733, -2.1571136, 1.9260168
5: -1.2441812, 1.2841363, -0.6237407, 0.7517011, -1.9958823, 1.9078770
6: -1.3003306, 1.2826148, -0.6284815, 0.6844633, -1.9847939, 1.9110963
7: -1.2989641, 1.2887585, -0.6580283, 0.6697225, -1.9686866, 1.9467869
8: -1.9934293, 2.7261720, -0.7007135, 2.4838147, -4.4772439, 3.4268856
9: -1.1918334, 1.4212428, -0.7118984, 0.8324041, -2.0242376, 2.1331413

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8413274, upper bound: 10.8258480
time: 1.98 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8412345, upper bound: 10.8256224
time: 1.94 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -2.0062315, 1.7255309, -2.0007687, 1.7010241, -3.7072556, 3.7262995
1: -1.6156466, 1.6984586, -1.6158538, 1.6915816, -3.3072281, 3.3143125
2: -2.0448954, 1.9389292, -2.0379157, 1.9306110, -3.9755063, 3.9768448
3: -2.1843436, 1.6935439, -2.2058797, 1.7166601, -3.9010038, 3.8994236
4: -2.2300267, 1.7726552, -2.2317688, 1.7679502, -3.9979768, 4.0044241
5: -1.8494927, 1.7609714, -1.8419056, 1.7622378, -3.6117306, 3.6028771
6: -1.9021021, 1.8621922, -1.8868501, 1.8576339, -3.7597361, 3.7490423
7: -1.9531038, 1.8900838, -1.9551194, 1.8965813, -3.8496852, 3.8452032
8: -3.1061475, 2.8515821, -3.0724049, 2.7401683, -5.8463159, 5.9239869
9: -1.6866269, 1.9836403, -1.6734636, 1.9777030, -3.6643300, 3.6571040

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 218

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8483825, upper bound: 10.8487559
time: 6.95 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8483771, upper bound: 10.8483771
time: 2.85 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 11.19 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 11.19
Output dim: 8, lower bound: -10.8037597, upper bound: 10.8340135
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 11.19
Output dim: 8, lower bound: -10.7756909, upper bound: 10.8295545
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 11.19
Output dim: 8, lower bound: -10.8037597, upper bound: 10.8340135
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 11.19
Output dim: 8, lower bound: -10.7756909, upper bound: 10.8295545
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 11.19
Output dim: 8, lower bound: -10.8478681, upper bound: 10.8479126
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 11.19
Output dim: 8, lower bound: -10.8477468, upper bound: 10.8475180
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 11.19
Output dim: 8, lower bound: -10.8478681, upper bound: 10.8479125
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 11.19
Output dim: 8, lower bound: -10.8477468, upper bound: 10.8475180
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 11.19
Output dim: 8, lower bound: -10.8094147, upper bound: 10.8362946
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 11.19
Output dim: 8, lower bound: -10.7817172, upper bound: 10.8317979
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 11.19
Output dim: 8, lower bound: -10.8094147, upper bound: 10.8362946
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 11.19
Output dim: 8, lower bound: -10.7817172, upper bound: 10.8317979
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 11.19
Output dim: 8, lower bound: -10.8481978, upper bound: 10.8481799
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 11.19
Output dim: 8, lower bound: -10.8481303, upper bound: 10.8476902
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 11.19
Output dim: 8, lower bound: -10.8481978, upper bound: 10.8481796
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 11.19
Output dim: 8, lower bound: -10.8481302, upper bound: 10.8476904
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 11.19
Output dim: 8, lower bound: -10.8407394, upper bound: 10.8239898
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 11.19
Output dim: 8, lower bound: -10.8404607, upper bound: 10.8229561
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 11.19
Output dim: 8, lower bound: -10.8483221, upper bound: 10.8485347
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 11.19
Output dim: 8, lower bound: -10.8483099, upper bound: 10.8481744
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 11.19
Output dim: 8, lower bound: -10.8413274, upper bound: 10.8258480
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 11.19
Output dim: 8, lower bound: -10.8412345, upper bound: 10.8256224
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 11.19
Output dim: 8, lower bound: -10.8483825, upper bound: 10.8487566
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 11.19
Output dim: 8, lower bound: -10.8483771, upper bound: 10.8483771
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 11.19
Output dim: 8, lower bound: -10.8407394, upper bound: 10.8239898
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 11.19
Output dim: 8, lower bound: -10.8404607, upper bound: 10.8229561
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 11.19
Output dim: 8, lower bound: -10.8483221, upper bound: 10.8485341
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 11.19
Output dim: 8, lower bound: -10.8483099, upper bound: 10.8481744
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 11.19
Output dim: 8, lower bound: -10.8413274, upper bound: 10.8258480
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 11.19
Output dim: 8, lower bound: -10.8412345, upper bound: 10.8256224
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 11.19
Output dim: 8, lower bound: -10.8483825, upper bound: 10.8487559
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 11.19
Output dim: 8, lower bound: -10.8483771, upper bound: 10.8483771

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.2837172, 0.4566129, -0.5682322, 0.7299621, -1.0136793, 1.0248451
1: -0.2366557, 0.3478408, -0.5222386, 0.6414773, -0.8781331, 0.8700794
2: -0.4080100, 0.4674220, -0.5474883, 0.8410208, -1.2490308, 1.0149103
3: -0.1631940, 0.4603103, -0.4089364, 0.8762745, -1.0394685, 0.8692467
4: -0.3625288, 0.3168460, -0.5996383, 0.6361418, -0.9986706, 0.9164843
5: -0.2873004, 0.3946393, -0.5687888, 0.6923754, -0.9796758, 0.9634281
6: -0.2573539, 0.3556260, -0.5787522, 0.6261016, -0.8834555, 0.9343782
7: -0.3418233, 0.2827336, -0.5882402, 0.5957198, -0.9375432, 0.8709738
8: 0.0547081, 2.3356428, -0.5613427, 2.4380288, -2.3833208, 2.8969855
9: -0.5331606, 0.4568836, -0.6749626, 0.7706154, -1.3037760, 1.1318462

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5089659, upper bound: 10.4871174
time: 2.53 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6611291, upper bound: 10.7331084
time: 2.73 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.6162384, 0.8692144, -0.5350649, 0.7045001, -1.3207386, 1.4042792
1: -0.5474260, 0.6027564, -0.4962384, 0.6128722, -1.1602982, 1.0989947
2: -0.5596392, 0.8958975, -0.5308120, 0.8063342, -1.3659735, 1.4267095
3: -0.3756389, 0.8745847, -0.3724231, 0.8504226, -1.2260615, 1.2470078
4: -0.6084315, 0.7183238, -0.5728198, 0.6036881, -1.2121196, 1.2911437
5: -0.5698940, 0.7748866, -0.5388077, 0.6663621, -1.2362561, 1.3136944
6: -0.5673255, 0.7321869, -0.5481439, 0.6015072, -1.1688327, 1.2803307
7: -0.6234372, 0.6861230, -0.5575675, 0.5687987, -1.1922359, 1.2436905
8: -0.6795669, 2.4134986, -0.5015911, 2.4205139, -3.1000807, 2.9150898
9: -0.7263808, 0.8062595, -0.6586135, 0.7418987, -1.4682795, 1.4648731

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4933578, upper bound: 10.4820897
time: 2.71 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6319641, upper bound: 10.7267906
time: 2.73 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.3007282, 0.4875547, -0.8943683, 0.9703861, -1.2711143, 1.3819230
1: -0.2556671, 0.3679626, -0.7699892, 0.9033329, -1.1590000, 1.1379519
2: -0.4200501, 0.4993166, -0.7933325, 1.1236167, -1.5436668, 1.2926490
3: -0.1788405, 0.4890025, -0.7946840, 1.0818241, -1.2606646, 1.2836864
4: -0.3811160, 0.3381993, -0.9277589, 0.9185477, -1.2996638, 1.2659582
5: -0.3052583, 0.4215920, -0.8368798, 0.9539266, -1.2591850, 1.2584718
6: -0.2779050, 0.3772423, -0.8732637, 0.8968022, -1.1747073, 1.2505060
7: -0.3587787, 0.3074970, -0.8822821, 0.9047309, -1.2635095, 1.1897792
8: 0.0037651, 2.3597918, -1.1654706, 2.5097160, -2.5059509, 3.5252624
9: -0.5450766, 0.4847860, -0.8736245, 1.0480125, -1.5930891, 1.3584105

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5089659, upper bound: 10.4871174
time: 2.43 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6611291, upper bound: 10.7331084
time: 2.88 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.6558267, 0.9088413, -0.8241374, 0.9272930, -1.5831196, 1.7329787
1: -0.5724534, 0.6294317, -0.7195543, 0.8533888, -1.4258422, 1.3489859
2: -0.5879613, 0.9440218, -0.7301275, 1.0776024, -1.6655637, 1.6741493
3: -0.4285598, 0.9083832, -0.7112947, 1.0444882, -1.4730480, 1.6196778
4: -0.6409537, 0.7519498, -0.8546547, 0.8625112, -1.5034649, 1.6066046
5: -0.6127214, 0.8038550, -0.7763430, 0.9067537, -1.5194751, 1.5801980
6: -0.6300936, 0.7621012, -0.8085641, 0.8411447, -1.4712384, 1.5706654
7: -0.6458867, 0.7335248, -0.8201933, 0.8460959, -1.4919825, 1.5537181
8: -0.7486411, 2.4346743, -1.0442222, 2.4853017, -3.2339430, 3.4788966
9: -0.7502123, 0.8601401, -0.8301407, 0.9922072, -1.7424195, 1.6902808

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4933578, upper bound: 10.4820897
time: 2.12 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6319641, upper bound: 10.7267906
time: 2.14 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.6174310, 0.7739822, -0.8518682, 0.9393665, -1.5567975, 1.6258504
1: -0.5507535, 0.6839973, -0.7527273, 0.8898670, -1.4406205, 1.4367247
2: -0.5866647, 0.9705589, -0.7608837, 1.1361163, -1.7227809, 1.7314427
3: -0.4464211, 0.8758153, -0.7357815, 1.1017866, -1.5482078, 1.6115968
4: -0.6430910, 0.6640586, -0.8762621, 0.8881248, -1.5312157, 1.5403206
5: -0.5958856, 0.7332711, -0.8034533, 0.9337807, -1.5296664, 1.5367244
6: -0.6314876, 0.6748120, -0.8480909, 0.8700425, -1.5015302, 1.5229028
7: -0.6228527, 0.6306599, -0.8407389, 0.8617207, -1.4845734, 1.4713988
8: -0.6953082, 2.4709792, -1.1143382, 2.5043280, -3.1996362, 3.5853174
9: -0.7101520, 0.8180386, -0.8515879, 1.0215318, -1.7316839, 1.6696265

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6739700, upper bound: 10.5800244
time: 2.49 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8423197, upper bound: 10.8412798
time: 1.77 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -1.9407549, 1.6419548, -0.7950152, 0.9022931, -2.8430481, 2.4369700
1: -1.5002409, 1.6215090, -0.7107618, 0.8463430, -2.3465838, 2.3322706
2: -1.8262285, 1.8804337, -0.7112815, 1.0956557, -2.9218841, 2.5917153
3: -2.0337439, 1.5741997, -0.6674180, 1.0687395, -3.1024833, 2.2416177
4: -2.0956712, 1.6871412, -0.8166280, 0.8421798, -2.9378510, 2.5037692
5: -1.7136470, 1.7011950, -0.7561272, 0.8921697, -2.6058168, 2.4573221
6: -1.8611972, 1.7258128, -0.7980303, 0.8214135, -2.6826108, 2.5238431
7: -1.7944573, 1.8362811, -0.7937586, 0.8138941, -2.6083515, 2.6300397
8: -2.9806154, 2.5952339, -1.0161079, 2.4825883, -5.4632034, 3.6113420
9: -1.6060051, 1.8368833, -0.8179531, 0.9739928, -2.5799980, 2.6548364

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6596500, upper bound: 10.5761317
time: 2.55 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8307307, upper bound: 10.8376879
time: 2.96 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.6839392, 0.8341589, -1.3426545, 1.2427678, -1.9267070, 2.1768134
1: -0.6071441, 0.7422991, -1.1036911, 1.2281764, -1.8353205, 1.8459902
2: -0.6304923, 1.0427392, -1.2406734, 1.4586885, -2.0891809, 2.2834125
3: -0.5263523, 0.9272783, -1.3193403, 1.3437486, -1.8701010, 2.2466187
4: -0.7090896, 0.7296945, -1.4503925, 1.2569575, -1.9660470, 2.1800871
5: -0.6555626, 0.7928280, -1.2396553, 1.2816305, -1.9371932, 2.0324831
6: -0.6971338, 0.7312867, -1.2902322, 1.2734959, -1.9706297, 2.0215189
7: -0.6860474, 0.7020286, -1.2967932, 1.2863851, -1.9724324, 1.9988219
8: -0.8324839, 2.5002272, -1.9564599, 2.5866575, -3.4191415, 4.4566870
9: -0.7513066, 0.8814057, -1.1790289, 1.4084811, -2.1597877, 2.0604346

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6739700, upper bound: 10.5800244
time: 2.56 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8423197, upper bound: 10.8412640
time: 2.07 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -2.0985000, 1.7295101, -1.2566954, 1.1891844, -3.2876844, 2.9862056
1: -1.6092354, 1.7341914, -1.0395417, 1.1696683, -2.7789037, 2.7737331
2: -1.9973451, 1.9941630, -1.1530864, 1.4047447, -3.4020898, 3.1472495
3: -2.1945701, 1.6789614, -1.2165463, 1.2988788, -3.4934487, 2.8955078
4: -2.3196001, 1.8215735, -1.3474007, 1.1915921, -3.5111923, 3.1689742
5: -1.9149272, 1.7856332, -1.1608363, 1.2209663, -3.1358936, 2.9464695
6: -1.9827391, 1.8685017, -1.2130525, 1.2022126, -3.1849518, 3.0815542
7: -1.9389383, 1.9622091, -1.2151542, 1.2125492, -3.1514874, 3.1773634
8: -3.2658720, 2.6309085, -1.8098272, 2.5595503, -5.8254223, 4.4407358
9: -1.6995120, 2.0145674, -1.1191804, 1.3412135, -3.0407255, 3.1337478

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6596500, upper bound: 10.5761317
time: 2.94 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8307307, upper bound: 10.8376879
time: 1.99 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.2762590, 0.4432430, -1.0732609, 1.0903459, -1.3666048, 1.5165038
1: -0.2268272, 0.3355546, -0.8984916, 1.0381279, -1.2649550, 1.2340461
2: -0.4070354, 0.4657289, -0.9798259, 1.2976695, -1.7047050, 1.4455547
3: -0.1565800, 0.4401356, -0.9995790, 1.1761720, -1.3327520, 1.4397146
4: -0.3521411, 0.3059670, -1.1330552, 1.0492313, -1.4013724, 1.4390223
5: -0.2765473, 0.3815849, -0.9924094, 1.0901511, -1.3666984, 1.3739944
6: -0.2479128, 0.3436202, -1.0507615, 1.0516071, -1.2995199, 1.3943816
7: -0.3318422, 0.2711821, -1.0391017, 1.0596232, -1.3914654, 1.3102838
8: 0.0625017, 2.3389528, -1.4987190, 2.5498347, -2.4873331, 3.8376718
9: -0.5320836, 0.4451236, -0.9951489, 1.1974541, -1.7295377, 1.4402726

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5096823, upper bound: 10.4964377
time: 2.82 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6651469, upper bound: 10.7546932
time: 3.53 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.5957246, 0.8224653, -0.9910657, 1.0389423, -1.6346669, 1.8135310
1: -0.5211629, 0.5753700, -0.8399827, 0.9809563, -1.5021192, 1.4153526
2: -0.5500283, 0.8834287, -0.8962902, 1.2442799, -1.7943082, 1.7797189
3: -0.3653175, 0.8159733, -0.9035264, 1.1316526, -1.4969702, 1.7194996
4: -0.5726804, 0.5873835, -1.0365956, 0.9859313, -1.5586116, 1.6239791
5: -0.5493295, 0.7486562, -0.9157909, 1.0326858, -1.5820153, 1.6644471
6: -0.5095139, 0.7061470, -0.9747267, 0.9849675, -1.4944813, 1.6808736
7: -0.5824388, 0.6305768, -0.9634036, 0.9893062, -1.5717450, 1.5939804
8: -0.5716054, 2.4159245, -1.3575977, 2.5237937, -3.0953991, 3.7735224
9: -0.6750190, 0.7902375, -0.9400241, 1.1309505, -1.8059695, 1.7302617

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4920426, upper bound: 10.4909146
time: 2.56 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6357786, upper bound: 10.7484359
time: 2.76 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.2924749, 0.4738486, -1.5599929, 1.3977785, -1.6902535, 2.0338414
1: -0.2449671, 0.3554446, -1.2621958, 1.3678271, -1.6127942, 1.6176405
2: -0.4185296, 0.4972199, -1.4828370, 1.6033913, -2.0219209, 1.9800569
3: -0.1713458, 0.4685903, -1.5879194, 1.4206009, -1.5919467, 2.0565095
4: -0.3704672, 0.3265674, -1.7174926, 1.4139450, -1.7844121, 2.0440600
5: -0.2938722, 0.4072210, -1.4350073, 1.4320472, -1.7259195, 1.8422284
6: -0.2677978, 0.3649530, -1.4785318, 1.4604006, -1.7281983, 1.8434848
7: -0.3485916, 0.2943018, -1.5087252, 1.4813133, -1.8299049, 1.8030270
8: 0.0128798, 2.3629889, -2.3273563, 2.6388512, -2.6259713, 4.6903453
9: -0.5432180, 0.4726730, -1.3396621, 1.5827632, -2.1259813, 1.8123350

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5096823, upper bound: 10.4964377
time: 2.82 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6651469, upper bound: 10.7546932
time: 1.88 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.6243826, 0.8821505, -1.4624097, 1.3307372, -1.9551198, 2.3445601
1: -0.5537275, 0.6313586, -1.1852744, 1.2993062, -1.8530337, 1.8166330
2: -0.5728458, 0.9206786, -1.3670100, 1.5426455, -2.1154914, 2.2876885
3: -0.3928201, 0.8792998, -1.4627523, 1.3668486, -1.7596687, 2.3420522
4: -0.6208358, 0.7263362, -1.6000729, 1.3368379, -1.9576738, 2.3264091
5: -0.5858924, 0.7834985, -1.3429221, 1.3630651, -1.9489576, 2.1264205
6: -0.5808225, 0.7396041, -1.3915867, 1.3749884, -1.9558109, 2.1311908
7: -0.6378068, 0.6953799, -1.4091756, 1.3943231, -2.0321298, 2.1045556
8: -0.7100611, 2.4370079, -2.1588893, 2.6090689, -3.3191299, 4.5958972
9: -0.7347291, 0.8312135, -1.2687638, 1.4991004, -2.2338295, 2.0999773

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4920426, upper bound: 10.4909146
time: 2.52 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6357786, upper bound: 10.7484359
time: 2.10 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.6506287, 0.8000517, -1.7281692, 1.5096272, -2.1602559, 2.5282209
1: -0.5765964, 0.7115775, -1.3994378, 1.5011170, -2.0777135, 2.1110153
2: -0.6075222, 1.0246105, -1.7035484, 1.7581879, -2.3657103, 2.7281590
3: -0.4787563, 0.8970340, -1.8298241, 1.5783305, -2.0570869, 2.7268581
4: -0.6734983, 0.6911918, -1.9016328, 1.5522850, -2.2257833, 2.5928245
5: -0.6221793, 0.7599991, -1.5864748, 1.5654094, -2.1875887, 2.3464739
6: -0.6637344, 0.7032583, -1.6457851, 1.6130786, -2.2768130, 2.3490434
7: -0.6485103, 0.6610963, -1.6710149, 1.6383038, -2.2868142, 2.3321114
8: -0.7672678, 2.4850290, -2.6084905, 2.6650815, -3.4323492, 5.0935192
9: -0.7294382, 0.8489685, -1.4673884, 1.7387915, -2.4682298, 2.3163569

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6889039, upper bound: 10.5972798
time: 2.56 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8452626, upper bound: 10.8452494
time: 3.11 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -1.9876039, 1.6729136, -1.6341085, 1.4421138, -3.4297175, 3.3070221
1: -1.5278269, 1.6611063, -1.3262494, 1.4334517, -2.9612784, 2.9873557
2: -1.8768392, 1.9410697, -1.5878367, 1.6995506, -3.5763898, 3.5289063
3: -2.0894592, 1.6008039, -1.6999316, 1.5246990, -3.6141582, 3.3007355
4: -2.1541729, 1.7451100, -1.7893083, 1.4763494, -3.6305223, 3.5344183
5: -1.7674332, 1.7315589, -1.4977585, 1.4976623, -3.2650955, 3.2293174
6: -1.9094657, 1.7756639, -1.5603436, 1.5293136, -3.4387794, 3.3360076
7: -1.8331852, 1.8732913, -1.5761654, 1.5500617, -3.3832469, 3.4494567
8: -3.0504601, 2.6195223, -2.4469905, 2.6337805, -5.6842403, 5.0665131
9: -1.6387987, 1.8977200, -1.3985780, 1.6558903, -3.2946892, 3.2962980

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6737220, upper bound: 10.5933411
time: 2.37 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8447067, upper bound: 10.8441537
time: 1.71 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.7207297, 0.8623386, -2.2387977, 1.8818722, -2.6026020, 3.1011362
1: -0.6343191, 0.7710211, -1.8112235, 1.8703984, -2.5047174, 2.5822446
2: -0.6595692, 1.0963399, -2.3434572, 2.0860367, -2.7456059, 3.4397972
3: -0.5610950, 0.9518531, -2.5375078, 1.8643327, -2.4254277, 3.4893608
4: -0.7443541, 0.7579960, -2.5144746, 1.9666109, -2.7109652, 3.2724705
5: -0.6846889, 0.8243388, -2.0767894, 1.9417765, -2.6264653, 2.9011283
6: -0.7337345, 0.7608014, -2.1095963, 2.0777431, -2.8114777, 2.8703976
7: -0.7151073, 0.7368076, -2.2053254, 2.1372700, -2.8523774, 2.9421329
8: -0.9095598, 2.5142663, -3.4757612, 2.7910361, -3.7005959, 5.9900274
9: -0.7757834, 0.9131776, -1.8944793, 2.1884005, -2.9641838, 2.8076568

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6889039, upper bound: 10.5972798
time: 2.48 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8452626, upper bound: 10.8452478
time: 5.02 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -2.2035151, 1.7839757, -2.1355982, 1.8007196, -4.0042348, 3.9195738
1: -1.6817923, 1.7851045, -1.7240957, 1.7946668, -3.4764590, 3.5092001
2: -2.0758550, 2.0604239, -2.2104030, 2.0175402, -4.0933952, 4.2708268
3: -2.2898259, 1.7147843, -2.3951910, 1.8013387, -4.0911646, 4.1099753
4: -2.4309649, 1.8614993, -2.3932261, 1.8781884, -4.3091536, 4.2547255
5: -2.0106227, 1.8501642, -1.9675773, 1.8642371, -3.8748598, 3.8177414
6: -2.0437231, 1.9730543, -2.0129266, 1.9828782, -4.0266013, 3.9859810
7: -2.0782585, 2.0074468, -2.0954387, 2.0339868, -4.1122456, 4.1028852
8: -3.4136415, 2.6530185, -3.3011301, 2.7495515, -6.1631927, 5.9541483
9: -1.8021276, 2.0520260, -1.8022116, 2.0957422, -3.8978698, 3.8542376

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6737220, upper bound: 10.5933411
time: 2.60 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8447067, upper bound: 10.8441537
time: 2.04 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.4694453, 0.6529000, -0.4276201, 0.6031079, -1.0725532, 1.0805202
1: -0.4308256, 0.5381819, -0.3947893, 0.4838384, -0.9146640, 0.9329712
2: -0.5157504, 0.7337205, -0.4735304, 0.6604729, -1.1762233, 1.2072510
3: -0.3096458, 0.7643185, -0.2726524, 0.7075283, -1.0171740, 1.0369709
4: -0.5159682, 0.5407785, -0.4776782, 0.4951326, -1.0111008, 1.0184567
5: -0.4675617, 0.6146805, -0.4243473, 0.5720962, -1.0396580, 1.0390278
6: -0.4750152, 0.5464661, -0.4103143, 0.5095930, -0.9846082, 0.9567804
7: -0.5024090, 0.5058461, -0.4635707, 0.4608580, -0.9632670, 0.9694169
8: -0.3893265, 2.4832556, -0.2701023, 2.3885517, -2.7778783, 2.7533579
9: -0.6466318, 0.6783789, -0.5977399, 0.6235150, -1.2701468, 1.2761188

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 69

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5934016, upper bound: 10.4522011
time: 2.50 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7749076, upper bound: 10.6884798
time: 3.55 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -1.2045536, 1.3313465, -0.4115116, 0.5890689, -1.7936225, 1.7428582
1: -1.0645616, 1.1782119, -0.3808619, 0.4710928, -1.5356544, 1.5590738
2: -0.9912682, 1.5091075, -0.4658744, 0.6435844, -1.6348525, 1.9749818
3: -1.1620898, 1.3224357, -0.2617567, 0.6890308, -1.8511206, 1.5841925
4: -1.2598410, 1.2332203, -0.4674135, 0.4767891, -1.7366300, 1.7006338
5: -1.1154909, 1.2897841, -0.4092528, 0.5583978, -1.6738887, 1.6990368
6: -1.1952012, 1.1631618, -0.3933637, 0.4958337, -1.6910348, 1.5565255
7: -1.1993785, 1.3130215, -0.4517347, 0.4454527, -1.6448312, 1.7647562
8: -1.8790401, 2.5717559, -0.2399555, 2.3766830, -4.2557230, 2.8117113
9: -1.1304786, 1.3456553, -0.5893623, 0.6086237, -1.7391024, 1.9350176

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 69

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5879611, upper bound: 10.4503066
time: 2.57 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7623304, upper bound: 10.6866906
time: 6.93 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.6700748, 0.8202695, -1.2395741, 1.1744721, -1.8445469, 2.0598435
1: -0.6062815, 0.7360718, -1.0256579, 1.1587013, -1.7649828, 1.7617297
2: -0.6243469, 0.9936755, -1.1414099, 1.4298596, -2.0542064, 2.1350853
3: -0.5181841, 0.9499421, -1.1834252, 1.2958721, -1.8140562, 2.1333673
4: -0.6953346, 0.7274643, -1.3209059, 1.1747738, -1.8701084, 2.0483704
5: -0.6516439, 0.7822532, -1.1409409, 1.2123230, -1.8639668, 1.9231942
6: -0.6879987, 0.7176587, -1.2029917, 1.1857358, -1.8737345, 1.9206505
7: -0.6780669, 0.6925409, -1.1905929, 1.1935657, -1.8716326, 1.8831338
8: -0.7985743, 2.5394235, -1.7795602, 2.5668201, -3.3653946, 4.3189836
9: -0.7494843, 0.8712636, -1.1084423, 1.3259530, -2.0754373, 1.9797058

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7055027, upper bound: 10.6031621
time: 2.89 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8456860, upper bound: 10.8457536
time: 14.77 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -1.9746218, 1.6639342, -1.1637559, 1.1276543, -3.1022761, 2.8276901
1: -1.5325344, 1.6627880, -0.9698576, 1.1087554, -2.6412897, 2.6326456
2: -1.8741894, 1.8866677, -1.0651000, 1.3818308, -3.2560201, 2.9517677
3: -2.0608418, 1.6458868, -1.0950379, 1.2564892, -3.3173308, 2.7409248
4: -2.1849644, 1.7361028, -1.2312050, 1.1167901, -3.3017545, 2.9673078
5: -1.8128721, 1.7062726, -1.0721095, 1.1587096, -2.9715817, 2.7783821
6: -1.8726227, 1.7809684, -1.1345339, 1.1244578, -2.9970806, 2.9155023
7: -1.8385283, 1.8592323, -1.1180942, 1.1295327, -2.9680610, 2.9773264
8: -3.0769825, 2.6543026, -1.6532750, 2.5429430, -5.6199255, 4.3075776
9: -1.6216440, 1.9278944, -1.0566082, 1.2662485, -2.8878925, 2.9845026

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6983119, upper bound: 10.6013238
time: 2.77 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8455124, upper bound: 10.8453548
time: 2.40 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.7465351, 0.8966106, -0.4150795, 0.5930120, -1.3395470, 1.3116901
1: -0.6574052, 0.7951088, -0.3835334, 0.4761210, -1.1335262, 1.1786423
2: -0.6786262, 1.0814242, -0.4714028, 0.6620464, -1.3406726, 1.5528271
3: -0.6073010, 0.9720529, -0.2662764, 0.6888413, -1.2961423, 1.2383293
4: -0.7775435, 0.7924795, -0.4705992, 0.4790593, -1.2566028, 1.2630787
5: -0.7097749, 0.8526873, -0.4118507, 0.5614027, -1.2711775, 1.2645380
6: -0.7563305, 0.7856948, -0.3977675, 0.4973940, -1.2537246, 1.1834624
7: -0.7493750, 0.7751951, -0.4540552, 0.4491329, -1.1985079, 1.2292503
8: -0.9541827, 2.5753782, -0.2586430, 2.3918302, -3.3460131, 2.8340211
9: -0.7972384, 0.9406792, -0.5934395, 0.6156373, -1.4128757, 1.5341187

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6100872, upper bound: 10.4571540
time: 2.30 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7940340, upper bound: 10.6929824
time: 2.60 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -2.1998842, 1.8017216, -0.3993203, 0.5794075, -2.7792916, 2.2010419
1: -1.7313070, 1.7806100, -0.3697996, 0.4642047, -2.1955118, 2.1504095
2: -2.0754449, 2.0146561, -0.4640645, 0.6455398, -2.7209847, 2.4787207
3: -2.3053186, 1.7019336, -0.2555650, 0.6708801, -2.9761987, 1.9574987
4: -2.4604816, 1.8718842, -0.4605155, 0.4612517, -2.9217334, 2.3323998
5: -1.9893636, 1.8764817, -0.3974241, 0.5479141, -2.5372777, 2.2739058
6: -2.0316086, 1.9815786, -0.3821775, 0.4838560, -2.5154645, 2.3637562
7: -2.0977538, 2.0293136, -0.4427114, 0.4339628, -2.5317166, 2.4720249
8: -3.4372487, 2.7035983, -0.2295602, 2.3801157, -5.8173647, 2.9331584
9: -1.7858515, 2.0671854, -0.5854630, 0.6010057, -2.3868570, 2.6526484

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6053704, upper bound: 10.4550879
time: 2.66 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7815601, upper bound: 10.6909031
time: 3.39 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -1.2987497, 1.2309320, -1.3535562, 1.2442929, -2.5430427, 2.5844882
1: -1.0604417, 1.1940243, -1.1095760, 1.2348181, -2.2952600, 2.3036003
2: -1.2121027, 1.4971941, -1.2602552, 1.5218059, -2.7339087, 2.7574492
3: -1.2406735, 1.3034413, -1.3114239, 1.3666087, -2.6072822, 2.6148653
4: -1.3866558, 1.2131051, -1.4519858, 1.2584186, -2.6450744, 2.6650910
5: -1.1885052, 1.2553871, -1.2386136, 1.2948731, -2.4833784, 2.4940007
6: -1.2671200, 1.2373180, -1.3071373, 1.2808717, -2.5479918, 2.5444553
7: -1.2378148, 1.2425256, -1.2950454, 1.2883277, -2.5261426, 2.5375710
8: -1.9056187, 2.6665485, -1.9716592, 2.5893874, -4.4950061, 4.6382074
9: -1.1634939, 1.3766992, -1.1885052, 1.4135730, -2.5770669, 2.5652044

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7318947, upper bound: 10.6177226
time: 2.32 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8462860, upper bound: 10.8465753
time: 1.91 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -2.8719826, 2.3476377, -1.2754455, 1.1945536, -4.0665359, 3.6230831
1: -2.3167362, 2.3183875, -1.0503674, 1.1817983, -3.4985347, 3.3687549
2: -3.1324685, 2.4790809, -1.1766207, 1.4730213, -4.6054897, 3.6557016
3: -3.3822448, 2.1518722, -1.2151463, 1.3237869, -4.7060318, 3.3670185
4: -3.2858994, 2.4399142, -1.3586398, 1.1982636, -4.4841633, 3.7985539
5: -2.6562257, 2.3793125, -1.1675112, 1.2393675, -3.8955932, 3.5468237
6: -2.6575119, 2.6440129, -1.2372985, 1.2154679, -3.8729799, 3.8813114
7: -2.8255553, 2.7240617, -1.2189356, 1.2212034, -4.0467587, 3.9429975
8: -4.5677142, 2.8929093, -1.8436160, 2.5645123, -7.1322265, 4.7365255
9: -2.3400669, 2.7334747, -1.1347842, 1.3501769, -3.6902437, 3.8682590

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7247528, upper bound: 10.6159623
time: 2.56 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8461716, upper bound: 10.8461716
time: 2.81 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.5172318, 0.7050183, -0.5979371, 0.7663171, -1.2835490, 1.3029554
1: -0.4750557, 0.5896472, -0.5347099, 0.6520414, -1.1270971, 1.1243571
2: -0.5386876, 0.7937901, -0.5492778, 0.8661235, -1.4048111, 1.3430679
3: -0.3540901, 0.8071781, -0.4422859, 0.8510857, -1.2051758, 1.2494640
4: -0.5586371, 0.5861020, -0.6245806, 0.6615785, -1.2202156, 1.2106826
5: -0.5179222, 0.6524178, -0.5877079, 0.7124512, -1.2303734, 1.2401257
6: -0.5305310, 0.5920297, -0.5852681, 0.6509149, -1.1814460, 1.1772978
7: -0.5430107, 0.5526607, -0.6179329, 0.6292885, -1.1722991, 1.1705935
8: -0.4868004, 2.5082645, -0.6065448, 2.4486089, -2.9354093, 3.1148093
9: -0.6715684, 0.7302775, -0.6861031, 0.7911121, -1.4626805, 1.4163806

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5934016, upper bound: 10.4522011
time: 3.07 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7749076, upper bound: 10.6884798
time: 2.62 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -1.4801620, 1.4624062, -0.5668811, 0.7400931, -2.2202551, 2.0292873
1: -1.2316577, 1.3836164, -0.5083077, 0.6245642, -1.8562219, 1.8919241
2: -1.3605956, 1.6033825, -0.5321833, 0.8315542, -2.1921496, 2.1355658
3: -1.5181490, 1.4243308, -0.4065125, 0.8246904, -2.3428395, 1.8308433
4: -1.5564102, 1.4299099, -0.5968595, 0.6296929, -2.1861031, 2.0267694
5: -1.3191689, 1.4497371, -0.5597867, 0.6861740, -2.0053430, 2.0095239
6: -1.3498297, 1.4960114, -0.5558566, 0.6279033, -1.9777330, 2.0518680
7: -1.4252164, 1.4852396, -0.5883636, 0.5984873, -2.0237036, 2.0736032
8: -2.2880831, 2.6048017, -0.5445579, 2.4308381, -4.7189212, 3.1493597
9: -1.2572235, 1.6132156, -0.6687155, 0.7615446, -2.0187681, 2.2819312

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5879611, upper bound: 10.4503066
time: 2.61 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7623304, upper bound: 10.6866906
time: 5.99 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.7459809, 0.8847358, -1.7422148, 1.5138743, -2.2598553, 2.6269507
1: -0.6671487, 0.8012104, -1.4101571, 1.5079199, -2.1750686, 2.2113676
2: -0.6797488, 1.0675014, -1.7127388, 1.7480435, -2.4277923, 2.7802401
3: -0.6073182, 1.0073941, -1.8496084, 1.5606236, -2.1679418, 2.8570025
4: -0.7709308, 0.7988558, -1.9272691, 1.5617913, -2.3327222, 2.7261248
5: -0.7155277, 0.8517933, -1.6030878, 1.5718079, -2.2873354, 2.4548812
6: -0.7604897, 0.7818516, -1.6485229, 1.6234634, -2.3839531, 2.4303744
7: -0.7504506, 0.7705733, -1.6900994, 1.6539010, -2.4043517, 2.4606726
8: -0.9489703, 2.5684536, -2.6338062, 2.6618683, -3.6108387, 5.2022600
9: -0.7987324, 0.9399019, -1.4731649, 1.7489244, -2.5476568, 2.4130669

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7055027, upper bound: 10.6031621
time: 2.39 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8456860, upper bound: 10.8457536
time: 3.28 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -2.1949854, 1.7961634, -1.6539887, 1.4509407, -3.6459260, 3.4501522
1: -1.7254845, 1.7858444, -1.3412626, 1.4456468, -3.1711311, 3.1271071
2: -2.0550306, 2.0069385, -1.6044955, 1.6921459, -3.7471766, 3.6114340
3: -2.3170943, 1.7339885, -1.7273405, 1.5104665, -3.8275609, 3.4613290
4: -2.4669859, 1.8834062, -1.8218744, 1.4907380, -3.9577241, 3.7052805
5: -2.0013623, 1.8806791, -1.5200970, 1.5076841, -3.5090466, 3.4007761
6: -2.0361242, 1.9711777, -1.5676606, 1.5452360, -3.5813603, 3.5388384
7: -2.1120110, 1.9872661, -1.6012831, 1.5710124, -3.6830235, 3.5885491
8: -3.4218044, 2.6954520, -2.4827385, 2.6315432, -6.0533476, 5.1781902
9: -1.7880751, 2.0727630, -1.4086487, 1.6709894, -3.4590645, 3.4814117

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6983119, upper bound: 10.6013238
time: 3.05 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8455124, upper bound: 10.8453547
time: 1.72 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.8548315, 0.9729112, -0.5654026, 0.7410037, -1.5958352, 1.5383139
1: -0.7371748, 0.8777912, -0.5064967, 0.6246005, -1.3617754, 1.3842878
2: -0.7716476, 1.1615790, -0.5356548, 0.8421050, -1.6137526, 1.6972339
3: -0.7352794, 1.0363644, -0.4047453, 0.8206455, -1.5559249, 1.4411098
4: -0.8884984, 0.8800038, -0.5945676, 0.6272699, -1.5157683, 1.4745713
5: -0.7959436, 0.9356470, -0.5582622, 0.6852054, -1.4811490, 1.4939092
6: -0.8502627, 0.8783861, -0.5565071, 0.6262538, -1.4765165, 1.4348931
7: -0.8415049, 0.8708535, -0.5858593, 0.5956960, -1.4372008, 1.4567128
8: -1.1454779, 2.6065810, -0.5516490, 2.4494874, -3.5949655, 3.1582298
9: -0.8617782, 1.0319117, -0.6702948, 0.7642033, -1.6259815, 1.7022066

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6100872, upper bound: 10.4571540
time: 2.65 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7940340, upper bound: 10.6929824
time: 3.83 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -2.3595486, 1.9238670, -0.5353293, 0.7163302, -3.0758789, 2.4591963
1: -1.8683621, 1.9268239, -0.4822469, 0.5977703, -2.4661324, 2.4090707
2: -2.2591367, 2.1099219, -0.5208566, 0.8093985, -3.0685353, 2.6307786
3: -2.5422058, 1.7912340, -0.3713251, 0.7960217, -3.3382275, 2.1625590
4: -2.6833220, 1.9881929, -0.5694184, 0.5973306, -3.2806525, 2.5576115
5: -2.1519923, 1.9924625, -0.5313693, 0.6601417, -2.8121340, 2.5238318
6: -2.1684113, 2.1414213, -0.5281698, 0.6035422, -2.7719536, 2.6695910
7: -2.2507727, 2.1795776, -0.5575559, 0.5693669, -2.8201396, 2.7371335
8: -3.6928582, 2.7405651, -0.4956278, 2.4334347, -6.1262932, 3.2361927
9: -1.9282557, 2.1871796, -0.6555150, 0.7363492, -2.6646049, 2.8426945

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6053704, upper bound: 10.4550879
time: 2.46 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7815601, upper bound: 10.6909031
time: 2.32 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -1.4542458, 1.3336086, -1.8286511, 1.5763104, -3.0305562, 3.1622596
1: -1.1817398, 1.3012757, -1.4772263, 1.5679286, -2.7496684, 2.7785020
2: -1.3826720, 1.5937638, -1.8236774, 1.8206487, -3.2033205, 3.4174414
3: -1.4261439, 1.3886158, -1.9659264, 1.6213303, -3.0474741, 3.3545423
4: -1.5705551, 1.3333344, -2.0250397, 1.6300628, -3.2006178, 3.3583741
5: -1.3300396, 1.3638097, -1.6792562, 1.6365993, -2.9666390, 3.0430660
6: -1.4050778, 1.3695748, -1.7308002, 1.7010179, -3.1060958, 3.1003749
7: -1.3923072, 1.3741715, -1.7740963, 1.7326549, -3.1249621, 3.1482677
8: -2.1670704, 2.7040758, -2.7788346, 2.6871822, -4.8542528, 5.4829102
9: -1.2735286, 1.5020270, -1.5380309, 1.8252078, -3.0987363, 3.0400579

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7318947, upper bound: 10.6177226
time: 2.46 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8462860, upper bound: 10.8465753
time: 2.16 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -3.0257187, 2.4723048, -1.7394091, 1.5126401, -4.5383587, 4.2117138
1: -2.4504106, 2.4407337, -1.4075046, 1.5051103, -3.9555209, 3.8482382
2: -3.3347898, 2.5942631, -1.7143441, 1.7644563, -5.0992460, 4.3086071
3: -3.6078854, 2.2497952, -1.8419312, 1.5707083, -5.1785936, 4.0917263
4: -3.4680569, 2.5706856, -1.9181325, 1.5583391, -5.0263958, 4.4888182
5: -2.8108008, 2.5008090, -1.5952160, 1.5719333, -4.3827343, 4.0960250
6: -2.8072321, 2.8044040, -1.6491621, 1.6217294, -4.4289618, 4.4535661
7: -3.0377550, 2.8797894, -1.6841174, 1.6488222, -4.6865773, 4.5639067
8: -4.8393989, 2.9502950, -2.6262064, 2.6559191, -7.4953179, 5.5765014
9: -2.5222812, 2.8679452, -1.4724617, 1.7461843, -4.2684655, 4.3404069

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7247528, upper bound: 10.6159623
time: 2.01 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8461716, upper bound: 10.8461716
time: 4.06 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 7.60 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.5089659, upper bound: 10.4871174
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.6611291, upper bound: 10.7331084
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.4933578, upper bound: 10.4820897
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.6319641, upper bound: 10.7267906
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.5089659, upper bound: 10.4871174
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.6611291, upper bound: 10.7331084
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.4933578, upper bound: 10.4820897
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.6319641, upper bound: 10.7267906
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.6739700, upper bound: 10.5800244
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.8423197, upper bound: 10.8412798
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.6596500, upper bound: 10.5761317
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.8307307, upper bound: 10.8376879
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.6739700, upper bound: 10.5800244
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.8423197, upper bound: 10.8412640
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.6596500, upper bound: 10.5761317
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.8307307, upper bound: 10.8376879
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.5096823, upper bound: 10.4964377
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.6651469, upper bound: 10.7546932
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.4920426, upper bound: 10.4909146
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.6357786, upper bound: 10.7484359
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.5096823, upper bound: 10.4964377
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.6651469, upper bound: 10.7546932
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.4920426, upper bound: 10.4909146
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.6357786, upper bound: 10.7484359
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.6889039, upper bound: 10.5972798
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.8452626, upper bound: 10.8452494
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.6737220, upper bound: 10.5933411
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.8447067, upper bound: 10.8441537
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.6889039, upper bound: 10.5972798
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.8452626, upper bound: 10.8452478
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.6737220, upper bound: 10.5933411
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.8447067, upper bound: 10.8441537
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.5934016, upper bound: 10.4522011
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.7749076, upper bound: 10.6884798
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.5879611, upper bound: 10.4503066
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.7623304, upper bound: 10.6866906
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.7055027, upper bound: 10.6031621
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.8456860, upper bound: 10.8457536
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.6983119, upper bound: 10.6013238
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.8455124, upper bound: 10.8453548
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.6100872, upper bound: 10.4571540
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.7940340, upper bound: 10.6929824
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.6053704, upper bound: 10.4550879
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.7815601, upper bound: 10.6909031
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.7318947, upper bound: 10.6177226
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.8462860, upper bound: 10.8465753
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.7247528, upper bound: 10.6159623
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.8461716, upper bound: 10.8461716
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.5934016, upper bound: 10.4522011
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.7749076, upper bound: 10.6884798
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.5879611, upper bound: 10.4503066
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.7623304, upper bound: 10.6866906
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.7055027, upper bound: 10.6031621
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.8456860, upper bound: 10.8457536
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.6983119, upper bound: 10.6013238
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.8455124, upper bound: 10.8453547
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.6100872, upper bound: 10.4571540
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.7940340, upper bound: 10.6929824
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.6053704, upper bound: 10.4550879
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.7815601, upper bound: 10.6909031
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.7318947, upper bound: 10.6177226
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.8462860, upper bound: 10.8465753
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.7247528, upper bound: 10.6159623
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 8, lower bound: -10.8461716, upper bound: 10.8461716

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1834026, 0.1764473, -0.1791120, 0.0765092, -0.2599118, 0.3555593
1: -0.0800586, 0.1190268, -0.0697137, 0.0772574, -0.1573161, 0.1887406
2: -0.3262616, 0.1666474, -0.3195477, 0.0984057, -0.4246672, 0.4861951
3: -0.1103528, 0.1115013, -0.1087228, 0.0523004, -0.1626532, 0.2202242
4: -0.0976256, 0.1612447, -0.0776929, 0.0993005, -0.1969261, 0.2239298
5: -0.1709299, 0.1100812, -0.1500118, 0.0487945, -0.2197244, 0.2600929
6: -0.1728325, 0.1125856, -0.1678101, 0.0825879, -0.2554204, 0.2803957
7: -0.1775251, 0.1136709, -0.1270058, 0.0795917, -0.2571167, 0.2406767
8: 0.6282828, 2.2568166, 0.8292130, 2.2556286, -1.6273458, 1.4276036
9: -0.4277024, 0.2061759, -0.4246244, 0.0642048, -0.4919072, 0.6308002

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.3099583, upper bound: 10.2593744
time: 3.77 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.2640191, upper bound: 10.2418163
time: 2.83 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.2412965, 0.3832748, -0.3036451, 0.4737175, -0.7150140, 0.6869199
1: -0.1823967, 0.2908354, -0.2734125, 0.3755666, -0.5579634, 0.5642479
2: -0.3864003, 0.3735359, -0.4191704, 0.4976563, -0.8840566, 0.7927063
3: -0.1333454, 0.3631682, -0.1829801, 0.5457308, -0.6790763, 0.5461484
4: -0.3098158, 0.2674221, -0.3832262, 0.3519857, -0.6618015, 0.6506482
5: -0.2422853, 0.3310988, -0.3081066, 0.4302544, -0.6725397, 0.6392055
6: -0.2120025, 0.2956091, -0.2908444, 0.3761039, -0.5881064, 0.5864536
7: -0.2932711, 0.2248116, -0.3596906, 0.3087701, -0.6020412, 0.5845022
8: 0.1954161, 2.3098345, -0.0131831, 2.3599081, -2.1644921, 2.3230176
9: -0.5150667, 0.3840154, -0.5432958, 0.4861261, -1.0011928, 0.9273112

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4518468, upper bound: 10.4500155
time: 2.69 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3906070, upper bound: 10.4311742
time: 2.42 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.3397722, 0.5972340, -0.1771812, 0.0749006, -0.4146729, 0.7744151
1: -0.2790638, 0.3868424, -0.0691073, 0.0750459, -0.3541096, 0.4445450
2: -0.4223440, 0.5238652, -0.3169151, 0.0952817, -0.5176257, 0.8407803
3: -0.2119896, 0.4432561, -0.1077644, 0.0511214, -0.2631110, 0.5510206
4: -0.4098796, 0.3799648, -0.0760680, 0.0972200, -0.5070996, 0.4560328
5: -0.3434416, 0.4803380, -0.1485906, 0.0464545, -0.3898961, 0.6136907
6: -0.3051775, 0.4398654, -0.1661121, 0.0810642, -0.3862417, 0.5789990
7: -0.3981190, 0.3790659, -0.1257960, 0.0782662, -0.4763852, 0.5048618
8: -0.0507057, 2.3272233, 0.8348623, 2.2462804, -2.2969861, 1.4923611
9: -0.5478314, 0.5362103, -0.4220487, 0.0605982, -0.6084297, 0.9582590

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.2949447, upper bound: 10.2548473
time: 2.41 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.2360759, upper bound: 10.2317539
time: 2.50 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.5264078, 0.7847248, -0.2946350, 0.4594832, -0.9858910, 1.0793599
1: -0.4650661, 0.5355406, -0.2630968, 0.3646730, -0.8297391, 0.7986374
2: -0.5103696, 0.7826304, -0.4130050, 0.4811913, -0.9915609, 1.1956353
3: -0.3217826, 0.7515637, -0.1749537, 0.5289830, -0.8507656, 0.9265174
4: -0.5464209, 0.6123769, -0.3732445, 0.3407606, -0.8871815, 0.9856213
5: -0.4986731, 0.6824483, -0.2984511, 0.4169059, -0.9155791, 0.9808994
6: -0.4842524, 0.6405575, -0.2799693, 0.3649483, -0.8492007, 0.9205268
7: -0.5525504, 0.5884560, -0.3509088, 0.2966950, -0.8492454, 0.9393648
8: -0.4955102, 2.3866658, 0.0119740, 2.3463500, -2.8418603, 2.3746917
9: -0.6577470, 0.7249359, -0.5375638, 0.4721926, -1.1299396, 1.2624997

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 92

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6314970, upper bound: 10.7267906
time: 1.71 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6314970, upper bound: 10.7267906
time: 2.21 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.1899148, 0.2068106, -0.1885968, 0.1378345, -0.3277493, 0.3954074
1: -0.0892890, 0.1332570, -0.0766421, 0.1032509, -0.1925398, 0.2098991
2: -0.3332822, 0.1914296, -0.3345287, 0.1380781, -0.4713603, 0.5259584
3: -0.1136551, 0.1309398, -0.1139035, 0.0845147, -0.1981698, 0.2448434
4: -0.1089217, 0.1757824, -0.0923967, 0.1447089, -0.2536305, 0.2681790
5: -0.1783367, 0.1316034, -0.1662403, 0.0816970, -0.2600338, 0.2978436
6: -0.1767288, 0.1303246, -0.1764469, 0.0970212, -0.2737499, 0.3067715
7: -0.1936997, 0.1288736, -0.1578196, 0.1043441, -0.2980438, 0.2866932
8: 0.5739703, 2.2793279, 0.6962041, 2.2994108, -1.7254405, 1.5831238
9: -0.4365484, 0.2321771, -0.4416606, 0.1696829, -0.6062313, 0.6738377

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3166770, upper bound: 10.2594505
time: 3.12 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.2656324, upper bound: 10.2418163
time: 2.67 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.2554555, 0.4138108, -0.3921732, 0.5807011, -0.8361566, 0.8059841
1: -0.1997797, 0.3087630, -0.3619731, 0.4567900, -0.6565697, 0.6707362
2: -0.3954350, 0.4037646, -0.4658814, 0.6251750, -1.0206100, 0.8696460
3: -0.1432922, 0.3895264, -0.2491045, 0.6592833, -0.8025755, 0.6386308
4: -0.3268787, 0.2838901, -0.4563432, 0.4528587, -0.7797374, 0.7402333
5: -0.2570049, 0.3529486, -0.3925681, 0.5408685, -0.7978733, 0.7455167
6: -0.2262357, 0.3170287, -0.3798470, 0.4777821, -0.7040178, 0.6968757
7: -0.3100711, 0.2442461, -0.4366936, 0.4254894, -0.7355605, 0.6809398
8: 0.1448242, 2.3336868, -0.2151144, 2.4129944, -2.2681701, 2.5488012
9: -0.5253584, 0.4099215, -0.5897624, 0.5951633, -1.1205217, 0.9996840

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4573517, upper bound: 10.4506307
time: 3.36 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3907717, upper bound: 10.4317282
time: 45.55 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.3656377, 0.6285113, -0.1850634, 0.1285653, -0.4942030, 0.8135747
1: -0.3033233, 0.4071929, -0.0747519, 0.0986975, -0.4020209, 0.4819449
2: -0.4350048, 0.5593778, -0.3307958, 0.1320857, -0.5670905, 0.8901737
3: -0.2281181, 0.4774151, -0.1124245, 0.0777100, -0.3058281, 0.5898396
4: -0.4303830, 0.4117566, -0.0888105, 0.1403531, -0.5707361, 0.5005671
5: -0.3662443, 0.5072734, -0.1641309, 0.0764610, -0.4427052, 0.6661026
6: -0.3287994, 0.4700850, -0.1745906, 0.0928706, -0.4216701, 0.6271623
7: -0.4201081, 0.4106033, -0.1519522, 0.1020224, -0.5221305, 0.5625556
8: -0.1135096, 2.3471603, 0.7089939, 2.2861819, -2.3996916, 1.6381664
9: -0.5630724, 0.5654861, -0.4370255, 0.1610476, -0.7241200, 1.0025115

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4933578, upper bound: 10.4820897
time: 2.81 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4933578, upper bound: 10.4820897
time: 2.72 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.5621834, 0.8213363, -0.3753376, 0.5625460, -1.1247294, 1.1966739
1: -0.4898159, 0.5601392, -0.3452935, 0.4419125, -0.9317284, 0.9054327
2: -0.5337468, 0.8265116, -0.4561038, 0.6026912, -1.1364379, 1.2826154
3: -0.3636509, 0.7850611, -0.2367326, 0.6359648, -0.9996156, 1.0217936
4: -0.5748242, 0.6456550, -0.4429095, 0.4310102, -1.0058343, 1.0885645
5: -0.5342237, 0.7113045, -0.3762266, 0.5223534, -1.0565771, 1.0875311
6: -0.5345932, 0.6701932, -0.3606584, 0.4593026, -0.9938958, 1.0308516
7: -0.5748042, 0.6304934, -0.4223571, 0.4048313, -0.9796355, 1.0528505
8: -0.5616231, 2.4076796, -0.1758693, 2.3969479, -2.9585710, 2.5835490
9: -0.6795173, 0.7702864, -0.5796244, 0.5757311, -1.2552484, 1.3499109

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 92

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6314970, upper bound: 10.7267906
time: 1.98 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6314970, upper bound: 10.7267906
time: 2.23 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 5.69 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 8, lower bound: -10.3099583, upper bound: 10.2593744
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 8, lower bound: -10.2640191, upper bound: 10.2418163
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.69
Output dim: 8, lower bound: -10.4518468, upper bound: 10.4500155
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.69
Output dim: 8, lower bound: -10.3906070, upper bound: 10.4311742
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 8, lower bound: -10.2949447, upper bound: 10.2548473
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 8, lower bound: -10.2360759, upper bound: 10.2317539
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.69
Output dim: 8, lower bound: -10.6314970, upper bound: 10.7267906
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.69
Output dim: 8, lower bound: -10.6314970, upper bound: 10.7267906
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.69
Output dim: 8, lower bound: -10.3166770, upper bound: 10.2594505
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 8, lower bound: -10.2656324, upper bound: 10.2418163
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.69
Output dim: 8, lower bound: -10.4573517, upper bound: 10.4506307
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.69
Output dim: 8, lower bound: -10.3907717, upper bound: 10.4317282
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.69
Output dim: 8, lower bound: -10.4933578, upper bound: 10.4820897
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.69
Output dim: 8, lower bound: -10.4933578, upper bound: 10.4820897
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.69
Output dim: 8, lower bound: -10.6314970, upper bound: 10.7267906
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.69
Output dim: 8, lower bound: -10.6314970, upper bound: 10.7267906
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.69
Output dim: 8, lower bound: -10.6739700, upper bound: 10.5800244
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.69
Output dim: 8, lower bound: -10.8423197, upper bound: 10.8412798
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.69
Output dim: 8, lower bound: -10.6596500, upper bound: 10.5761317
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.69
Output dim: 8, lower bound: -10.8307307, upper bound: 10.8376879
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.69
Output dim: 8, lower bound: -10.6739700, upper bound: 10.5800244
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.69
Output dim: 8, lower bound: -10.8423197, upper bound: 10.8412640
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.69
Output dim: 8, lower bound: -10.6596500, upper bound: 10.5761317
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.69
Output dim: 8, lower bound: -10.8307307, upper bound: 10.8376879
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.69
Output dim: 8, lower bound: -10.5096823, upper bound: 10.4964377
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.69
Output dim: 8, lower bound: -10.6651469, upper bound: 10.7546932
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.69
Output dim: 8, lower bound: -10.4920426, upper bound: 10.4909146
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.69
Output dim: 8, lower bound: -10.6357786, upper bound: 10.7484359
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.69
Output dim: 8, lower bound: -10.5096823, upper bound: 10.4964377
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.69
Output dim: 8, lower bound: -10.6651469, upper bound: 10.7546932
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.69
Output dim: 8, lower bound: -10.4920426, upper bound: 10.4909146
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.69
Output dim: 8, lower bound: -10.6357786, upper bound: 10.7484359
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.69
Output dim: 8, lower bound: -10.6889039, upper bound: 10.5972798
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.69
Output dim: 8, lower bound: -10.8452626, upper bound: 10.8452494
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.69
Output dim: 8, lower bound: -10.6737220, upper bound: 10.5933411
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.69
Output dim: 8, lower bound: -10.8447067, upper bound: 10.8441537
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.69
Output dim: 8, lower bound: -10.6889039, upper bound: 10.5972798
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.69
Output dim: 8, lower bound: -10.8452626, upper bound: 10.8452478
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.69
Output dim: 8, lower bound: -10.6737220, upper bound: 10.5933411
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.69
Output dim: 8, lower bound: -10.8447067, upper bound: 10.8441537
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.69
Output dim: 8, lower bound: -10.5934016, upper bound: 10.4522011
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.69
Output dim: 8, lower bound: -10.7749076, upper bound: 10.6884798
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.69
Output dim: 8, lower bound: -10.5879611, upper bound: 10.4503066
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.69
Output dim: 8, lower bound: -10.7623304, upper bound: 10.6866906
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.69
Output dim: 8, lower bound: -10.7055027, upper bound: 10.6031621
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.69
Output dim: 8, lower bound: -10.8456860, upper bound: 10.8457536
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.69
Output dim: 8, lower bound: -10.6983119, upper bound: 10.6013238
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.69
Output dim: 8, lower bound: -10.8455124, upper bound: 10.8453548
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.69
Output dim: 8, lower bound: -10.6100872, upper bound: 10.4571540
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.69
Output dim: 8, lower bound: -10.7940340, upper bound: 10.6929824
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.69
Output dim: 8, lower bound: -10.6053704, upper bound: 10.4550879
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.69
Output dim: 8, lower bound: -10.7815601, upper bound: 10.6909031
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.69
Output dim: 8, lower bound: -10.7318947, upper bound: 10.6177226
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.69
Output dim: 8, lower bound: -10.8462860, upper bound: 10.8465753
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.69
Output dim: 8, lower bound: -10.7247528, upper bound: 10.6159623
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.69
Output dim: 8, lower bound: -10.8461716, upper bound: 10.8461716
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.69
Output dim: 8, lower bound: -10.5934016, upper bound: 10.4522011
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.69
Output dim: 8, lower bound: -10.7749076, upper bound: 10.6884798
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.69
Output dim: 8, lower bound: -10.5879611, upper bound: 10.4503066
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.69
Output dim: 8, lower bound: -10.7623304, upper bound: 10.6866906
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.69
Output dim: 8, lower bound: -10.7055027, upper bound: 10.6031621
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.69
Output dim: 8, lower bound: -10.8456860, upper bound: 10.8457536
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.69
Output dim: 8, lower bound: -10.6983119, upper bound: 10.6013238
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.69
Output dim: 8, lower bound: -10.8455124, upper bound: 10.8453547
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.69
Output dim: 8, lower bound: -10.6100872, upper bound: 10.4571540
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.69
Output dim: 8, lower bound: -10.7940340, upper bound: 10.6929824
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.69
Output dim: 8, lower bound: -10.6053704, upper bound: 10.4550879
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.69
Output dim: 8, lower bound: -10.7815601, upper bound: 10.6909031
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.69
Output dim: 8, lower bound: -10.7318947, upper bound: 10.6177226
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.69
Output dim: 8, lower bound: -10.8462860, upper bound: 10.8465753
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.69
Output dim: 8, lower bound: -10.7247528, upper bound: 10.6159623
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.69
Output dim: 8, lower bound: -10.8461716, upper bound: 10.8461716
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=12.009641647338867
rel_dist={8: [-10.85331894689149, 10.853318913028158]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 92

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8515843, upper bound: 10.8514032
time: 5.08 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8518150, upper bound: 10.8518150
time: 4.42 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 9.66 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 9.66
Output dim: 8, lower bound: -10.8515843, upper bound: 10.8514032
IS_A2, status: Status.UNKNOWN, split count: 1, time: 9.66
Output dim: 8, lower bound: -10.8518150, upper bound: 10.8518150

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -2.9945498, 2.4854207, -3.4675641, 2.8765092, -5.8710589, 5.9529848
1: -2.4506655, 2.4260209, -2.9101114, 2.7755547, -5.2262201, 5.3361320
2: -3.2870402, 2.6394088, -3.8645773, 2.9711130, -6.2581530, 6.5039864
3: -3.5422220, 2.3601861, -4.1797991, 2.6550899, -6.1973119, 6.5399852
4: -3.3720222, 2.6054800, -3.9647605, 3.0023437, -6.3743658, 6.5702405
5: -2.8525395, 2.5022659, -3.3423719, 2.8772759, -5.7298155, 5.8446379
6: -2.8254833, 2.7629449, -3.2796764, 3.2031515, -6.0286350, 6.0426216
7: -2.9726825, 2.8755846, -3.4648035, 3.3636727, -6.3363552, 6.3403883
8: -4.6693630, 3.0776982, -5.4010711, 3.2388330, -7.9081960, 8.4787693
9: -2.5680308, 2.8494966, -3.0075715, 3.2855663, -5.8535972, 5.8570681

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8500766, upper bound: 10.8498973
time: 4.87 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8501112, upper bound: 10.8499351
time: 4.25 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -3.6680012, 3.0478530, -3.6003656, 2.9886067, -6.6566076, 6.6482186
1: -3.1057496, 2.9268103, -3.0417533, 2.8774354, -5.9831848, 5.9685636
2: -4.1053696, 3.1162860, -4.0287738, 3.0631433, -7.1685128, 7.1450596
3: -4.4386749, 2.7713633, -4.3698554, 2.7481973, -7.1868725, 7.1412182
4: -4.2074580, 3.1692934, -4.1355762, 3.1175027, -7.3249607, 7.3048697
5: -3.5509248, 3.0351212, -3.4831014, 2.9859633, -6.5368881, 6.5182209
6: -3.4747064, 3.3891344, -3.4105644, 3.3304861, -6.8051910, 6.7996988
7: -3.6690035, 3.5624459, -3.6068439, 3.5026700, -7.1716738, 7.1692896
8: -5.7219377, 3.3660746, -5.6054168, 3.2970796, -9.0190172, 8.9714890
9: -3.1934023, 3.4740181, -3.1355391, 3.4113989, -6.6048012, 6.6095562

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8503476, upper bound: 10.8502912
time: 4.37 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8503126, upper bound: 10.8503126
time: 9.35 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 15.16 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 15.16
Output dim: 8, lower bound: -10.8500766, upper bound: 10.8498973
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 15.16
Output dim: 8, lower bound: -10.8501112, upper bound: 10.8499351
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 15.16
Output dim: 8, lower bound: -10.8503476, upper bound: 10.8502912
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 15.16
Output dim: 8, lower bound: -10.8503126, upper bound: 10.8503126

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -1.5412021, 1.3843178, -1.5427128, 1.3840736, -2.9252758, 2.9270306
1: -1.2478821, 1.3638480, -1.2527486, 1.3651376, -2.6130197, 2.6165967
2: -1.4677565, 1.6153791, -1.4641082, 1.5963488, -3.0641053, 3.0794873
3: -1.5536703, 1.4384565, -1.5684260, 1.4342909, -2.9879613, 3.0068827
4: -1.6745836, 1.4071953, -1.6854768, 1.4092901, -3.0838737, 3.0926721
5: -1.4148538, 1.4224372, -1.4226217, 1.4228230, -2.8376768, 2.8450589
6: -1.4739804, 1.4424994, -1.4715596, 1.4449676, -2.9189482, 2.9140592
7: -1.4820031, 1.4524717, -1.4879954, 1.4605652, -2.9425683, 2.9404671
8: -2.2980630, 2.6832056, -2.3032236, 2.6718597, -4.9699230, 4.9864292
9: -1.3262157, 1.5712144, -1.3265111, 1.5745573, -2.9007730, 2.8977256

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8332841, upper bound: 10.8417885
time: 3.33 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8484843, upper bound: 10.8483793
time: 5.46 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -1.7434754, 1.5293049, -2.3916759, 2.0084593, -3.7519348, 3.9209809
1: -1.4063792, 1.5058761, -1.9347035, 1.9816840, -3.3880632, 3.4405794
2: -1.7240989, 1.7814090, -2.5368788, 2.1960611, -3.9201601, 4.3182878
3: -1.8273036, 1.5781411, -2.7319479, 1.9561021, -3.7834058, 4.3100891
4: -1.9022294, 1.5671549, -2.6815929, 2.0960226, -3.9982519, 4.2487478
5: -1.5960102, 1.5718076, -2.2315023, 2.0520768, -3.6480870, 3.8033099
6: -1.6669428, 1.6225477, -2.2572417, 2.2156110, -3.8825538, 3.8797894
7: -1.6787665, 1.6348361, -2.3598320, 2.2766018, -3.9553683, 3.9946680
8: -2.6387472, 2.7388935, -3.7346678, 2.8918090, -5.5305562, 6.4735613
9: -1.4794488, 1.7484398, -2.0281460, 2.3224649, -3.8019137, 3.7765858

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8367311, upper bound: 10.8425341
time: 3.01 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8487570, upper bound: 10.8485548
time: 3.69 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -2.1247478, 1.7987877, -1.6417656, 1.4497175, -3.5744653, 3.4405532
1: -1.7164754, 1.8007530, -1.3346362, 1.4411354, -3.1576109, 3.1353893
2: -2.1830628, 1.9952677, -1.5799146, 1.6590519, -3.8421147, 3.5751824
3: -2.3743002, 1.7915000, -1.7166711, 1.5049717, -3.8792720, 3.5081711
4: -2.3708091, 1.8764683, -1.8035454, 1.4888620, -3.8596711, 3.6800137
5: -1.9622750, 1.8548110, -1.5158985, 1.4976747, -3.4599497, 3.3707094
6: -2.0061753, 1.9745511, -1.5602719, 1.5349798, -3.5411551, 3.5348229
7: -2.0786653, 2.0129178, -1.5877794, 1.5540950, -3.6327603, 3.6006970
8: -3.3007159, 2.8547273, -2.4710965, 2.6911170, -5.9918327, 5.3258238
9: -1.7904391, 2.0915890, -1.3997310, 1.6645845, -3.4550238, 3.4913201

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8503476, upper bound: 10.8502912
time: 9.99 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8503476, upper bound: 10.8502913
time: 3.95 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -2.3235788, 1.9538214, -2.5012212, 2.0933247, -4.4169035, 4.4550428
1: -1.8799505, 1.9388784, -2.0278986, 2.0633426, -3.9432931, 3.9667768
2: -2.4514494, 2.1641703, -2.6724238, 2.2682962, -4.7197456, 4.8365941
3: -2.6419506, 1.9311829, -2.8902085, 2.0306323, -4.6725826, 4.8213911
4: -2.5942898, 2.0413616, -2.8100729, 2.1891317, -4.7834215, 4.8514347
5: -2.1560919, 2.0047474, -2.3509288, 2.1325727, -4.2886648, 4.3556762
6: -2.2005167, 2.1545496, -2.3554320, 2.3159957, -4.5165124, 4.5099816
7: -2.2837088, 2.2001359, -2.4727566, 2.3902297, -4.6739388, 4.6728926
8: -3.6297746, 2.9235787, -3.9074121, 2.9242697, -6.5540442, 6.8309908
9: -1.9670525, 2.2671919, -2.1263056, 2.4222331, -4.3892856, 4.3934975

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8503126, upper bound: 10.8503128
time: 3.50 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8503117, upper bound: 10.8503119
time: 8.05 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 12.94 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 12.94
Output dim: 8, lower bound: -10.8332841, upper bound: 10.8417885
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 12.94
Output dim: 8, lower bound: -10.8484843, upper bound: 10.8483793
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 12.94
Output dim: 8, lower bound: -10.8367311, upper bound: 10.8425341
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 12.94
Output dim: 8, lower bound: -10.8487570, upper bound: 10.8485548
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 12.94
Output dim: 8, lower bound: -10.8503476, upper bound: 10.8502912
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 12.94
Output dim: 8, lower bound: -10.8503476, upper bound: 10.8502913
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 12.94
Output dim: 8, lower bound: -10.8503126, upper bound: 10.8503128
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 12.94
Output dim: 8, lower bound: -10.8503117, upper bound: 10.8503119

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.3033301, 0.5009193, -0.4347709, 0.6336210, -0.9369511, 0.9356902
1: -0.2565477, 0.3699337, -0.3949790, 0.4849157, -0.7414634, 0.7649127
2: -0.4237015, 0.4730245, -0.4869479, 0.6524806, -1.0761821, 0.9599724
3: -0.1788631, 0.4883424, -0.2737954, 0.6933329, -0.8721960, 0.7621378
4: -0.3822553, 0.3414720, -0.4840193, 0.5051310, -0.8873863, 0.8254913
5: -0.3092260, 0.4229606, -0.4356337, 0.5774351, -0.8866611, 0.8585943
6: -0.2799601, 0.3817134, -0.4236736, 0.5220466, -0.8020067, 0.8053869
7: -0.3612310, 0.3100859, -0.4696781, 0.4692780, -0.8305089, 0.7797639
8: 0.0161316, 2.4149513, -0.2871199, 2.4672818, -2.4511502, 2.7020712
9: -0.5553781, 0.4857947, -0.6173273, 0.6313142, -1.1866922, 1.1031220

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7680343, upper bound: 10.8027016
time: 3.64 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7676869, upper bound: 10.7995298
time: 2.90 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.6615117, 0.8256562, -0.7885386, 0.9213811, -1.5828929, 1.6141949
1: -0.5885813, 0.7170686, -0.6948274, 0.8293065, -1.4178878, 1.4118960
2: -0.6206705, 0.9812637, -0.7143534, 1.0825403, -1.7032108, 1.6956171
3: -0.5068913, 0.9076658, -0.6615426, 1.0216218, -1.5285131, 1.5692084
4: -0.6843212, 0.7182095, -0.8100107, 0.8379004, -1.5222216, 1.5282202
5: -0.6450727, 0.7683245, -0.7527391, 0.8825316, -1.5276043, 1.5210636
6: -0.6738904, 0.7081649, -0.7913052, 0.8155885, -1.4894788, 1.4994702
7: -0.6687038, 0.6773407, -0.7904302, 0.8055669, -1.4742707, 1.4677708
8: -0.7727322, 2.5620189, -1.0101616, 2.5742028, -3.3469350, 3.5721805
9: -0.7408921, 0.8630701, -0.8180741, 0.9737084, -1.7146004, 1.6811442

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8484843, upper bound: 10.8483793
time: 4.03 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8484843, upper bound: 10.8483793
time: 3.78 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.3068296, 0.5046486, -0.6639615, 0.8491679, -1.1559975, 1.1686101
1: -0.2579511, 0.3698316, -0.5846307, 0.7075744, -0.9655255, 0.9544623
2: -0.4313770, 0.5079864, -0.6121228, 0.9557352, -1.3871121, 1.1201092
3: -0.1824338, 0.4844498, -0.5185983, 0.8788131, -1.0612470, 1.0030481
4: -0.3840692, 0.3433982, -0.6948328, 0.7232252, -1.1072944, 1.0382309
5: -0.3082154, 0.4259034, -0.6457686, 0.7726705, -1.0808859, 1.0716720
6: -0.2838933, 0.3825286, -0.6635957, 0.7103506, -0.9942439, 1.0461243
7: -0.3613067, 0.3121558, -0.6792555, 0.6956475, -1.0569541, 0.9914113
8: -0.0214524, 2.4292402, -0.7725388, 2.5626485, -2.5841010, 3.2017791
9: -0.5603834, 0.4934707, -0.7390887, 0.8645127, -1.4248961, 1.2325593

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8367311, upper bound: 10.8425341
time: 3.27 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8367311, upper bound: 10.8425341
time: 2.31 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.8054641, 0.9367293, -1.5861409, 1.4292789, -2.2347429, 2.5228701
1: -0.6990361, 0.8378878, -1.2822019, 1.3838038, -2.0828400, 2.1200895
2: -0.7339010, 1.1549634, -1.5325036, 1.6571686, -2.3910697, 2.6874671
3: -0.6634372, 1.0071825, -1.5984378, 1.4384910, -2.1019282, 2.6056204
4: -0.8281921, 0.8388489, -1.7333827, 1.4375681, -2.2657602, 2.5722315
5: -0.7596627, 0.8971661, -1.4561983, 1.4523323, -2.2119949, 2.3533645
6: -0.8118343, 0.8326406, -1.5157355, 1.4821268, -2.2939610, 2.3483760
7: -0.7945108, 0.8173739, -1.5273900, 1.4946792, -2.2891901, 2.3447640
8: -1.0649683, 2.6011367, -2.3852341, 2.7230279, -3.7879963, 4.9863710
9: -0.8320751, 0.9897887, -1.3616405, 1.6113976, -2.4434726, 2.3514290

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8487568, upper bound: 10.8485549
time: 3.51 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8487568, upper bound: 10.8485549
time: 3.95 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -1.3972495, 1.2800345, -1.2322521, 1.1710618, -2.5683112, 2.5122867
1: -1.1434765, 1.2795658, -1.0202308, 1.1536188, -2.2970953, 2.2997966
2: -1.3048089, 1.5294175, -1.1343610, 1.3996609, -2.7044697, 2.6637785
3: -1.3730147, 1.3922766, -1.1834477, 1.2872326, -2.6602473, 2.5757244
4: -1.4990070, 1.2997576, -1.3145734, 1.1749828, -2.6739898, 2.6143310
5: -1.2848738, 1.3263178, -1.1399255, 1.2042933, -2.4891672, 2.4662433
6: -1.3505297, 1.3214672, -1.1953781, 1.1791595, -2.5296893, 2.5168452
7: -1.3361217, 1.3246170, -1.1898614, 1.1885655, -2.5246873, 2.5144784
8: -2.0601625, 2.6800482, -1.7679799, 2.5846782, -4.6448407, 4.4480281
9: -1.2251536, 1.4557114, -1.1027753, 1.3225446, -2.5476980, 2.5584867

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8457456, upper bound: 10.8459056
time: 15.64 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8488079, upper bound: 10.8488623
time: 3.48 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -1.8876481, 1.6225243, -1.4693859, 1.3243263, -3.2119744, 3.0919101
1: -1.5235884, 1.6265726, -1.1988996, 1.3163873, -2.8399758, 2.8254724
2: -1.8887882, 1.8401612, -1.3762059, 1.5488849, -3.4376731, 3.2163672
3: -2.0449083, 1.6536782, -1.4796183, 1.4089932, -3.4539015, 3.1332965
4: -2.0894380, 1.6839281, -1.5980670, 1.3536661, -3.4431040, 3.2819953
5: -1.7391579, 1.6787496, -1.3552639, 1.3728468, -3.1120048, 3.0340135
6: -1.7869570, 1.7567408, -1.4054005, 1.3803352, -3.1672921, 3.1621413
7: -1.8303158, 1.7839822, -1.4154274, 1.3958474, -3.2261634, 3.1994095
8: -2.8942285, 2.7786734, -2.1740232, 2.6420646, -5.5362930, 4.9526968
9: -1.5860312, 1.8808438, -1.2706354, 1.5123920, -3.0984232, 3.1514792

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8457456, upper bound: 10.8459056
time: 5.17 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8488079, upper bound: 10.8488626
time: 10.45 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -1.6366214, 1.4456291, -2.0565886, 1.7421763, -3.3787975, 3.5022178
1: -1.3276633, 1.4457343, -1.6568903, 1.7431509, -3.0708141, 3.1026244
2: -1.5983559, 1.7229198, -2.1120756, 1.9763347, -3.5746906, 3.8349953
3: -1.6902802, 1.5452168, -2.2805851, 1.7698534, -3.4601336, 3.8258018
4: -1.7727411, 1.4861259, -2.2880669, 1.8165661, -3.5893073, 3.7741928
5: -1.4954759, 1.5029227, -1.8919998, 1.8049783, -3.3004541, 3.3949225
6: -1.5714834, 1.5321118, -1.9459118, 1.9088714, -3.4803548, 3.4780235
7: -1.5704186, 1.5393517, -2.0078385, 1.9458796, -3.5162983, 3.5471902
8: -2.4617751, 2.7333279, -3.1671727, 2.7692606, -5.2310357, 5.9005003
9: -1.4009156, 1.6622075, -1.7250576, 2.0269499, -3.4278655, 3.3872652

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8455881, upper bound: 10.8459070
time: 2.35 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8489836, upper bound: 10.8489837
time: 3.10 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -2.0785227, 1.7628640, -2.3092275, 1.9363235, -4.0148463, 4.0720916
1: -1.6748991, 1.7597501, -1.8690253, 1.9258419, -3.6007409, 3.6287756
2: -2.1390142, 2.0013871, -2.4333866, 2.1412613, -4.2802753, 4.4347734
3: -2.3037641, 1.7860193, -2.6302192, 1.9184813, -4.2222452, 4.4162388
4: -2.3042941, 1.8358390, -2.5862954, 2.0286007, -4.3328948, 4.4221344
5: -1.9115433, 1.8211638, -2.1454749, 1.9939976, -3.9055409, 3.9666386
6: -1.9720315, 1.9286772, -2.1801448, 2.1411467, -4.1131783, 4.1088219
7: -2.0256190, 1.9569960, -2.2750626, 2.1967387, -4.2223577, 4.2320585
8: -3.2140336, 2.8406506, -3.5955327, 2.8531547, -6.0671883, 6.4361830
9: -1.7430817, 2.0485487, -1.9541086, 2.2512703, -3.9943519, 4.0026574

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8455881, upper bound: 10.8459070
time: 3.20 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8489836, upper bound: 10.8489836
time: 5.28 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 9.92 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 9.92
Output dim: 8, lower bound: -10.7680343, upper bound: 10.8027016
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 9.92
Output dim: 8, lower bound: -10.7676869, upper bound: 10.7995298
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 9.92
Output dim: 8, lower bound: -10.8484843, upper bound: 10.8483793
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 9.92
Output dim: 8, lower bound: -10.8484843, upper bound: 10.8483793
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 9.92
Output dim: 8, lower bound: -10.8367311, upper bound: 10.8425341
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 9.92
Output dim: 8, lower bound: -10.8367311, upper bound: 10.8425341
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 9.92
Output dim: 8, lower bound: -10.8487568, upper bound: 10.8485549
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 9.92
Output dim: 8, lower bound: -10.8487568, upper bound: 10.8485549
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 9.92
Output dim: 8, lower bound: -10.8457456, upper bound: 10.8459056
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 9.92
Output dim: 8, lower bound: -10.8488079, upper bound: 10.8488623
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 9.92
Output dim: 8, lower bound: -10.8457456, upper bound: 10.8459056
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 9.92
Output dim: 8, lower bound: -10.8488079, upper bound: 10.8488626
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 9.92
Output dim: 8, lower bound: -10.8455881, upper bound: 10.8459070
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 9.92
Output dim: 8, lower bound: -10.8489836, upper bound: 10.8489837
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 9.92
Output dim: 8, lower bound: -10.8455881, upper bound: 10.8459070
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 9.92
Output dim: 8, lower bound: -10.8489836, upper bound: 10.8489836

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.2634753, 0.4331239, -0.3349878, 0.5281829, -0.7916582, 0.7681117
1: -0.2123562, 0.3215431, -0.3006638, 0.4033735, -0.6157297, 0.6222069
2: -0.3995377, 0.3999001, -0.4365011, 0.5285087, -0.9280463, 0.8364012
3: -0.1492424, 0.4181051, -0.2062524, 0.5686896, -0.7179320, 0.6243575
4: -0.3373448, 0.2963237, -0.4105709, 0.3818018, -0.7191465, 0.7068946
5: -0.2674518, 0.3666346, -0.3389705, 0.4737950, -0.7412468, 0.7056051
6: -0.2374928, 0.3305657, -0.3181194, 0.4158093, -0.6533021, 0.6486850
7: -0.3208042, 0.2556733, -0.3881430, 0.3547681, -0.6755723, 0.6438163
8: 0.1307922, 2.3589842, -0.0702823, 2.3878808, -2.2570887, 2.4292665
9: -0.5332136, 0.4225212, -0.5631094, 0.5267708, -1.0599844, 0.9856306

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7680343, upper bound: 10.8027016
time: 4.29 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7680343, upper bound: 10.8027016
time: 3.57 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.2620566, 0.4323660, -0.7582306, 0.9906434, -1.2527000, 1.1905966
1: -0.2104947, 0.3200180, -0.6416180, 0.8212913, -1.0317860, 0.9616359
2: -0.3977387, 0.3970227, -0.6348675, 0.9935679, -1.3913066, 1.0318903
3: -0.1487240, 0.4137803, -0.4803025, 1.0227640, -1.1714880, 0.8940828
4: -0.3361829, 0.2948152, -0.7025636, 0.8521049, -1.1882877, 0.9973789
5: -0.2664331, 0.3655670, -0.7456697, 0.8658075, -1.1322407, 1.1112367
6: -0.2360903, 0.3294200, -0.7623510, 0.8086687, -1.0447590, 1.0917709
7: -0.3199244, 0.2547413, -0.7604851, 0.7800712, -1.0999956, 1.0152264
8: 0.1363361, 2.3523393, -0.8974227, 2.4678864, -2.3315504, 3.2497621
9: -0.5310623, 0.4206717, -0.8288423, 0.9197922, -1.4508545, 1.2495140

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7676869, upper bound: 10.7995298
time: 4.22 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7676869, upper bound: 10.7995298
time: 2.97 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.5052357, 0.6807394, -0.4639375, 0.6376657, -1.1429014, 1.1446769
1: -0.4594569, 0.5724559, -0.4339293, 0.5352571, -0.9947140, 1.0063851
2: -0.5355271, 0.8032890, -0.5146303, 0.7450003, -1.2805274, 1.3179193
3: -0.3396125, 0.7834533, -0.3095367, 0.7715531, -1.1111655, 1.0929900
4: -0.5424656, 0.5711019, -0.5110931, 0.5351535, -1.0776191, 1.0821950
5: -0.4983489, 0.6429588, -0.4625286, 0.6113017, -1.1096506, 1.1054873
6: -0.5180131, 0.5730734, -0.4736696, 0.5380102, -1.0560232, 1.0467430
7: -0.5272545, 0.5298163, -0.4925026, 0.4948169, -1.0220714, 1.0223188
8: -0.4639552, 2.4817235, -0.3881780, 2.4595227, -2.9234779, 2.8699017
9: -0.6581073, 0.7103513, -0.6350753, 0.6748389, -1.3329462, 1.3454267

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8475090, upper bound: 10.8474892
time: 3.21 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8474434, upper bound: 10.8473387
time: 3.71 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.5876770, 0.7579806, -0.6707143, 0.8229774, -1.4106544, 1.4286948
1: -0.5264745, 0.6530640, -0.6024979, 0.7294390, -1.2559135, 1.2555619
2: -0.5759048, 0.9013646, -0.6213479, 0.9760303, -1.5519352, 1.5227125
3: -0.4216719, 0.8503646, -0.5221077, 0.9375398, -1.3592117, 1.3724723
4: -0.6157273, 0.6456615, -0.6942980, 0.7314230, -1.3471503, 1.3399596
5: -0.5795171, 0.7048137, -0.6558735, 0.7794178, -1.3589349, 1.3606873
6: -0.6024343, 0.6458172, -0.6824707, 0.7126095, -1.3150438, 1.3282878
7: -0.5990110, 0.6043018, -0.6807916, 0.6913648, -1.2903758, 1.2850934
8: -0.6241788, 2.5251746, -0.7810371, 2.5258350, -3.1500139, 3.3062117
9: -0.7002425, 0.7921811, -0.7422490, 0.8682822, -1.5685247, 1.5344301

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8475090, upper bound: 10.8474891
time: 6.40 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8474434, upper bound: 10.8473387
time: 2.59 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.2649117, 0.4199429, -0.4068604, 0.6036784, -0.8685901, 0.8268033
1: -0.2096461, 0.3176579, -0.3698139, 0.4686721, -0.6783182, 0.6874719
2: -0.4006885, 0.4239005, -0.4807382, 0.6671062, -1.0677947, 0.9046387
3: -0.1473909, 0.4076343, -0.2618090, 0.6575226, -0.8049135, 0.6694434
4: -0.3350169, 0.2927608, -0.4696916, 0.4659915, -0.8010083, 0.7624524
5: -0.2633038, 0.3608207, -0.4062904, 0.5530006, -0.8163044, 0.7671111
6: -0.2344377, 0.3249663, -0.3988395, 0.4914775, -0.7259152, 0.7238058
7: -0.3162454, 0.2517450, -0.4468763, 0.4420404, -0.7582858, 0.6986213
8: 0.1167590, 2.3620660, -0.2654907, 2.4475613, -2.3308022, 2.6275568
9: -0.5298969, 0.4204434, -0.6030570, 0.6155246, -1.1454215, 1.0235004

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 69

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7821281, upper bound: 10.8122782
time: 3.29 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7723630, upper bound: 10.8101547
time: 3.03 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.2867858, 0.4666795, -0.5651317, 0.7569659, -1.0437517, 1.0318112
1: -0.2355663, 0.3463823, -0.5009040, 0.6210553, -0.8566216, 0.8472863
2: -0.4166623, 0.4709837, -0.5501457, 0.8491168, -1.2657791, 1.0211294
3: -0.1645143, 0.4503374, -0.4035409, 0.8008496, -0.9653639, 0.8538784
4: -0.3621775, 0.3183598, -0.5975495, 0.6262680, -0.9884455, 0.9159093
5: -0.2873838, 0.3944002, -0.5599939, 0.6828460, -0.9702297, 0.9543941
6: -0.2594598, 0.3572052, -0.5655710, 0.6297221, -0.8891819, 0.9227761
7: -0.3414775, 0.2833236, -0.5841417, 0.5936552, -0.9351327, 0.8674653
8: 0.0404112, 2.3981495, -0.5678247, 2.5142083, -2.4737971, 2.9659741
9: -0.5460339, 0.4603432, -0.6835768, 0.7693841, -1.3154180, 1.1439199

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7821281, upper bound: 10.8122782
time: 3.82 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7723630, upper bound: 10.8101547
time: 3.44 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.6069288, 0.7663359, -0.9040302, 0.9931020, -1.6000307, 1.6703660
1: -0.5397391, 0.6713459, -0.7782023, 0.9194229, -1.4591620, 1.4495482
2: -0.5887102, 0.9603049, -0.8238682, 1.2224874, -1.8111976, 1.7841730
3: -0.4343552, 0.8586525, -0.7836573, 1.0871239, -1.5214790, 1.6423098
4: -0.6330412, 0.6548696, -0.9321312, 0.9188656, -1.5519068, 1.5870007
5: -0.5898071, 0.7217666, -0.8405819, 0.9748642, -1.5646713, 1.5623485
6: -0.6240770, 0.6641850, -0.9030452, 0.9158459, -1.5399228, 1.5672302
7: -0.6099432, 0.6163650, -0.8786549, 0.9069485, -1.5168917, 1.4950199
8: -0.6759946, 2.5177431, -1.2295963, 2.5795271, -3.2555218, 3.7473392
9: -0.7058626, 0.8098958, -0.8889506, 1.0680612, -1.7739239, 1.6988463

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8479153, upper bound: 10.8478574
time: 4.72 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8478809, upper bound: 10.8476659
time: 7.49 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.7102745, 0.8588388, -1.3693807, 1.2780809, -1.9883554, 2.2282195
1: -0.6249480, 0.7596509, -1.1127610, 1.2336622, -1.8586103, 1.8724120
2: -0.6565287, 1.0698662, -1.2838553, 1.5205902, -2.1771188, 2.3537216
3: -0.5538708, 0.9378791, -1.3326133, 1.3208702, -1.8747410, 2.2704926
4: -0.7336173, 0.7542968, -1.4738588, 1.2688844, -2.0025017, 2.2281556
5: -0.6815660, 0.8132421, -1.2557907, 1.3004546, -1.9820206, 2.0690327
6: -0.7236909, 0.7508687, -1.3217441, 1.2939874, -2.0176783, 2.0726128
7: -0.7075821, 0.7232465, -1.3109403, 1.3039333, -2.0115154, 2.0341868
8: -0.8817154, 2.5627372, -2.0137072, 2.6647449, -3.5464602, 4.5764446
9: -0.7682095, 0.9066439, -1.2041118, 1.4311137, -2.1993232, 2.1107557

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8479153, upper bound: 10.8478565
time: 3.58 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8478809, upper bound: 10.8476659
time: 3.78 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.3972508, 0.5844560, -0.2718563, 0.4330966, -0.8303474, 0.8563123
1: -0.3648222, 0.4605123, -0.2299533, 0.3378035, -0.7026258, 0.6904656
2: -0.4782107, 0.6256948, -0.3959084, 0.3984242, -0.8766350, 1.0216032
3: -0.2514200, 0.6633652, -0.1535825, 0.4656726, -0.7170926, 0.8169476
4: -0.4604395, 0.4577954, -0.3472705, 0.3111452, -0.7715847, 0.8050659
5: -0.3982381, 0.5425999, -0.2785289, 0.3799600, -0.7781981, 0.8211288
6: -0.3871456, 0.4818622, -0.2483571, 0.3416885, -0.7288340, 0.7302193
7: -0.4388336, 0.4280612, -0.3316835, 0.2651462, -0.7039798, 0.7597448
8: -0.2252535, 2.4780171, 0.1255471, 2.3530605, -2.5783138, 2.3524699
9: -0.6058733, 0.5978824, -0.5278200, 0.4315571, -1.0374305, 1.1257024

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8399027, upper bound: 10.8267103
time: 2.50 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8391758, upper bound: 10.8261110
time: 2.84 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.7200512, 0.8601618, -0.5337362, 0.7031247, -1.4231759, 1.3938980
1: -0.6461574, 0.7765959, -0.4931794, 0.6089277, -1.2550851, 1.2697754
2: -0.6623576, 1.0425322, -0.5439133, 0.8192844, -1.4816420, 1.5864456
3: -0.5764097, 0.9839131, -0.3704591, 0.8380427, -1.4144524, 1.3543723
4: -0.7422813, 0.7762578, -0.5687149, 0.6028460, -1.3451273, 1.3449726
5: -0.6993327, 0.8248894, -0.5343431, 0.6656313, -1.3649640, 1.3592325
6: -0.7369990, 0.7559806, -0.5507957, 0.5999473, -1.3369464, 1.3067763
7: -0.7237718, 0.7384643, -0.5539568, 0.5606580, -1.2844298, 1.2924211
8: -0.8913577, 2.5852151, -0.5123742, 2.4822087, -3.3735664, 3.0975893
9: -0.7778831, 0.9159788, -0.6690869, 0.7413210, -1.5192040, 1.5850656

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8481563, upper bound: 10.8483628
time: 3.29 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8481525, upper bound: 10.8482428
time: 2.43 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.5276226, 0.7174433, -0.2978817, 0.4827912, -1.0104139, 1.0153251
1: -0.4745058, 0.5886714, -0.2593454, 0.3695207, -0.8440265, 0.8480169
2: -0.5371580, 0.7857497, -0.4147611, 0.4475904, -0.9847484, 1.2005107
3: -0.3636447, 0.7881001, -0.1752020, 0.5111876, -0.8748323, 0.9633020
4: -0.5656296, 0.5951887, -0.3764706, 0.3422179, -0.9078475, 0.9716594
5: -0.5329637, 0.6519668, -0.3069928, 0.4196652, -0.9526289, 0.9589596
6: -0.5312247, 0.5997481, -0.2781032, 0.3760106, -0.9072353, 0.8778512
7: -0.5499405, 0.5611771, -0.3585668, 0.3030327, -0.8529732, 0.9197440
8: -0.4861644, 2.5367031, 0.0429671, 2.3903389, -2.8765032, 2.4937360
9: -0.6722655, 0.7323323, -0.5453067, 0.4762058, -1.1484714, 1.2776389

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 92

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8399027, upper bound: 10.8267103
time: 4.24 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8391758, upper bound: 10.8261110
time: 3.84 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -1.0587261, 1.0862486, -0.6287169, 0.7867581, -1.8454841, 1.7149656
1: -0.8896841, 1.0284123, -0.5693458, 0.6925309, -1.5822150, 1.5977582
2: -0.9713391, 1.2908393, -0.5930571, 0.9245808, -1.8959199, 1.8838964
3: -0.9784393, 1.1690248, -0.4714006, 0.9112067, -1.8896459, 1.6404254
4: -1.1044174, 1.0454135, -0.6516184, 0.6920571, -1.7964746, 1.6970319
5: -0.9823597, 1.0791283, -0.6190842, 0.7412251, -1.7235849, 1.6982125
6: -1.0427842, 1.0386906, -0.6393099, 0.6778476, -1.7206318, 1.6780005
7: -1.0225496, 1.0406654, -0.6406441, 0.6464695, -1.6690192, 1.6813095
8: -1.4876153, 2.6547892, -0.6926648, 2.5274520, -4.0150671, 3.3474541
9: -0.9899337, 1.1919188, -0.7182586, 0.8303363, -1.8202701, 1.9101775

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8481562, upper bound: 10.8483628
time: 8.32 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8481525, upper bound: 10.8482428
time: 3.24 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.4112475, 0.5986434, -0.3461276, 0.5403813, -0.9516288, 0.9447711
1: -0.3755398, 0.4763140, -0.3075061, 0.4089915, -0.7845314, 0.7838200
2: -0.4911782, 0.6795380, -0.4436059, 0.5549201, -1.0460982, 1.1231439
3: -0.2665634, 0.6701604, -0.2126576, 0.5667566, -0.8333200, 0.8828179
4: -0.4731467, 0.4711899, -0.4167018, 0.3899208, -0.8630675, 0.8878917
5: -0.4111639, 0.5576638, -0.3463607, 0.4833242, -0.8944880, 0.9040245
6: -0.4049143, 0.4922827, -0.3246465, 0.4278624, -0.8327767, 0.8169292
7: -0.4495321, 0.4444528, -0.3951805, 0.3665243, -0.8160564, 0.8396333
8: -0.2805019, 2.4936230, -0.0998103, 2.4188659, -2.6993678, 2.5934334
9: -0.6150883, 0.6210396, -0.5674323, 0.5380678, -1.1531562, 1.1884718

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8406547, upper bound: 10.8240238
time: 2.78 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8404152, upper bound: 10.8236018
time: 2.21 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.9078974, 0.9899650, -1.0698247, 1.0922663, -2.0001636, 2.0597897
1: -0.7819861, 0.9241357, -0.8952937, 1.0349694, -1.8169556, 1.8194294
2: -0.8286564, 1.2337111, -0.9844245, 1.3239883, -2.1526446, 2.2181356
3: -0.7852799, 1.0921828, -0.9837755, 1.1730731, -1.9583529, 2.0759583
4: -0.9345101, 0.9218605, -1.1191063, 1.0469379, -1.9814481, 2.0409667
5: -0.8432287, 0.9791628, -0.9872468, 1.0902095, -1.9334382, 1.9664096
6: -0.9071294, 0.9183445, -1.0547458, 1.0476823, -1.9548117, 1.9730903
7: -0.8802819, 0.9088610, -1.0278083, 1.0515635, -1.9318454, 1.9366693
8: -1.2398566, 2.6251683, -1.5046304, 2.6071715, -3.8470283, 4.1297989
9: -0.8931963, 1.0725070, -0.9971280, 1.1971482, -2.0903444, 2.0696349

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8483771, upper bound: 10.8485307
time: 8.18 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8483747, upper bound: 10.8483747
time: 5.02 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.5321473, 0.7232394, -0.3896601, 0.5944457, -1.1265930, 1.1128995
1: -0.4765771, 0.5939242, -0.3506900, 0.4443030, -0.9208801, 0.9446142
2: -0.5452241, 0.8186105, -0.4677080, 0.6138409, -1.1590650, 1.2863185
3: -0.3672601, 0.7815726, -0.2426386, 0.6245147, -0.9917748, 1.0242112
4: -0.5679177, 0.5955160, -0.4502370, 0.4468541, -1.0147718, 1.0457530
5: -0.5317445, 0.6574713, -0.3864091, 0.5336899, -1.0654345, 1.0438803
6: -0.5378052, 0.6033511, -0.3691751, 0.4785878, -1.0163929, 0.9725262
7: -0.5516050, 0.5617902, -0.4328258, 0.4211158, -0.9727208, 0.9946160
8: -0.5122484, 2.5520477, -0.2068740, 2.4592540, -2.9715023, 2.7589216
9: -0.6764253, 0.7405280, -0.5951852, 0.5878147, -1.2642400, 1.3357131

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 218

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8406547, upper bound: 10.8240238
time: 2.69 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8404152, upper bound: 10.8236018
time: 2.04 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -1.2821544, 1.2213596, -1.2914523, 1.2298933, -2.5120478, 2.5128119
1: -1.0480664, 1.1757152, -1.0566691, 1.1807933, -2.2288597, 2.2323842
2: -1.1960309, 1.4766555, -1.2018259, 1.4646841, -2.6607151, 2.6784813
3: -1.2240034, 1.2804378, -1.2412879, 1.2854941, -2.5094976, 2.5217257
4: -1.3641855, 1.2048135, -1.3788882, 1.2134643, -2.5776496, 2.5837016
5: -1.1765630, 1.2409748, -1.1871719, 1.2455616, -2.4221246, 2.4281468
6: -1.2474474, 1.2205924, -1.2522717, 1.2292676, -2.4767151, 2.4728642
7: -1.2240160, 1.2251829, -1.2371019, 1.2350286, -2.4590445, 2.4622848
8: -1.8724318, 2.6983604, -1.8832489, 2.6619704, -4.5344019, 4.5816092
9: -1.1442850, 1.3651299, -1.1485474, 1.3713158, -2.5156007, 2.5136774

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 92

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8483771, upper bound: 10.8485307
time: 4.56 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8483747, upper bound: 10.8483747
time: 5.01 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 11.00 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 11.00
Output dim: 8, lower bound: -10.7680343, upper bound: 10.8027016
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 11.00
Output dim: 8, lower bound: -10.7680343, upper bound: 10.8027016
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 11.00
Output dim: 8, lower bound: -10.7676869, upper bound: 10.7995298
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 11.00
Output dim: 8, lower bound: -10.7676869, upper bound: 10.7995298
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 11.00
Output dim: 8, lower bound: -10.8475090, upper bound: 10.8474892
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 11.00
Output dim: 8, lower bound: -10.8474434, upper bound: 10.8473387
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 11.00
Output dim: 8, lower bound: -10.8475090, upper bound: 10.8474891
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 11.00
Output dim: 8, lower bound: -10.8474434, upper bound: 10.8473387
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 11.00
Output dim: 8, lower bound: -10.7821281, upper bound: 10.8122782
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 11.00
Output dim: 8, lower bound: -10.7723630, upper bound: 10.8101547
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 11.00
Output dim: 8, lower bound: -10.7821281, upper bound: 10.8122782
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 11.00
Output dim: 8, lower bound: -10.7723630, upper bound: 10.8101547
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 11.00
Output dim: 8, lower bound: -10.8479153, upper bound: 10.8478574
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 11.00
Output dim: 8, lower bound: -10.8478809, upper bound: 10.8476659
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 11.00
Output dim: 8, lower bound: -10.8479153, upper bound: 10.8478565
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 11.00
Output dim: 8, lower bound: -10.8478809, upper bound: 10.8476659
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 11.00
Output dim: 8, lower bound: -10.8399027, upper bound: 10.8267103
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 11.00
Output dim: 8, lower bound: -10.8391758, upper bound: 10.8261110
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 11.00
Output dim: 8, lower bound: -10.8481563, upper bound: 10.8483628
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 11.00
Output dim: 8, lower bound: -10.8481525, upper bound: 10.8482428
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 11.00
Output dim: 8, lower bound: -10.8399027, upper bound: 10.8267103
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 11.00
Output dim: 8, lower bound: -10.8391758, upper bound: 10.8261110
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 11.00
Output dim: 8, lower bound: -10.8481562, upper bound: 10.8483628
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 11.00
Output dim: 8, lower bound: -10.8481525, upper bound: 10.8482428
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 11.00
Output dim: 8, lower bound: -10.8406547, upper bound: 10.8240238
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 11.00
Output dim: 8, lower bound: -10.8404152, upper bound: 10.8236018
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 11.00
Output dim: 8, lower bound: -10.8483771, upper bound: 10.8485307
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 11.00
Output dim: 8, lower bound: -10.8483747, upper bound: 10.8483747
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 11.00
Output dim: 8, lower bound: -10.8406547, upper bound: 10.8240238
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 11.00
Output dim: 8, lower bound: -10.8404152, upper bound: 10.8236018
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 11.00
Output dim: 8, lower bound: -10.8483771, upper bound: 10.8485307
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 11.00
Output dim: 8, lower bound: -10.8483747, upper bound: 10.8483747

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.2033660, 0.2765330, -0.2797578, 0.4416237, -0.6449897, 0.5562907
1: -0.1266840, 0.2271811, -0.2406676, 0.3486266, -0.4753106, 0.4678487
2: -0.3634102, 0.2460726, -0.4006125, 0.4382688, -0.8016790, 0.6466851
3: -0.1162665, 0.2852808, -0.1602938, 0.4865114, -0.6027779, 0.4455746
4: -0.2495611, 0.2128615, -0.3594965, 0.3189606, -0.5685217, 0.5723580
5: -0.1964559, 0.2531729, -0.2861997, 0.3926084, -0.5890643, 0.5393726
6: -0.1812916, 0.2163949, -0.2581221, 0.3508326, -0.5321242, 0.4745170
7: -0.2305425, 0.1544580, -0.3395531, 0.2771067, -0.5076492, 0.4940111
8: 0.3891913, 2.2702231, 0.0797662, 2.3202581, -1.9310669, 2.1904569
9: -0.4920607, 0.2873344, -0.5274884, 0.4478546, -0.9399153, 0.8148227

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4288885, upper bound: 10.4300143
time: 3.45 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6246693, upper bound: 10.6647633
time: 3.32 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.2426904, 0.3849226, -0.3085995, 0.4922895, -0.7349799, 0.6935221
1: -0.1863980, 0.2940624, -0.2732000, 0.3815025, -0.5679005, 0.5672624
2: -0.3850578, 0.3529163, -0.4207292, 0.4901527, -0.8752105, 0.7736455
3: -0.1343580, 0.3753753, -0.1866446, 0.5336313, -0.6679894, 0.5620199
4: -0.3104978, 0.2717845, -0.3899740, 0.3547178, -0.6652156, 0.6617585
5: -0.2452656, 0.3334138, -0.3158754, 0.4386769, -0.6839425, 0.6492891
6: -0.2135573, 0.2984283, -0.2922179, 0.3862250, -0.5997824, 0.5906461
7: -0.2958883, 0.2255633, -0.3672711, 0.3204888, -0.6163771, 0.5928344
8: 0.2118259, 2.3208795, -0.0048706, 2.3575115, -2.1456857, 2.3257501
9: -0.5162807, 0.3828699, -0.5466158, 0.4935388, -1.0098195, 0.9294857

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4288885, upper bound: 10.4299987
time: 3.16 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6246693, upper bound: 10.6647630
time: 2.45 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.2017640, 0.2367137, -0.6535251, 0.8818232, -1.0835872, 0.8902389
1: -0.1256080, 0.1779312, -0.5594524, 0.7124696, -0.8380776, 0.7373837
2: -0.3322073, 0.2449048, -0.5729677, 0.8807235, -1.2129307, 0.8178725
3: -0.1156317, 0.2676298, -0.4125293, 0.9118733, -1.0275050, 0.6801591
4: -0.1428591, 0.2120259, -0.6310312, 0.7387758, -0.8816349, 0.8430571
5: -0.1958700, 0.1752577, -0.6416861, 0.7743155, -0.9701856, 0.8169438
6: -0.1804353, 0.1675599, -0.6494452, 0.7172532, -0.8976885, 0.8170050
7: -0.2300651, 0.1530630, -0.6693501, 0.6820548, -0.9121200, 0.8224132
8: 0.4296799, 2.2663553, -0.6947992, 2.4008837, -1.9712038, 2.9611545
9: -0.4368219, 0.2862458, -0.7436188, 0.8253105, -1.2621324, 1.0298645

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4264959, upper bound: 10.4258437
time: 3.46 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6241748, upper bound: 10.6613738
time: 3.21 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.2399571, 0.3816969, -0.7125570, 0.9419401, -1.1818972, 1.0942539
1: -0.1829496, 0.2910550, -0.6054774, 0.7735953, -0.9565449, 0.8965324
2: -0.3827922, 0.3475630, -0.6071290, 0.9439496, -1.3267419, 0.9546919
3: -0.1330940, 0.3690439, -0.4503167, 0.9738273, -1.1069213, 0.8193605
4: -0.3080266, 0.2688047, -0.6711507, 0.8025113, -1.1105380, 0.9399554
5: -0.2429604, 0.3304846, -0.6999643, 0.8256104, -1.0685709, 1.0304489
6: -0.2113446, 0.2953901, -0.7121978, 0.7686949, -0.9800395, 1.0075879
7: -0.2934763, 0.2231469, -0.7207243, 0.7373377, -1.0308141, 0.9438711
8: 0.2217759, 2.3130679, -0.8074920, 2.4366939, -2.2149179, 3.1205599
9: -0.5137725, 0.3786606, -0.7912881, 0.8777637, -1.3915362, 1.1699487

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4264959, upper bound: 10.4258437
time: 3.57 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6241748, upper bound: 10.6613738
time: 9.81 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.3811572, 0.5572445, -0.3761618, 0.5506995, -0.9318566, 0.9334063
1: -0.3531910, 0.4649173, -0.3588541, 0.4665316, -0.8197227, 0.8237714
2: -0.4731704, 0.6725160, -0.4713478, 0.6509982, -1.1241686, 1.1438638
3: -0.2526134, 0.6504880, -0.2525032, 0.6745625, -0.9271758, 0.9029913
4: -0.4534556, 0.4302919, -0.4512303, 0.4324256, -0.8858812, 0.8815222
5: -0.3815510, 0.5324404, -0.3812170, 0.5310287, -0.9125797, 0.9136574
6: -0.3809972, 0.4528032, -0.3798858, 0.4495284, -0.8305256, 0.8326890
7: -0.4255363, 0.4080224, -0.4241928, 0.4060304, -0.8315667, 0.8322152
8: -0.2295828, 2.3913360, -0.2205410, 2.3942411, -2.6238239, 2.6118770
9: -0.5886778, 0.5960807, -0.5865706, 0.5939941, -1.1826719, 1.1826513

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5841582, upper bound: 10.5539406
time: 3.38 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8284453, upper bound: 10.8285250
time: 2.82 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.8551544, 1.0976250, -0.3725644, 0.5471701, -1.4023244, 1.4701895
1: -0.7876960, 0.9845357, -0.3543329, 0.4619790, -1.2496750, 1.3388686
2: -0.7282043, 1.2586483, -0.4685503, 0.6456898, -1.3738941, 1.7271986
3: -0.6927937, 1.0890605, -0.2494148, 0.6669967, -1.3597904, 1.3384753
4: -0.9011434, 0.8999733, -0.4482166, 0.4270137, -1.3281572, 1.3481899
5: -0.9077783, 0.9193721, -0.3774304, 0.5264019, -1.4341801, 1.2968025
6: -0.9104009, 0.9664814, -0.3744315, 0.4456678, -1.3560686, 1.3409129
7: -0.8488402, 0.8704160, -0.4207172, 0.4016641, -1.2505043, 1.2911333
8: -1.2242458, 2.4780731, -0.2106174, 2.3872826, -3.6115284, 2.6886907
9: -0.8675770, 1.1101611, -0.5834973, 0.5895355, -1.4571126, 1.6936584

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5800598, upper bound: 10.5523212
time: 2.76 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8198025, upper bound: 10.8265364
time: 2.80 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.4340546, 0.6151971, -0.5164481, 0.6925579, -1.1266125, 1.1316452
1: -0.3999780, 0.5074356, -0.4774567, 0.5917422, -0.9917202, 0.9848922
2: -0.5001793, 0.7343758, -0.5346450, 0.8105382, -1.3107175, 1.2690208
3: -0.2872408, 0.7114563, -0.3509409, 0.8177869, -1.1050277, 1.0623972
4: -0.4912722, 0.4906250, -0.5553344, 0.5826262, -1.0738984, 1.0459594
5: -0.4271175, 0.5842452, -0.5108405, 0.6552299, -1.0823474, 1.0950857
6: -0.4380839, 0.5110152, -0.5325006, 0.5872143, -1.0252981, 1.0435158
7: -0.4686568, 0.4644264, -0.5395662, 0.5466235, -1.0152804, 1.0039926
8: -0.3399096, 2.4283087, -0.4880264, 2.4504020, -2.7903116, 2.9163351
9: -0.6203023, 0.6485343, -0.6612822, 0.7255370, -1.3458393, 1.3098166

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5841582, upper bound: 10.5539406
time: 2.29 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8284453, upper bound: 10.8285083
time: 4.02 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -1.0864739, 1.1697931, -0.5053793, 0.6826478, -1.7691216, 1.6751723
1: -0.9085506, 1.0744708, -0.4670330, 0.5792485, -1.4877992, 1.5415038
2: -0.7978830, 1.4403094, -0.5285088, 0.7960973, -1.5939803, 1.9688182
3: -0.9710221, 1.1982235, -0.3399609, 0.8060150, -1.7770371, 1.5381844
4: -1.0257577, 1.1097565, -0.5452605, 0.5717055, -1.5974631, 1.6550170
5: -0.9972668, 1.0961807, -0.4981013, 0.6465507, -1.6438174, 1.5942819
6: -1.0341730, 1.0371975, -0.5191944, 0.5769078, -1.6110809, 1.5563918
7: -1.0768235, 1.0296872, -0.5305659, 0.5362938, -1.6131172, 1.5602530
8: -1.5347277, 2.5142417, -0.4650133, 2.4399233, -3.9746509, 2.9792550
9: -0.9839303, 1.2125024, -0.6551431, 0.7137631, -1.6976933, 1.8676455

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5800598, upper bound: 10.5523212
time: 3.18 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8198025, upper bound: 10.8265364
time: 3.05 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.2191982, 0.3272352, -0.3328270, 0.5198463, -0.7390445, 0.6600622
1: -0.1529601, 0.2575914, -0.2974525, 0.4053556, -0.5583156, 0.5550438
2: -0.3765755, 0.3299013, -0.4424463, 0.5699944, -0.9465699, 0.7723476
3: -0.1228253, 0.3208438, -0.2096475, 0.5602229, -0.6830481, 0.5304913
4: -0.2792244, 0.2374260, -0.4128310, 0.3786399, -0.6578643, 0.6502571
5: -0.2160510, 0.2892162, -0.3336653, 0.4699596, -0.6860105, 0.6228814
6: -0.1934705, 0.2539471, -0.3199894, 0.4056638, -0.5991343, 0.5739365
7: -0.2602051, 0.1892502, -0.3841140, 0.3516309, -0.6118360, 0.5733642
8: 0.2703637, 2.2880991, -0.0997629, 2.3891468, -2.1187830, 2.3878620
9: -0.5029719, 0.3386327, -0.5628381, 0.5312946, -1.0342665, 0.9014708

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 218

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4333863, upper bound: 10.4340922
time: 3.72 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6377672, upper bound: 10.6733769
time: 3.21 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.3850526, 0.6976170, -0.3276455, 0.5142200, -0.8992726, 1.0252625
1: -0.3492867, 0.4717287, -0.2910871, 0.3989017, -0.7481884, 0.7628158
2: -0.4604677, 0.6742725, -0.4384146, 0.5618904, -1.0223581, 1.1126871
3: -0.2503528, 0.6305873, -0.2047345, 0.5507608, -0.8011136, 0.8353218
4: -0.4901288, 0.4248980, -0.4078655, 0.3722184, -0.8623471, 0.8327634
5: -0.3952941, 0.5552695, -0.3287034, 0.4623462, -0.8576404, 0.8839729
6: -0.3722773, 0.5080099, -0.3137613, 0.4006899, -0.7729672, 0.8217711
7: -0.4635563, 0.4228544, -0.3796591, 0.3443121, -0.8078684, 0.8025135
8: -0.2624631, 2.3639071, -0.0865789, 2.3816459, -2.6441090, 2.4504859
9: -0.5966938, 0.6296165, -0.5588134, 0.5237529, -1.1204467, 1.1884298

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 69

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4276827, upper bound: 10.4314266
time: 3.22 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6272153, upper bound: 10.6705644
time: 3.40 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.2387734, 0.3757393, -0.4352445, 0.6346941, -0.8734676, 0.8109838
1: -0.1782664, 0.2852933, -0.3928065, 0.4916434, -0.6699098, 0.6780998
2: -0.3889953, 0.3754276, -0.4927376, 0.7023590, -1.0913544, 0.8681653
3: -0.1323498, 0.3596866, -0.2821877, 0.6872294, -0.8195792, 0.6418743
4: -0.3044469, 0.2628278, -0.4888281, 0.4987743, -0.8032212, 0.7516558
5: -0.2376286, 0.3235435, -0.4325812, 0.5802755, -0.8179041, 0.7561247
6: -0.2098924, 0.2883176, -0.4271742, 0.5200868, -0.7299792, 0.7154918
7: -0.2870857, 0.2191412, -0.4709671, 0.4738733, -0.7609590, 0.6901082
8: 0.1905596, 2.3244233, -0.3211699, 2.4458709, -2.2553113, 2.6455932
9: -0.5180612, 0.3797634, -0.6195077, 0.6446066, -1.1626678, 0.9992710

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4333863, upper bound: 10.4340922
time: 5.20 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6377672, upper bound: 10.6733769
time: 3.28 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.4149485, 0.7424864, -0.4235338, 0.6244224, -1.0393709, 1.1660202
1: -0.3738865, 0.5126901, -0.3820713, 0.4803626, -0.8542490, 0.8947613
2: -0.4860300, 0.7176869, -0.4864166, 0.6887009, -1.1747309, 1.2041035
3: -0.2736476, 0.6714475, -0.2736817, 0.6723377, -0.9459853, 0.9451292
4: -0.5171357, 0.4639694, -0.4809034, 0.4847648, -1.0019006, 0.9448729
5: -0.4410000, 0.5870545, -0.4210073, 0.5697473, -1.0107473, 1.0080618
6: -0.4016743, 0.5387806, -0.4143048, 0.5097790, -0.9114532, 0.9530854
7: -0.4891208, 0.4668599, -0.4613622, 0.4622939, -0.9514146, 0.9282222
8: -0.3283100, 2.3956449, -0.2973709, 2.4342813, -2.7625914, 2.6930158
9: -0.6201918, 0.6870500, -0.6121814, 0.6330749, -1.2532666, 1.2992314

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4276827, upper bound: 10.4314266
time: 3.27 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6272153, upper bound: 10.6705644
time: 4.05 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.4501177, 0.6195143, -0.6637631, 0.8114857, -1.2616034, 1.2832775
1: -0.4110505, 0.5233831, -0.5948810, 0.7274135, -1.1384640, 1.1182641
2: -0.5071560, 0.7890224, -0.6180362, 1.0253736, -1.5325296, 1.4070585
3: -0.2979048, 0.7154437, -0.4996369, 0.9275970, -1.2255019, 1.2150806
4: -0.5020902, 0.4965073, -0.6856959, 0.7100555, -1.2121457, 1.1822032
5: -0.4336227, 0.5972465, -0.6391574, 0.7724338, -1.2060566, 1.2364038
6: -0.4551720, 0.5212092, -0.6813276, 0.7143449, -1.1695169, 1.2025368
7: -0.4764276, 0.4704258, -0.6632793, 0.6761492, -1.1525768, 1.1337051
8: -0.3867790, 2.4187357, -0.7930233, 2.4966903, -2.8834691, 3.2117591
9: -0.6226631, 0.6616094, -0.7368346, 0.8632129, -1.4858761, 1.3984441

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6046550, upper bound: 10.5758041
time: 6.80 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8446680, upper bound: 10.8446269
time: 14.24 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -1.0930381, 1.1675563, -0.6457763, 0.7964878, -1.8895259, 1.8133326
1: -0.9113925, 1.0814840, -0.5797567, 0.7121065, -1.6234990, 1.6612406
2: -0.8064278, 1.4866335, -0.6035268, 1.0074012, -1.8138291, 2.0901604
3: -0.9694631, 1.1969312, -0.4783041, 0.9123567, -1.8818197, 1.6752354
4: -1.0308630, 1.1073904, -0.6682423, 0.6915953, -1.7224584, 1.7756327
5: -0.9963021, 1.1023357, -0.6210614, 0.7577906, -1.7540927, 1.7233971
6: -1.0435956, 1.0443389, -0.6632846, 0.6995136, -1.7431092, 1.7076235
7: -1.0760051, 1.0285245, -0.6472241, 0.6578823, -1.7338874, 1.6757486
8: -1.5625966, 2.5073948, -0.7590053, 2.4810503, -4.0436468, 3.2664001
9: -0.9869045, 1.2173328, -0.7257925, 0.8455201, -1.8324246, 1.9431252

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6003274, upper bound: 10.5736135
time: 2.59 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8443858, upper bound: 10.8439208
time: 4.50 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.5097917, 0.6842276, -0.9977759, 1.0506004, -1.5603921, 1.6820035
1: -0.4605184, 0.5791093, -0.8423250, 0.9835413, -1.4440597, 1.4214343
2: -0.5394498, 0.8541126, -0.9134276, 1.2886527, -1.8281025, 1.7675402
3: -0.3410235, 0.7785763, -0.8960582, 1.1318076, -1.4728311, 1.6746345
4: -0.5485535, 0.5629795, -1.0386146, 0.9871988, -1.5357523, 1.6015941
5: -0.4888461, 0.6517946, -0.9167470, 1.0429721, -1.5318182, 1.5685415
6: -0.5239922, 0.5843182, -0.9903098, 0.9909143, -1.5149064, 1.5746280
7: -0.5272034, 0.5300444, -0.9620632, 0.9918994, -1.5191028, 1.4921076
8: -0.5035744, 2.4610875, -1.3905070, 2.5729170, -3.0764914, 3.8515944
9: -0.6602612, 0.7186484, -0.9525754, 1.1393964, -1.7996576, 1.6712239

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6046550, upper bound: 10.5758041
time: 2.41 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8446680, upper bound: 10.8446250
time: 2.46 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -1.2729815, 1.3444593, -0.9547482, 1.0236046, -2.2965860, 2.2992074
1: -1.0821991, 1.2224874, -0.8113899, 0.9539940, -2.0361931, 2.0338774
2: -1.0593271, 1.6299157, -0.8682653, 1.2599925, -2.3193197, 2.4981809
3: -1.2059677, 1.3586601, -0.8451068, 1.1072452, -2.3132129, 2.2037668
4: -1.3113695, 1.2676296, -0.9902481, 0.9521925, -2.2635622, 2.2578778
5: -1.1502364, 1.3343354, -0.8778787, 1.0121236, -2.1623600, 2.2122140
6: -1.2568996, 1.1988200, -0.9479994, 0.9576210, -2.2145205, 2.1468194
7: -1.2393744, 1.3553423, -0.9230896, 0.9549605, -2.1943350, 2.2784319
8: -1.9967144, 2.5519094, -1.3177330, 2.5520122, -4.5487266, 3.8696425
9: -1.1692362, 1.3893396, -0.9243120, 1.1045671, -2.2738032, 2.3136516

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6003274, upper bound: 10.5736135
time: 2.80 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8443858, upper bound: 10.8439208
time: 2.52 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.3079293, 0.4802124, -0.2352958, 0.3638943, -0.6718236, 0.7155082
1: -0.2741423, 0.3796028, -0.1862079, 0.2902398, -0.5643821, 0.5658107
2: -0.4331419, 0.5038130, -0.3758747, 0.3227167, -0.7558585, 0.8796878
3: -0.1856593, 0.5438061, -0.1294704, 0.3929948, -0.5786541, 0.6732765
4: -0.3881966, 0.3536496, -0.3014644, 0.2693124, -0.6575090, 0.6551141
5: -0.3120538, 0.4337967, -0.2400863, 0.3247645, -0.6368183, 0.6738830
6: -0.2939611, 0.3809230, -0.2079564, 0.2888157, -0.5827768, 0.5888794
7: -0.3632280, 0.3134341, -0.2899989, 0.2158175, -0.5790455, 0.6034330
8: -0.0217074, 2.4030919, 0.2437614, 2.2973323, -2.3190398, 2.1593304
9: -0.5613488, 0.4911945, -0.5068148, 0.3699413, -0.9312900, 0.9980092

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5907780, upper bound: 10.5783510
time: 3.31 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5761037, upper bound: 10.5456772
time: 2.78 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.6863332, 0.9014071, -0.2350072, 0.3650030, -1.0513362, 1.1364143
1: -0.6001660, 0.6697282, -0.1857990, 0.2903402, -0.8905061, 0.8555273
2: -0.6086285, 0.9377967, -0.3750223, 0.3224281, -0.9310566, 1.3128190
3: -0.4494186, 0.9584821, -0.1294853, 0.3910071, -0.8404257, 1.0879674
4: -0.6472908, 0.7855884, -0.3019102, 0.2692402, -0.9165310, 1.0874987
5: -0.6735144, 0.8044729, -0.2403736, 0.3254495, -0.9989638, 1.0448465
6: -0.6820218, 0.7527453, -0.2079692, 0.2893975, -0.9714193, 0.9607145
7: -0.6615828, 0.7254847, -0.2904902, 0.2165192, -0.8781020, 1.0159749
8: -0.7718478, 2.4739065, 0.2459969, 2.2952337, -3.0670815, 2.2279096
9: -0.7555685, 0.8791144, -0.5058018, 0.3699008, -1.1254693, 1.3849163

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5826125, upper bound: 10.5745732
time: 3.91 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5719347, upper bound: 10.5447682
time: 3.72 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.5037924, 0.6713780, -0.4256729, 0.5991240, -1.1029165, 1.0970509
1: -0.4672565, 0.5796757, -0.4020746, 0.5054963, -0.9727528, 0.9817502
2: -0.5381757, 0.8093951, -0.4923339, 0.7033521, -1.2415278, 1.3017290
3: -0.3374651, 0.8129521, -0.2835439, 0.7359684, -1.0734334, 1.0964961
4: -0.5435786, 0.5682331, -0.4854532, 0.4914358, -1.0350144, 1.0536864
5: -0.4917487, 0.6480746, -0.4248536, 0.5797230, -1.0714717, 1.0729282
6: -0.5218014, 0.5742900, -0.4313195, 0.5022602, -1.0240616, 1.0056095
7: -0.5262837, 0.5298987, -0.4654033, 0.4612500, -0.9875337, 0.9953020
8: -0.4759055, 2.4861214, -0.3140081, 2.4131131, -2.8890185, 2.8001294
9: -0.6628054, 0.7126737, -0.6124048, 0.6405033, -1.3033087, 1.3250785

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6322164, upper bound: 10.5909573
time: 4.38 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8453137, upper bound: 10.8454833
time: 3.74 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -1.2569762, 1.3613125, -0.4226635, 0.5966700, -1.8536463, 1.7839760
1: -1.1067213, 1.2548268, -0.3985967, 0.5022240, -1.6089453, 1.6534235
2: -1.0447080, 1.5954704, -0.4902732, 0.6993466, -1.7440546, 2.0857437
3: -1.2060573, 1.3890160, -0.2811623, 0.7306135, -1.9366708, 1.6701783
4: -1.3031793, 1.2736546, -0.4831688, 0.4879421, -1.7911214, 1.7568233
5: -1.1555589, 1.3335457, -0.4223174, 0.5765736, -1.7321326, 1.7558631
6: -1.2477385, 1.2451636, -0.4274991, 0.4991914, -1.7469299, 1.6726627
7: -1.2445036, 1.3494430, -0.4630195, 0.4583923, -1.7028959, 1.8124624
8: -1.9806989, 2.5762944, -0.3067691, 2.4084129, -4.3891120, 2.8830636
9: -1.1640425, 1.4113437, -0.6103013, 0.6372303, -1.8012727, 2.0216451

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6308177, upper bound: 10.5902338
time: 10.80 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8452553, upper bound: 10.8453447
time: 1.84 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.3896865, 0.5788202, -0.2577911, 0.4150825, -0.8047690, 0.8366113
1: -0.3555349, 0.4527697, -0.2149710, 0.3209793, -0.6765142, 0.6677407
2: -0.4735010, 0.6222408, -0.3902908, 0.3744236, -0.8479246, 1.0125316
3: -0.2456513, 0.6468107, -0.1442693, 0.4405481, -0.6861994, 0.7910800
4: -0.4546746, 0.4464933, -0.3312183, 0.2962665, -0.7509411, 0.7777116
5: -0.3895405, 0.5356989, -0.2648595, 0.3620292, -0.7515697, 0.8005584
6: -0.3764507, 0.4755938, -0.2342568, 0.3244692, -0.7009200, 0.7098505
7: -0.4337000, 0.4218122, -0.3178812, 0.2492397, -0.6829396, 0.7396934
8: -0.2097569, 2.4511836, 0.1576173, 2.3343396, -2.5440965, 2.2935662
9: -0.6024609, 0.5909619, -0.5231971, 0.4129972, -1.0154581, 1.1141589

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4931848, upper bound: 10.4453976
time: 2.94 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7205977, upper bound: 10.6933194
time: 26.34 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.9806349, 1.1106157, -0.2557656, 0.4125015, -1.3931365, 1.3663814
1: -0.8200831, 0.9865527, -0.2121467, 0.3184518, -1.1385349, 1.1986994
2: -0.7335892, 1.2945457, -0.3882407, 0.3697509, -1.1033400, 1.6827863
3: -0.8821957, 1.1139307, -0.1432071, 0.4345644, -1.3167601, 1.2571378
4: -0.9433004, 1.0326934, -0.3291087, 0.2939817, -1.2372822, 1.3618021
5: -0.9249151, 1.0191169, -0.2630648, 0.3596698, -1.2845849, 1.2821817
6: -0.9406049, 0.9289368, -0.2320760, 0.3221356, -1.2627406, 1.1610128
7: -0.9969211, 0.9642166, -0.3160470, 0.2471990, -1.2441200, 1.2802637
8: -1.3517857, 2.5292428, 0.1662491, 2.3279996, -3.6797853, 2.3629937
9: -0.9233739, 1.1354082, -0.5209664, 0.4096473, -1.3330213, 1.6563746

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4920902, upper bound: 10.4448734
time: 3.17 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7166724, upper bound: 10.6926329
time: 7.29 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.6860019, 0.8334757, -0.4879348, 0.6626357, -1.3486376, 1.3214105
1: -0.6168090, 0.7486603, -0.4536187, 0.5610396, -1.1778486, 1.2022790
2: -0.6343580, 1.0203208, -0.5220760, 0.7677305, -1.4020885, 1.5423968
3: -0.5348656, 0.9529571, -0.3246880, 0.7984979, -1.3333635, 1.2776451
4: -0.7101170, 0.7400178, -0.5287728, 0.5576718, -1.2677888, 1.2687906
5: -0.6631879, 0.7962027, -0.4801771, 0.6332964, -1.2964842, 1.2763798
6: -0.7020803, 0.7311605, -0.4985113, 0.5607910, -1.2628713, 1.2296717
7: -0.6919082, 0.7073001, -0.5160767, 0.5189652, -1.2108735, 1.2233768
8: -0.8336006, 2.5442188, -0.4297178, 2.4516456, -3.2852464, 2.9739366
9: -0.7582317, 0.8848379, -0.6485712, 0.6958364, -1.4540682, 1.5334091

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6322164, upper bound: 10.5909573
time: 3.08 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8453137, upper bound: 10.8454833
time: 2.13 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -1.9648912, 1.6727191, -0.4788750, 0.6543547, -2.6192460, 2.1515942
1: -1.5396888, 1.6665983, -0.4448566, 0.5515517, -2.0912404, 2.1114550
2: -1.8644098, 1.8976407, -0.5170678, 0.7576407, -2.6220505, 2.4147086
3: -2.0587592, 1.6298723, -0.3181404, 0.7876403, -2.8463995, 1.9480128
4: -2.2076688, 1.7249840, -0.5221716, 0.5480380, -2.7557068, 2.2471557
5: -1.8200293, 1.7228205, -0.4717896, 0.6253623, -2.4453917, 2.1946101
6: -1.8553421, 1.8065562, -0.4876192, 0.5530998, -2.4084420, 2.2941754
7: -1.8680689, 1.8428371, -0.5085671, 0.5109141, -2.3789830, 2.3514042
8: -3.0759318, 2.6560991, -0.4121733, 2.4430847, -5.5190163, 3.0682724
9: -1.6177682, 1.9128779, -0.6431735, 0.6873887, -2.3051567, 2.5560515

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6308177, upper bound: 10.5902338
time: 2.64 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8452553, upper bound: 10.8453447
time: 2.85 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.3200305, 0.4923196, -0.2909081, 0.4689804, -0.7890109, 0.7832277
1: -0.2840987, 0.3933890, -0.2495183, 0.3577935, -0.6418923, 0.6429073
2: -0.4448241, 0.5574368, -0.4135142, 0.4736918, -0.9185159, 0.9709510
3: -0.1984450, 0.5486759, -0.1705202, 0.4888404, -0.6872854, 0.7191961
4: -0.4002783, 0.3633492, -0.3691683, 0.3304559, -0.7307342, 0.7325175
5: -0.3210293, 0.4482880, -0.2953090, 0.4081187, -0.7291480, 0.7435970
6: -0.3061389, 0.3897881, -0.2698089, 0.3644720, -0.6706110, 0.6595969
7: -0.3722514, 0.3290889, -0.3490945, 0.2927328, -0.6649842, 0.6781834
8: -0.0757026, 2.4166617, 0.0291025, 2.3607450, -2.4364476, 2.3875592
9: -0.5683860, 0.5132188, -0.5400597, 0.4691775, -1.0375636, 1.0532784

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 218

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4950433, upper bound: 10.4407654
time: 3.69 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7240061, upper bound: 10.6854648
time: 4.15 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.6929268, 0.9078397, -0.2886491, 0.4663060, -1.1592327, 1.1964887
1: -0.6025909, 0.6803139, -0.2464186, 0.3550883, -0.9576792, 0.9267325
2: -0.6151102, 0.9808078, -0.4113944, 0.4691451, -1.0842552, 1.3922021
3: -0.4572954, 0.9544945, -0.1684535, 0.4829727, -0.9402681, 1.1229481
4: -0.6524405, 0.7880850, -0.3669645, 0.3272536, -0.9796941, 1.1550496
5: -0.6764324, 0.8130766, -0.2931730, 0.4046852, -1.0811176, 1.1062496
6: -0.6928050, 0.7541282, -0.2667459, 0.3620614, -1.0548663, 1.0208740
7: -0.6663863, 0.7316203, -0.3471546, 0.2899473, -0.9563336, 1.0787749
8: -0.8068883, 2.4869175, 0.0367427, 2.3564181, -3.1633065, 2.4501748
9: -0.7607912, 0.8899285, -0.5379496, 0.4658235, -1.2266147, 1.4278781

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4937695, upper bound: 10.4401019
time: 3.88 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7195050, upper bound: 10.6847374
time: 2.62 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.6117356, 0.7630829, -0.7537737, 0.8863891, -1.4981247, 1.5168566
1: -0.5525742, 0.6844066, -0.6692640, 0.8062903, -1.3588645, 1.3536706
2: -0.5917937, 0.9773406, -0.6823413, 1.1101296, -1.7019233, 1.6596820
3: -0.4366021, 0.8898871, -0.6085942, 0.9999913, -1.4365934, 1.4984813
4: -0.6377454, 0.6581568, -0.7790619, 0.7972442, -1.4349897, 1.4372187
5: -0.5911746, 0.7317085, -0.7171484, 0.8584596, -1.4496342, 1.4488568
6: -0.6336744, 0.6724296, -0.7673926, 0.7897186, -1.4233930, 1.4398222
7: -0.6141174, 0.6229627, -0.7519842, 0.7754075, -1.3895249, 1.3749470
8: -0.6995431, 2.5185962, -0.9705565, 2.5192831, -3.2188263, 3.4891527
9: -0.7122207, 0.8166543, -0.7964343, 0.9437572, -1.6559780, 1.6130886

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6506840, upper bound: 10.6087063
time: 3.52 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8461828, upper bound: 10.8463151
time: 4.17 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -1.8223336, 1.5836242, -0.7354426, 0.8718119, -2.6941454, 2.3190668
1: -1.4241189, 1.5599555, -0.6546274, 0.7909701, -2.2150891, 2.2145829
2: -1.7142729, 1.8376014, -0.6651328, 1.0935473, -2.8078203, 2.5027342
3: -1.8447185, 1.5469937, -0.5867846, 0.9848100, -2.8295283, 2.1337783
4: -1.9324362, 1.5982400, -0.7596824, 0.7792674, -2.7117038, 2.3579226
5: -1.5503788, 1.6369989, -0.7003757, 0.8412968, -2.3916755, 2.3373747
6: -1.7741828, 1.6474332, -0.7495822, 0.7739285, -2.5481114, 2.3970153
7: -1.6658391, 1.7433083, -0.7343404, 0.7574931, -2.4233322, 2.4776487
8: -2.7325888, 2.6327415, -0.9357544, 2.5051692, -5.2377577, 3.5684958
9: -1.4565963, 1.7686013, -0.7835243, 0.9268624, -2.3834586, 2.5521255

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6485238, upper bound: 10.6080994
time: 2.56 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8461391, upper bound: 10.8461391
time: 2.29 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 6.37 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.4288885, upper bound: 10.4300143
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.6246693, upper bound: 10.6647633
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.4288885, upper bound: 10.4299987
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.6246693, upper bound: 10.6647630
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.4264959, upper bound: 10.4258437
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.6241748, upper bound: 10.6613738
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.4264959, upper bound: 10.4258437
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.6241748, upper bound: 10.6613738
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.5841582, upper bound: 10.5539406
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.8284453, upper bound: 10.8285250
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.5800598, upper bound: 10.5523212
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.8198025, upper bound: 10.8265364
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.5841582, upper bound: 10.5539406
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.8284453, upper bound: 10.8285083
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.5800598, upper bound: 10.5523212
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.8198025, upper bound: 10.8265364
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.4333863, upper bound: 10.4340922
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.6377672, upper bound: 10.6733769
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.4276827, upper bound: 10.4314266
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.6272153, upper bound: 10.6705644
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.4333863, upper bound: 10.4340922
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.6377672, upper bound: 10.6733769
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.4276827, upper bound: 10.4314266
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.6272153, upper bound: 10.6705644
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.6046550, upper bound: 10.5758041
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.8446680, upper bound: 10.8446269
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.6003274, upper bound: 10.5736135
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.8443858, upper bound: 10.8439208
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.6046550, upper bound: 10.5758041
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.8446680, upper bound: 10.8446250
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.6003274, upper bound: 10.5736135
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.8443858, upper bound: 10.8439208
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.5907780, upper bound: 10.5783510
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.5761037, upper bound: 10.5456772
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.5826125, upper bound: 10.5745732
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.5719347, upper bound: 10.5447682
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.6322164, upper bound: 10.5909573
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.8453137, upper bound: 10.8454833
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.6308177, upper bound: 10.5902338
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.8452553, upper bound: 10.8453447
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.4931848, upper bound: 10.4453976
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.7205977, upper bound: 10.6933194
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.4920902, upper bound: 10.4448734
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.7166724, upper bound: 10.6926329
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.6322164, upper bound: 10.5909573
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.8453137, upper bound: 10.8454833
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.6308177, upper bound: 10.5902338
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.8452553, upper bound: 10.8453447
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.4950433, upper bound: 10.4407654
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.7240061, upper bound: 10.6854648
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.4937695, upper bound: 10.4401019
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.7195050, upper bound: 10.6847374
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.6506840, upper bound: 10.6087063
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.8461828, upper bound: 10.8463151
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.6485238, upper bound: 10.6080994
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.37
Output dim: 8, lower bound: -10.8461391, upper bound: 10.8461391
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.37
Output dim: 8, lower bound: -10.8406547, upper bound: 10.8240238
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.37
Output dim: 8, lower bound: -10.8404152, upper bound: 10.8236018
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.37
Output dim: 8, lower bound: -10.8483771, upper bound: 10.8485307
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.37
Output dim: 8, lower bound: -10.8483747, upper bound: 10.8483747
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=12.009641647338867
rel_dist={8: [-10.853314892972527, 10.853314942404314]}

## Binary Search with IS_dual_ind Result
status: None
Maximum delta epsilon: None
execution time: 1805.38 seconds
