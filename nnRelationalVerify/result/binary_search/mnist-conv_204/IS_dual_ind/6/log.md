## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 1.49125804913
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2435699, 3.2435699)
1: (-13.8976402, -9.6007652, -13.8976402, -9.6007652, -4.2968750, 4.2968750)
2: (-7.1633539, -3.7632816, -7.1633539, -3.7632816, -3.4000723, 3.4000723)
3: (-12.8481083, -9.6415062, -12.8481083, -9.6415062, -3.2066021, 3.2066021)
4: (-6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.5599658, 3.5599658)
5: (-2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.9095392, 2.9095392)
6: (8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.4539471, 3.4539471)
7: (-18.6063366, -15.0106993, -18.6063366, -15.0106993, -3.5956373, 3.5956373)
8: (-1.4227927, 1.5777073, -1.4227927, 1.5777073, -3.0005000, 3.0005000)
9: (-16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.6913013, 3.6913013)

## BASE Result
execution time: IAR + LP analysis = 15.16 + 32.63 = 47.80 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3552.20 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=3.453947067260742
rel_dist={6: [-1.951663254659758, 1.9516631580018462]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=3.1755638122558594
rel_dist={6: [-1.493016446904722, 1.493015957466044]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=2.9873008728027344
rel_dist={6: [-1.11031708800226, 1.1103168328917867]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=3.081432342529297
rel_dist={6: [-1.3094574363629174, 1.309455504183358]}

## Binary Search Result
Binary search time: 211.13 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Individual Split (IS_dual_ind) starts
Time budget: 3341.08 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 5717
type: A, layer: 1, pos: 6199
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 110

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 515

## Relational analysis of IS_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5717

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0967211, upper bound: 2.0837102
time: 9.51 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0967558, upper bound: 2.0967577
time: 6.02 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 25.81 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 25.81
Output dim: 6, lower bound: -2.0967211, upper bound: 2.0837102
IS_A2, status: Status.UNKNOWN, split count: 1, time: 25.81
Output dim: 6, lower bound: -2.0967558, upper bound: 2.0967577

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -7.2004657, -3.9949496, -7.2200589, -3.9836211, -3.2168446, 3.2251093
1: -13.8742924, -9.6101704, -13.8917580, -9.6015539, -4.2331152, 4.2380095
2: -7.1523190, -3.7931824, -7.1629109, -3.7715960, -3.1645765, 3.1551046
3: -12.8441906, -9.6504488, -12.8471737, -9.6434221, -2.9591360, 2.9545307
4: -6.9580803, -3.4165711, -6.9660921, -3.4101610, -3.3638964, 3.3660054
5: -2.8751955, 0.0102386, -2.8861535, 0.0184953, -2.8936908, 2.8963921
6: 8.6851187, 12.0907812, 8.6598549, 12.1053238, -3.4202051, 3.4309263
7: -18.5978069, -15.0268421, -18.6055164, -15.0150661, -3.1692095, 3.1672277
8: -1.3889344, 1.5655031, -1.4139769, 1.5773182, -2.9182405, 2.9283972
9: -16.1282272, -12.4528694, -16.1369438, -12.4491673, -3.6101866, 3.6128063

Time for backsubstitution: 14.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 6199
type: B, layer: 1, pos: 5717
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 6185
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 110

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 515

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0829285, upper bound: 2.0836094
time: 7.67 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0967027, upper bound: 2.0836906
time: 7.18 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -7.2269263, -3.9833636, -7.2269320, -3.9833636, -3.2435627, 3.2435684
1: -13.8976326, -9.6007652, -13.8976402, -9.6007633, -4.2476406, 4.2545223
2: -7.1633534, -3.7632983, -7.1633549, -3.7632828, -3.1836476, 3.1652622
3: -12.8481054, -9.6415062, -12.8481092, -9.6415062, -2.9649329, 2.9656115
4: -6.9691410, -3.4091768, -6.9691439, -3.4091768, -3.3746643, 3.3768878
5: -2.8902140, 0.0193191, -2.8902192, 0.0193193, -2.9095333, 2.9095383
6: 8.6571102, 12.1110420, 8.6571064, 12.1110516, -3.4539413, 3.4539356
7: -18.6063347, -15.0107040, -18.6063366, -15.0106983, -3.1821499, 3.1778588
8: -1.4227853, 1.5777087, -1.4227917, 1.5777078, -2.9318924, 2.9497261
9: -16.1397781, -12.4484835, -16.1397820, -12.4484806, -3.6226768, 3.6283236

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 5717
type: B, layer: 1, pos: 6199
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 6185
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 110

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 515

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0829656, upper bound: 2.0967307
time: 10.05 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0967374, upper bound: 2.0967375
time: 6.70 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 31.32 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 31.32
Output dim: 6, lower bound: -2.0829285, upper bound: 2.0836094
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 31.32
Output dim: 6, lower bound: -2.0967027, upper bound: 2.0836906
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 31.32
Output dim: 6, lower bound: -2.0829656, upper bound: 2.0967307
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 31.32
Output dim: 6, lower bound: -2.0967374, upper bound: 2.0967375

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -7.1961555, -3.9974475, -7.1978612, -3.9971828, -3.1989727, 3.2004137
1: -13.8690424, -9.6108265, -13.8697767, -9.6084824, -4.2072840, 4.2086496
2: -7.1492920, -3.7965381, -7.1460304, -3.7880857, -3.1400928, 3.1312914
3: -12.8406734, -9.6544199, -12.8312016, -9.6659698, -2.9271970, 2.9278889
4: -6.9525919, -3.4185412, -6.9481101, -3.4262638, -3.3342009, 3.3468251
5: -2.8722868, 0.0045521, -2.8580980, -0.0022659, -2.8700209, 2.8626502
6: 8.6884193, 12.0826168, 8.6976089, 12.0773096, -3.3888903, 3.3850079
7: -18.5966988, -15.0304918, -18.5944405, -15.0304585, -3.1482668, 3.1500616
8: -1.3805168, 1.5640950, -1.3783441, 1.5630426, -2.8940763, 2.8884392
9: -16.1180534, -12.4539967, -16.0971107, -12.4752731, -3.5729733, 3.5700922

Time for backsubstitution: 14.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6199
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 110

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6199

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0820651, upper bound: 2.0719063
time: 5.63 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0829222, upper bound: 2.0836038
time: 14.27 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -7.2004638, -3.9949493, -7.2200499, -3.9836237, -3.2168400, 3.2251005
1: -13.8742876, -9.6101704, -13.8917475, -9.6015577, -4.2140102, 4.2571907
2: -7.1523199, -3.7931836, -7.1629057, -3.7716041, -3.1553249, 3.1746445
3: -12.8441887, -9.6504488, -12.8471642, -9.6434288, -2.9780025, 2.9461360
4: -6.9580784, -3.4165721, -6.9660845, -3.4101641, -3.3497038, 3.3803740
5: -2.8751955, 0.0102363, -2.8861485, 0.0184851, -2.8936806, 2.8963847
6: 8.6851196, 12.0907784, 8.6598616, 12.1053104, -3.4201908, 3.4309168
7: -18.5978069, -15.0268421, -18.6055183, -15.0150738, -3.1678600, 3.1672235
8: -1.3889315, 1.5655031, -1.4139555, 1.5773144, -2.9182348, 2.9216866
9: -16.1282253, -12.4528694, -16.1369286, -12.4491692, -3.6101818, 3.5812416

Time for backsubstitution: 14.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 6199
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 110

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 515

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0966961, upper bound: 2.0697623
time: 6.54 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0966960, upper bound: 2.0697641
time: 5.97 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -7.2225809, -3.9858627, -7.2047048, -3.9969301, -3.2256508, 3.2188420
1: -13.8923950, -9.6014299, -13.8756723, -9.6076946, -4.2217894, 4.2251415
2: -7.1603255, -3.7666333, -7.1464748, -3.7797542, -3.1591477, 3.1414356
3: -12.8446007, -9.6455021, -12.8321457, -9.6640778, -2.9329648, 2.9389453
4: -6.9636526, -3.4111655, -6.9511623, -3.4252965, -3.3449583, 3.3576880
5: -2.8872848, 0.0136454, -2.8621078, -0.0014493, -2.8858354, 2.8757532
6: 8.6603832, 12.1028814, 8.6948261, 12.0830364, -3.4226532, 3.4080553
7: -18.6052322, -15.0143337, -18.5952663, -15.0260725, -3.1612225, 3.1606917
8: -1.4143810, 1.5762935, -1.3871708, 1.5634294, -2.9077544, 2.9097939
9: -16.1296215, -12.4496078, -16.0999680, -12.4745846, -3.5854816, 3.5856323

Time for backsubstitution: 14.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6199
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 110

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6199

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0821060, upper bound: 2.0850995
time: 6.34 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0829593, upper bound: 2.0850960
time: 17.91 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -7.2269235, -3.9833636, -7.2269230, -3.9833670, -3.2435565, 3.2435594
1: -13.8976307, -9.6007633, -13.8976297, -9.6007643, -4.2285385, 4.2736855
2: -7.1633554, -3.7633009, -7.1633506, -3.7632923, -3.1743608, 3.1846962
3: -12.8481045, -9.6415081, -12.8480968, -9.6415119, -2.9837656, 2.9572034
4: -6.9691391, -3.4091773, -6.9691343, -3.4091799, -3.3604898, 3.3911834
5: -2.8902130, 0.0193167, -2.8902135, 0.0193095, -2.9095225, 2.9095302
6: 8.6571121, 12.1110401, 8.6571140, 12.1110363, -3.4539242, 3.4539261
7: -18.6063347, -15.0107031, -18.6063347, -15.0107040, -3.1808004, 3.1778545
8: -1.4227819, 1.5777073, -1.4227729, 1.5777063, -2.9318857, 2.9430170
9: -16.1397762, -12.4484825, -16.1397629, -12.4484806, -3.6226692, 3.5967579

Time for backsubstitution: 15.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 6199
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 110

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 515

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0967311, upper bound: 2.0829642
time: 7.26 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0967310, upper bound: 2.0829672
time: 5.72 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 28.24 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 28.24
Output dim: 6, lower bound: -2.0820651, upper bound: 2.0719063
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 28.24
Output dim: 6, lower bound: -2.0829222, upper bound: 2.0836038
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 28.24
Output dim: 6, lower bound: -2.0966961, upper bound: 2.0697623
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 28.24
Output dim: 6, lower bound: -2.0966960, upper bound: 2.0697641
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 28.24
Output dim: 6, lower bound: -2.0821060, upper bound: 2.0850995
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 28.24
Output dim: 6, lower bound: -2.0829593, upper bound: 2.0850960
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 28.24
Output dim: 6, lower bound: -2.0967311, upper bound: 2.0829642
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 28.24
Output dim: 6, lower bound: -2.0967310, upper bound: 2.0829672

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -7.1698103, -4.0097022, -7.1921082, -3.9985144, -3.1712959, 3.1824059
1: -13.8423929, -9.6361256, -13.8682747, -9.6156750, -4.1734848, 4.0996237
2: -7.1367316, -3.8077552, -7.1439257, -3.7898839, -3.1261482, 3.0209856
3: -12.8189068, -9.6837502, -12.8288918, -9.6721058, -2.8409128, 2.8980498
4: -6.9457083, -3.4329188, -6.9471960, -3.4295275, -3.3185892, 3.3149028
5: -2.8412778, -0.0190537, -2.8545985, -0.0083845, -2.8328934, 2.8355448
6: 8.7167692, 12.0594921, 8.6999111, 12.0712757, -3.3545065, 3.3595810
7: -18.5858459, -15.0514030, -18.5936012, -15.0357857, -3.1392584, 3.1262770
8: -1.3459377, 1.5527754, -1.3702829, 1.5620737, -2.8580837, 2.8912358
9: -16.0781288, -12.4786158, -16.0866051, -12.4760494, -3.5304813, 3.5330391

Time for backsubstitution: 14.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5717
type: B, layer: 1, pos: 6199
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 6185
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 110

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5717

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0688965, upper bound: 2.0719064
time: 5.06 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0688966, upper bound: 2.0719082
time: 6.98 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -7.1961451, -3.9974465, -7.1978598, -3.9971814, -3.1989636, 3.2004132
1: -13.8690414, -9.6108456, -13.8697739, -9.6084852, -4.2112598, 4.2082214
2: -7.1492834, -3.7965415, -7.1460285, -3.7880878, -3.1497221, 3.1194830
3: -12.8406715, -9.6544313, -12.8311996, -9.6659718, -2.9221869, 2.9114599
4: -6.9525909, -3.4185476, -6.9481125, -3.4262648, -3.3469305, 3.3411016
5: -2.8722808, 0.0045476, -2.8580976, -0.0022688, -2.8700120, 2.8626451
6: 8.6884193, 12.0826073, 8.6976070, 12.0773058, -3.3888865, 3.3850002
7: -18.5966949, -15.0305023, -18.5944405, -15.0304594, -3.1483922, 3.1500406
8: -1.3805051, 1.5640931, -1.3783388, 1.5630426, -2.8908734, 2.8884363
9: -16.1180420, -12.4539986, -16.0971069, -12.4752750, -3.5403709, 3.5700865

Time for backsubstitution: 14.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6199
type: B, layer: 1, pos: 5717
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 6185
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 110

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 6199

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0711981, upper bound: 2.0827762
time: 5.60 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0711981, upper bound: 2.0836054
time: 6.10 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -7.1783314, -4.0085049, -7.2200499, -3.9836237, -3.1947076, 3.2115450
1: -13.8522911, -9.6170874, -13.8917475, -9.6015577, -4.1890602, 4.2056847
2: -7.1354346, -3.8096886, -7.1629057, -3.7716041, -3.1355057, 3.1201105
3: -12.8281994, -9.6729450, -12.8471642, -9.6434288, -2.9238315, 2.9191413
4: -6.9400978, -3.4326439, -6.9660845, -3.4101641, -3.3342485, 3.3356190
5: -2.8472624, -0.0105217, -2.8861485, 0.0184851, -2.8657475, 2.8756268
6: 8.7228813, 12.0627613, 8.6598616, 12.1053104, -3.3824291, 3.4028997
7: -18.5867138, -15.0422554, -18.6055183, -15.0150738, -3.1569524, 3.1476998
8: -1.3533051, 1.5512390, -1.4139555, 1.5773144, -2.8797150, 2.9138637
9: -16.0883598, -12.4789734, -16.1369286, -12.4491692, -3.5687456, 3.5861330

Time for backsubstitution: 14.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6199
type: B, layer: 1, pos: 5717
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 6185
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 110

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6199

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0712038, upper bound: 2.0688837
time: 7.92 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0829217, upper bound: 2.0697565
time: 9.52 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -7.2004547, -3.9949539, -7.2200499, -3.9836237, -3.2168310, 3.2250960
1: -13.8742809, -9.6101694, -13.8917475, -9.6015577, -4.2522717, 4.2571793
2: -7.1523161, -3.7931890, -7.1629057, -3.7716041, -3.1840286, 3.1746287
3: -12.8441830, -9.6504555, -12.8471642, -9.6434288, -2.9779816, 2.9733577
4: -6.9580717, -3.4165778, -6.9660845, -3.4101641, -3.3782034, 3.3803596
5: -2.8751903, 0.0102272, -2.8861485, 0.0184851, -2.8936753, 2.8963757
6: 8.6851244, 12.0907650, 8.6598616, 12.1053104, -3.4201860, 3.4309034
7: -18.5978031, -15.0268478, -18.6055183, -15.0150738, -3.1678600, 3.1658773
8: -1.3889146, 1.5655003, -1.4139555, 1.5773144, -2.9115295, 2.9216857
9: -16.1282082, -12.4528685, -16.1369286, -12.4491692, -3.5786219, 3.5812397

Time for backsubstitution: 14.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6199
type: B, layer: 1, pos: 5717
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 6185
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 110

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6199

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0712038, upper bound: 2.0688856
time: 5.87 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0829215, upper bound: 2.0697564
time: 8.76 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -7.1962037, -3.9981275, -7.1989460, -3.9982634, -3.1979403, 3.2008185
1: -13.8657093, -9.6267176, -13.8741722, -9.6148882, -4.1879749, 4.1161251
2: -7.1477780, -3.7779160, -7.1443691, -3.7815518, -3.1452036, 3.0311766
3: -12.8228245, -9.6747799, -12.8298388, -9.6702099, -2.8466682, 2.9091582
4: -6.9567671, -3.4254966, -6.9502468, -3.4285617, -3.3293476, 3.3257446
5: -2.8561811, -0.0099609, -2.8586020, -0.0075650, -2.8486161, 2.8486412
6: 8.6887112, 12.0797606, 8.6971264, 12.0770054, -3.3882942, 3.3826342
7: -18.5944023, -15.0352077, -18.5944290, -15.0313997, -3.1522713, 3.1369195
8: -1.3798304, 1.5649686, -1.3791168, 1.5624599, -2.8718348, 2.9125953
9: -16.0897350, -12.4742203, -16.0894661, -12.4753637, -3.5430412, 3.5485811

Time for backsubstitution: 15.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5717
type: B, layer: 1, pos: 6199
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 6185
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 110

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5717

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0688839, upper bound: 2.0850581
time: 5.65 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0688838, upper bound: 2.0851017
time: 7.02 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -7.2225699, -3.9858637, -7.2047029, -3.9969311, -3.2256389, 3.2188392
1: -13.8923912, -9.6014490, -13.8756723, -9.6076994, -4.2257681, 4.2247181
2: -7.1603184, -3.7666359, -7.1464729, -3.7797542, -3.1687613, 3.1296678
3: -12.8445969, -9.6455135, -12.8321447, -9.6640797, -2.9279699, 2.9225845
4: -6.9636497, -3.4111741, -6.9511638, -3.4252987, -3.3576736, 3.3519177
5: -2.8872805, 0.0136414, -2.8621078, -0.0014501, -2.8858304, 2.8757491
6: 8.6603870, 12.1028709, 8.6948261, 12.0830355, -3.4226484, 3.4080448
7: -18.6052303, -15.0143414, -18.5952663, -15.0260782, -3.1613498, 3.1606693
8: -1.4143691, 1.5762930, -1.3871660, 1.5634279, -2.9045506, 2.9097919
9: -16.1296082, -12.4496088, -16.0999603, -12.4745846, -3.5528822, 3.5856295

Time for backsubstitution: 15.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5717
type: B, layer: 1, pos: 6199
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 6185
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 110

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5717

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0697568, upper bound: 2.0966896
time: 6.17 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0697567, upper bound: 2.0967260
time: 8.78 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -7.2046962, -3.9969320, -7.2269230, -3.9833670, -3.2213292, 3.2299910
1: -13.8756638, -9.6076965, -13.8976297, -9.6007643, -4.2035437, 4.2221670
2: -7.1464734, -3.7797689, -7.1633506, -3.7632923, -3.1545386, 3.1301832
3: -12.8321457, -9.6640778, -12.8480968, -9.6415119, -2.9295993, 2.9301643
4: -6.9511600, -3.4252970, -6.9691343, -3.4091799, -3.3450346, 3.3464804
5: -2.8621018, -0.0014508, -2.8902135, 0.0193095, -2.8814113, 2.8887627
6: 8.6948299, 12.0830297, 8.6571140, 12.1110363, -3.4162064, 3.4259157
7: -18.5952644, -15.0260830, -18.6063347, -15.0107040, -3.1698871, 3.1583695
8: -1.3871617, 1.5634260, -1.4227729, 1.5777063, -2.8934793, 2.9351521
9: -16.0999603, -12.4745846, -16.1397629, -12.4484806, -3.5812941, 3.6016521

Time for backsubstitution: 14.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5717
type: B, layer: 1, pos: 6199
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 6185
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 110

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5717

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0697623, upper bound: 2.0829299
time: 5.91 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0697623, upper bound: 2.0829653
time: 9.48 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -7.2269163, -3.9833684, -7.2269230, -3.9833670, -3.2435493, 3.2435546
1: -13.8976240, -9.6007671, -13.8976297, -9.6007643, -4.2667894, 4.2736721
2: -7.1633496, -3.7633066, -7.1633506, -3.7632923, -3.2030659, 3.1846795
3: -12.8480988, -9.6415148, -12.8480968, -9.6415119, -2.9837456, 2.9844251
4: -6.9691334, -3.4091804, -6.9691343, -3.4091799, -3.3889446, 3.3911672
5: -2.8902075, 0.0193090, -2.8902135, 0.0193095, -2.9095170, 2.9095225
6: 8.6571178, 12.1110287, 8.6571140, 12.1110363, -3.4539185, 3.4539146
7: -18.6063347, -15.0107079, -18.6063347, -15.0107040, -3.1807985, 3.1765070
8: -1.4227667, 1.5777054, -1.4227729, 1.5777063, -2.9251814, 2.9430151
9: -16.1397572, -12.4484863, -16.1397629, -12.4484806, -3.5911121, 3.5967579

Time for backsubstitution: 15.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5717
type: B, layer: 1, pos: 6199
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 6185
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 110

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5717

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0697622, upper bound: 2.0829281
time: 7.66 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0697622, upper bound: 2.0967384
time: 9.25 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 32.23 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 32.23
Output dim: 6, lower bound: -2.0688965, upper bound: 2.0719064
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 32.23
Output dim: 6, lower bound: -2.0688966, upper bound: 2.0719082
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 32.23
Output dim: 6, lower bound: -2.0711981, upper bound: 2.0827762
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 32.23
Output dim: 6, lower bound: -2.0711981, upper bound: 2.0836054
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 32.23
Output dim: 6, lower bound: -2.0712038, upper bound: 2.0688837
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 32.23
Output dim: 6, lower bound: -2.0829217, upper bound: 2.0697565
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 32.23
Output dim: 6, lower bound: -2.0712038, upper bound: 2.0688856
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 32.23
Output dim: 6, lower bound: -2.0829215, upper bound: 2.0697564
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 32.23
Output dim: 6, lower bound: -2.0688839, upper bound: 2.0850581
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 32.23
Output dim: 6, lower bound: -2.0688838, upper bound: 2.0851017
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 32.23
Output dim: 6, lower bound: -2.0697568, upper bound: 2.0966896
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 32.23
Output dim: 6, lower bound: -2.0697567, upper bound: 2.0967260
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 32.23
Output dim: 6, lower bound: -2.0697623, upper bound: 2.0829299
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 32.23
Output dim: 6, lower bound: -2.0697623, upper bound: 2.0829653
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 32.23
Output dim: 6, lower bound: -2.0697622, upper bound: 2.0829281
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 32.23
Output dim: 6, lower bound: -2.0697622, upper bound: 2.0967384

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -7.1698103, -4.0097022, -7.1725883, -4.0098352, -3.1599751, 3.1628861
1: -13.8423929, -9.6361256, -13.8507900, -9.6242790, -4.1628218, 4.0841579
2: -7.1367316, -3.8077552, -7.1333280, -3.8114896, -3.1053352, 3.0095625
3: -12.8189068, -9.6837502, -12.8258877, -9.6790876, -2.8336477, 2.8952212
4: -6.9457083, -3.4329188, -6.9391828, -3.4359038, -3.3119688, 3.3061519
5: -2.8412778, -0.0190537, -2.8437617, -0.0166414, -2.8246365, 2.8247080
6: 8.7167692, 12.0594921, 8.7251978, 12.0567284, -3.3399591, 3.3342943
7: -18.5858459, -15.0514030, -18.5858746, -15.0475941, -3.1283884, 3.1174116
8: -1.3459377, 1.5527754, -1.3452253, 1.5502734, -2.8457232, 2.8686666
9: -16.0781288, -12.4786158, -16.0778465, -12.4797554, -3.5253363, 3.5252295

Time for backsubstitution: 14.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 110

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 515

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0688965, upper bound: 2.0579707
time: 4.94 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0688966, upper bound: 2.0719064
time: 5.05 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -7.1698103, -4.0097022, -7.1989417, -3.9982634, -3.1715469, 3.1892395
1: -13.8423929, -9.6361256, -13.8741608, -9.6148872, -4.1720381, 4.1068525
2: -7.1367316, -3.8077552, -7.1443701, -3.7815685, -3.1349158, 3.0198402
3: -12.8189068, -9.6837502, -12.8298340, -9.6702127, -2.8426609, 2.8989754
4: -6.9457083, -3.4329188, -6.9502439, -3.4285626, -3.3196163, 3.3180990
5: -2.8412778, -0.0190537, -2.8585978, -0.0075665, -2.8337114, 2.8395441
6: 8.7167692, 12.0594921, 8.6971312, 12.0769939, -3.3602247, 3.3623610
7: -18.5858459, -15.0514030, -18.5944271, -15.0314054, -3.1449995, 3.1246052
8: -1.3459377, 1.5527754, -1.3791056, 1.5624599, -2.8563633, 2.9019098
9: -16.0781288, -12.4786158, -16.0894604, -12.4753647, -3.5293188, 3.5372877

Time for backsubstitution: 15.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 110

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 515

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0688965, upper bound: 2.0579725
time: 5.67 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0688966, upper bound: 2.0719082
time: 5.90 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -7.1961355, -3.9974461, -7.1721148, -4.0074310, -3.1887045, 3.1746688
1: -13.8690376, -9.6108456, -13.8469219, -9.6337500, -4.1797314, 4.1044445
2: -7.1492662, -3.7965412, -7.1387625, -3.7987132, -3.1189108, 3.0261645
3: -12.8406696, -9.6544380, -12.8098087, -9.6896496, -2.8456712, 2.9009132
4: -6.9525900, -3.4185562, -6.9444051, -3.4403882, -3.3081417, 3.3209100
5: -2.8722811, 0.0045409, -2.8275211, -0.0239601, -2.8483210, 2.8320620
6: 8.6884260, 12.0826092, 8.7254829, 12.0566225, -3.3681965, 3.3571262
7: -18.5966892, -15.0305042, -18.5843716, -15.0510559, -3.1346931, 3.1405616
8: -1.3805027, 1.5640836, -1.3467014, 1.5518050, -2.8840470, 2.8800554
9: -16.1180382, -12.4540005, -16.0579166, -12.4988928, -3.5495634, 3.5279388

Time for backsubstitution: 14.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 110

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 515

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0711981, upper bound: 2.0688855
time: 8.59 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0711981, upper bound: 2.0827737
time: 8.44 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -7.1961451, -3.9974465, -7.1978512, -3.9971850, -3.1989601, 3.2004046
1: -13.8690414, -9.6108456, -13.8697739, -9.6084986, -4.2112532, 4.2126102
2: -7.1492834, -3.7965415, -7.1460252, -3.7880893, -3.1497183, 3.1409287
3: -12.8406715, -9.6544313, -12.8311958, -9.6659784, -2.9124312, 2.9114561
4: -6.9525909, -3.4185476, -6.9481096, -3.4262710, -3.3469219, 3.3595514
5: -2.8722808, 0.0045476, -2.8580945, -0.0022721, -2.8700087, 2.8626420
6: 8.6884193, 12.0826073, 8.6976080, 12.0773001, -3.3888807, 3.3849993
7: -18.5966949, -15.0305023, -18.5944405, -15.0304689, -3.1483860, 3.1501837
8: -1.3805051, 1.5640931, -1.3783293, 1.5630417, -2.8908730, 2.8852353
9: -16.1180420, -12.4539986, -16.0970974, -12.4752731, -3.5403690, 3.5374956

Time for backsubstitution: 18.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 110

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 515

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0711981, upper bound: 2.0697603
time: 7.70 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0711981, upper bound: 2.0827763
time: 6.91 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -7.1725883, -4.0098352, -7.1935844, -3.9956496, -3.1769388, 3.1837492
1: -13.8507900, -9.6242790, -13.8651066, -9.6268501, -4.0941315, 4.1699400
2: -7.1333280, -3.8114896, -7.1498327, -3.7828190, -3.0337324, 3.1062012
3: -12.8258877, -9.6790876, -12.8254356, -9.6729441, -2.8935103, 2.8407927
4: -6.9391828, -3.4359038, -6.9587307, -3.4245367, -3.3037138, 3.3186140
5: -2.8437617, -0.0166414, -2.8551610, -0.0052352, -2.8385265, 2.8385196
6: 8.7251978, 12.0567284, 8.6881208, 12.0819979, -3.3568001, 3.3686075
7: -18.5858746, -15.0475941, -18.5947552, -15.0359907, -3.1330671, 3.1361938
8: -1.3452253, 1.5502734, -1.3792951, 1.5660353, -2.8788180, 2.8777084
9: -16.0778465, -12.4797554, -16.0969810, -12.4736691, -3.5314894, 3.5435781

Time for backsubstitution: 15.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6199
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 110

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6199

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0850546, upper bound: 2.0579640
time: 9.94 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0850546, upper bound: 2.0688836
time: 10.02 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -7.1783285, -4.0085044, -7.2200403, -3.9836249, -3.1947036, 3.2115359
1: -13.8522902, -9.6170921, -13.8917475, -9.6015749, -4.1886387, 4.2096634
2: -7.1354332, -3.8096910, -7.1629004, -3.7716074, -3.1236963, 3.1297803
3: -12.8281994, -9.6729469, -12.8471632, -9.6434422, -2.9074087, 2.9140935
4: -6.9400978, -3.4326441, -6.9660807, -3.4101689, -3.3285503, 3.3483763
5: -2.8472619, -0.0105228, -2.8861444, 0.0184827, -2.8657446, 2.8756216
6: 8.7228823, 12.0627613, 8.6598644, 12.1053019, -3.3824196, 3.4028969
7: -18.5867138, -15.0422583, -18.6055164, -15.0150814, -3.1569328, 3.1478267
8: -1.3533051, 1.5512390, -1.4139483, 1.5773153, -2.8797092, 2.9106603
9: -16.0883560, -12.4789753, -16.1369171, -12.4491711, -3.5687428, 3.5535316

Time for backsubstitution: 15.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6199
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 110

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6199

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0958748, upper bound: 2.0579645
time: 5.68 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0958749, upper bound: 2.0697564
time: 8.33 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -7.1944971, -3.9966908, -7.1935844, -3.9956496, -3.1988475, 3.1968937
1: -13.8717556, -9.6173630, -13.8651066, -9.6268501, -4.2243252, 4.2232265
2: -7.1488037, -3.7951574, -7.1498327, -3.7828190, -3.1626153, 3.1605372
3: -12.8417521, -9.6579609, -12.8254356, -9.6729441, -2.9475155, 2.9390187
4: -6.9560771, -3.4198713, -6.9587307, -3.4245367, -3.3563395, 3.3647394
5: -2.8716502, 0.0035911, -2.8551610, -0.0052352, -2.8664150, 2.8587520
6: 8.6874895, 12.0840578, 8.6881208, 12.0819979, -3.3945084, 3.3959370
7: -18.5968075, -15.0322895, -18.5947552, -15.0359907, -3.1436453, 3.1486845
8: -1.3800657, 1.5645313, -1.3792951, 1.5660353, -2.8911219, 2.8852777
9: -16.1174965, -12.4538527, -16.0969810, -12.4736691, -3.5426407, 3.5383739

Time for backsubstitution: 14.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6199
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 110

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6199

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0713758, upper bound: 2.0720542
time: 11.28 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0713758, upper bound: 2.0828880
time: 11.18 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -7.2004533, -3.9949546, -7.2200403, -3.9836249, -3.2168283, 3.2250857
1: -13.8742809, -9.6101770, -13.8917475, -9.6015749, -4.2518520, 4.2611599
2: -7.1523123, -3.7931917, -7.1629004, -3.7716074, -3.1738153, 3.1843348
3: -12.8441830, -9.6504574, -12.8471632, -9.6434422, -2.9617181, 2.9683113
4: -6.9580717, -3.4165759, -6.9660807, -3.4101689, -3.3725119, 3.3931675
5: -2.8751900, 0.0102272, -2.8861444, 0.0184827, -2.8936727, 2.8963716
6: 8.6851244, 12.0907640, 8.6598644, 12.1053019, -3.4201775, 3.4308996
7: -18.5978031, -15.0268488, -18.6055164, -15.0150814, -3.1678352, 3.1660037
8: -1.3889117, 1.5654993, -1.4139483, 1.5773153, -2.9115267, 2.9184828
9: -16.1282082, -12.4528713, -16.1369171, -12.4491711, -3.5786180, 3.5486412

Time for backsubstitution: 14.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6199
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 110

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6199

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0821783, upper bound: 2.0720542
time: 5.48 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0821783, upper bound: 2.0720518
time: 12.19 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -7.1962037, -3.9981275, -7.1725883, -4.0098352, -3.1863685, 3.1744609
1: -13.8657093, -9.6267176, -13.8507900, -9.6242790, -4.1856356, 4.0934219
2: -7.1477780, -3.7779160, -7.1333280, -3.8114896, -3.1156130, 3.0392752
3: -12.8228245, -9.6747799, -12.8258877, -9.6790876, -2.8374491, 2.9045181
4: -6.9567671, -3.4254966, -6.9391828, -3.4359038, -3.3239193, 3.3137918
5: -2.8561811, -0.0099609, -2.8437617, -0.0166414, -2.8395398, 2.8338008
6: 8.6887112, 12.0797606, 8.7251978, 12.0567284, -3.3680172, 3.3545628
7: -18.5944023, -15.0352077, -18.5858746, -15.0475941, -3.1356516, 3.1340451
8: -1.3798304, 1.5649686, -1.3452253, 1.5502734, -2.8789997, 2.8793406
9: -16.0897350, -12.4742203, -16.0778465, -12.4797554, -3.5373831, 3.5292034

Time for backsubstitution: 15.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 110

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 515

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0688838, upper bound: 2.0712035
time: 5.45 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0688838, upper bound: 2.0850581
time: 5.69 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -7.1962037, -3.9981275, -7.1989417, -3.9982634, -3.1979403, 3.2008142
1: -13.8657093, -9.6267176, -13.8741608, -9.6148872, -4.1879721, 4.1092453
2: -7.1477780, -3.7779160, -7.1443701, -3.7815685, -3.1268139, 3.0311747
3: -12.8228245, -9.6747799, -12.8298340, -9.6702127, -2.8473454, 2.9091573
4: -6.9567671, -3.4254966, -6.9502439, -3.4285626, -3.3293457, 3.3235188
5: -2.8561811, -0.0099609, -2.8585978, -0.0075665, -2.8486147, 2.8486369
6: 8.6887112, 12.0797606, 8.6971312, 12.0769939, -3.3882828, 3.3826294
7: -18.5944023, -15.0352077, -18.5944271, -15.0314054, -3.1479616, 3.1369162
8: -1.3798304, 1.5649686, -1.3791056, 1.5624599, -2.8718328, 2.8948126
9: -16.0897350, -12.4742203, -16.0894604, -12.4753647, -3.5486794, 3.5485754

Time for backsubstitution: 15.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 110

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 515

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0688839, upper bound: 2.0712478
time: 7.48 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0688838, upper bound: 2.0851017
time: 6.46 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 29.58 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 29.58
Output dim: 6, lower bound: -2.0688965, upper bound: 2.0579707
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 29.58
Output dim: 6, lower bound: -2.0688966, upper bound: 2.0719064
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 29.58
Output dim: 6, lower bound: -2.0688965, upper bound: 2.0579725
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 29.58
Output dim: 6, lower bound: -2.0688966, upper bound: 2.0719082
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 29.58
Output dim: 6, lower bound: -2.0711981, upper bound: 2.0688855
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 29.58
Output dim: 6, lower bound: -2.0711981, upper bound: 2.0827737
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 29.58
Output dim: 6, lower bound: -2.0711981, upper bound: 2.0697603
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 29.58
Output dim: 6, lower bound: -2.0711981, upper bound: 2.0827763
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 29.58
Output dim: 6, lower bound: -2.0850546, upper bound: 2.0579640
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 29.58
Output dim: 6, lower bound: -2.0850546, upper bound: 2.0688836
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 29.58
Output dim: 6, lower bound: -2.0958748, upper bound: 2.0579645
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 29.58
Output dim: 6, lower bound: -2.0958749, upper bound: 2.0697564
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 29.58
Output dim: 6, lower bound: -2.0713758, upper bound: 2.0720542
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 29.58
Output dim: 6, lower bound: -2.0713758, upper bound: 2.0828880
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 29.58
Output dim: 6, lower bound: -2.0821783, upper bound: 2.0720542
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 29.58
Output dim: 6, lower bound: -2.0821783, upper bound: 2.0720518
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 29.58
Output dim: 6, lower bound: -2.0688838, upper bound: 2.0712035
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 29.58
Output dim: 6, lower bound: -2.0688838, upper bound: 2.0850581
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 29.58
Output dim: 6, lower bound: -2.0688839, upper bound: 2.0712478
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 29.58
Output dim: 6, lower bound: -2.0688838, upper bound: 2.0851017
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 29.58
Output dim: 6, lower bound: -2.0697568, upper bound: 2.0966896
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 29.58
Output dim: 6, lower bound: -2.0697567, upper bound: 2.0967260
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 29.58
Output dim: 6, lower bound: -2.0697623, upper bound: 2.0829299
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.58
Output dim: 6, lower bound: -2.0697623, upper bound: 2.0829653
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 29.58
Output dim: 6, lower bound: -2.0697622, upper bound: 2.0829281
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 29.58
Output dim: 6, lower bound: -2.0697622, upper bound: 2.0967384
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=3.453947067260742
rel_dist={6: [-2.09676497388166, 2.096767128916582]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 5717
type: A, layer: 1, pos: 6199
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 110

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 515

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6521165, upper bound: 1.6441523
time: 7.81 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6524956, upper bound: 1.6524946
time: 5.54 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 13.58 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 13.58
Output dim: 6, lower bound: -1.6521165, upper bound: 1.6441523
IS_A2, status: Status.UNKNOWN, split count: 1, time: 13.58
Output dim: 6, lower bound: -1.6524956, upper bound: 1.6524946

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -7.2047043, -3.9969299, -7.2195220, -3.9877000, -3.2031307, 3.2083387
1: -13.8756742, -9.6076956, -13.8888397, -9.6018991, -3.8241987, 3.8272257
2: -7.1464763, -3.7797542, -7.1580505, -3.7689996, -2.8801818, 2.8801003
3: -12.8321466, -9.6640778, -12.8421078, -9.6484785, -2.6009092, 2.5953355
4: -6.9511647, -3.4252973, -6.9598460, -3.4125862, -3.0487642, 3.0389671
5: -2.8621097, -0.0014484, -2.8851871, 0.0096278, -2.7256336, 2.7371669
6: 8.6948242, 12.0830383, 8.6627445, 12.0971184, -3.2182198, 3.2348633
7: -18.5952663, -15.0260735, -18.6044407, -15.0169096, -2.7899179, 2.7885556
8: -1.3871694, 1.5634274, -1.4084203, 1.5752726, -2.6505070, 2.6605744
9: -16.0999680, -12.4745846, -16.1223965, -12.4504328, -3.2513676, 3.2503185

Time for backsubstitution: 15.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5717
type: B, layer: 1, pos: 6199
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 6185
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 110

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5717

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6444017, upper bound: 1.6441466
time: 8.14 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6521112, upper bound: 1.6441472
time: 5.94 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -7.2269239, -3.9833694, -7.2269287, -3.9833653, -3.2258224, 3.2292738
1: -13.8976297, -9.6007652, -13.8976345, -9.6007652, -3.8676128, 3.8302717
2: -7.1633496, -3.7632909, -7.1633520, -3.7632875, -2.9211035, 2.8928485
3: -12.8480988, -9.6415119, -12.8481035, -9.6415081, -2.6179380, 2.6447363
4: -6.9691348, -3.4091814, -6.9691401, -3.4091792, -3.0780954, 3.0552464
5: -2.8902125, 0.0193090, -2.8902183, 0.0193155, -2.7737646, 2.7588558
6: 8.6571150, 12.1110344, 8.6571131, 12.1110439, -3.2696838, 3.2506399
7: -18.6063347, -15.0107040, -18.6063347, -15.0107002, -2.8105373, 2.8089581
8: -1.4227738, 1.5777054, -1.4227839, 1.5777082, -2.6836252, 2.6915011
9: -16.1397629, -12.4484825, -16.1397743, -12.4484825, -3.2578783, 3.2949972

Time for backsubstitution: 15.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5717
type: B, layer: 1, pos: 6199
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 6185
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 110

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5717

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6448064, upper bound: 1.6524895
time: 25.74 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6524903, upper bound: 1.6524895
time: 5.35 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 46.30 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 46.30
Output dim: 6, lower bound: -1.6444017, upper bound: 1.6441466
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 46.30
Output dim: 6, lower bound: -1.6521112, upper bound: 1.6441472
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 46.30
Output dim: 6, lower bound: -1.6448064, upper bound: 1.6524895
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 46.30
Output dim: 6, lower bound: -1.6524903, upper bound: 1.6524895

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -7.1933823, -3.9973483, -7.1931186, -3.9992785, -3.1744280, 3.1814957
1: -13.8659115, -9.6090050, -13.8654747, -9.6112967, -3.8006811, 3.8030624
2: -7.1457376, -3.7935553, -7.1470141, -3.7989345, -2.8496141, 2.8535771
3: -12.8305807, -9.6672211, -12.8381729, -9.6573868, -2.5899248, 2.5882440
4: -6.9461079, -3.4269025, -6.9487810, -3.4199574, -3.0358133, 3.0252995
5: -2.8554688, -0.0028119, -2.8701887, 0.0005279, -2.7036123, 2.7216644
6: 8.6994543, 12.0735474, 8.6908016, 12.0768471, -3.1938810, 3.1980448
7: -18.5938931, -15.0333290, -18.5958881, -15.0330858, -2.7721062, 2.7694263
8: -1.3725526, 1.5627885, -1.3745370, 1.5630798, -2.6200547, 2.6267233
9: -16.0952396, -12.4757318, -16.1108093, -12.4548292, -3.2395086, 3.2373867

Time for backsubstitution: 15.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6199
type: A, layer: 1, pos: 5717
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 110

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6199

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6436737, upper bound: 1.6375477
time: 5.77 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6443992, upper bound: 1.6441425
time: 11.44 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -7.2047009, -3.9969313, -7.2195163, -3.9877005, -3.2028227, 3.1887684
1: -13.8756695, -9.6076975, -13.8888340, -9.6019020, -3.8219595, 3.8147602
2: -7.1464725, -3.7797582, -7.1580510, -3.7690141, -2.8557110, 2.8784051
3: -12.8321466, -9.6640768, -12.8421097, -9.6484833, -2.6015000, 2.5952578
4: -6.9511604, -3.4252970, -6.9598427, -3.4125865, -3.0487614, 3.0363541
5: -2.8621078, -0.0014491, -2.8851814, 0.0096269, -2.7219419, 2.7350349
6: 8.6948252, 12.0830355, 8.6627455, 12.0971079, -3.2005320, 3.2348566
7: -18.5952644, -15.0260725, -18.6044407, -15.0169134, -2.7803936, 2.7861686
8: -1.3871655, 1.5634260, -1.4084134, 1.5752721, -2.6484261, 2.6355457
9: -16.0999641, -12.4745836, -16.1223907, -12.4504356, -3.2562618, 3.2496948

Time for backsubstitution: 14.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6199
type: A, layer: 1, pos: 5717
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 110

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6199

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6513902, upper bound: 1.6375477
time: 10.71 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6521086, upper bound: 1.6441430
time: 6.47 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -7.2155471, -3.9837959, -7.2004590, -3.9949517, -3.1971550, 3.2023821
1: -13.8878937, -9.6020813, -13.8742867, -9.6101704, -3.8440838, 3.8061905
2: -7.1626158, -3.7770600, -7.1523175, -3.7931862, -2.8906174, 2.8663664
3: -12.8465519, -9.6446953, -12.8441868, -9.6504507, -2.6069336, 2.6376328
4: -6.9640822, -3.4108148, -6.9580774, -3.4165735, -3.0651875, 3.0415382
5: -2.8834805, 0.0179362, -2.8751941, 0.0102344, -2.7516575, 2.7433562
6: 8.6616840, 12.1015453, 8.6851215, 12.0907726, -3.2453690, 3.2138252
7: -18.6049786, -15.0179386, -18.5978069, -15.0268421, -2.7927566, 2.7898526
8: -1.4081717, 1.5770578, -1.3889260, 1.5655022, -2.6532230, 2.6576958
9: -16.1350670, -12.4496231, -16.1282196, -12.4528704, -3.2460575, 3.2821026

Time for backsubstitution: 14.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6199
type: A, layer: 1, pos: 5717
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 110

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6199

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6441558, upper bound: 1.6460646
time: 9.21 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6448039, upper bound: 1.6524867
time: 10.33 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -7.2269201, -3.9833689, -7.2269230, -3.9833655, -3.2255182, 3.2096176
1: -13.8976288, -9.6007671, -13.8976288, -9.6007662, -3.8653736, 3.8178291
2: -7.1633501, -3.7632985, -7.1633525, -3.7633018, -2.8966331, 2.8911552
3: -12.8480997, -9.6415119, -12.8481035, -9.6415100, -2.6185279, 2.6446590
4: -6.9691334, -3.4091811, -6.9691358, -3.4091792, -3.0780935, 3.0526314
5: -2.8902121, 0.0193086, -2.8902102, 0.0193157, -2.7700596, 2.7567825
6: 8.6571140, 12.1110344, 8.6571131, 12.1110401, -3.2519970, 3.2506323
7: -18.6063347, -15.0107059, -18.6063347, -15.0107059, -2.8010435, 2.8065681
8: -1.4227710, 1.5777059, -1.4227757, 1.5777059, -2.6815476, 2.6664929
9: -16.1397629, -12.4484844, -16.1397667, -12.4484844, -3.2627716, 3.2943745

Time for backsubstitution: 14.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6199
type: A, layer: 1, pos: 5717
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 110

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6199

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6518485, upper bound: 1.6460650
time: 6.82 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6524878, upper bound: 1.6524863
time: 7.73 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 29.65 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 29.65
Output dim: 6, lower bound: -1.6436737, upper bound: 1.6375477
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 29.65
Output dim: 6, lower bound: -1.6443992, upper bound: 1.6441425
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 29.65
Output dim: 6, lower bound: -1.6513902, upper bound: 1.6375477
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 29.65
Output dim: 6, lower bound: -1.6521086, upper bound: 1.6441430
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 29.65
Output dim: 6, lower bound: -1.6441558, upper bound: 1.6460646
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 29.65
Output dim: 6, lower bound: -1.6448039, upper bound: 1.6524867
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 29.65
Output dim: 6, lower bound: -1.6518485, upper bound: 1.6460650
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 29.65
Output dim: 6, lower bound: -1.6524878, upper bound: 1.6524863

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -7.1676426, -4.0075960, -7.1828203, -4.0024843, -3.1417971, 3.1419525
1: -13.8430605, -9.6342716, -13.8611345, -9.6238728, -3.6793175, 3.7747135
2: -7.1384664, -3.8041675, -7.1411190, -3.8023782, -2.7442298, 2.8320889
3: -12.8091936, -9.6909103, -12.8339128, -9.6704731, -2.5510678, 2.5022655
4: -6.9424005, -3.4410365, -6.9457722, -3.4256976, -3.0076828, 3.0028791
5: -2.8249123, -0.0244980, -2.8639059, -0.0110002, -2.7217760, 2.6915340
6: 8.7273369, 12.0528593, 8.6950226, 12.0652466, -3.1569433, 3.1738253
7: -18.5838280, -15.0539379, -18.5940876, -15.0425301, -2.7518950, 2.7575788
8: -1.3408940, 1.5515518, -1.3591542, 1.5613499, -2.6119356, 2.6006665
9: -16.0560341, -12.4993534, -16.0921497, -12.4566326, -3.1965437, 3.1943607

Time for backsubstitution: 14.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 6199
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 6185
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 110

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 515

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6356726, upper bound: 1.6375475
time: 5.91 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6356726, upper bound: 1.6375473
time: 20.03 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -7.1933708, -3.9973497, -7.1931095, -3.9992795, -3.1650734, 3.1787987
1: -13.8659077, -9.6090193, -13.8654737, -9.6113052, -3.8040800, 3.8021259
2: -7.1457314, -3.7935576, -7.1470127, -3.7989352, -2.8561606, 2.8418961
3: -12.8305740, -9.6672316, -12.8381691, -9.6573915, -2.5849748, 2.5693417
4: -6.9461079, -3.4269114, -6.9487805, -3.4199622, -3.0460939, 3.0174088
5: -2.8554649, -0.0028172, -2.8701854, 0.0005252, -2.7036009, 2.6982126
6: 8.6994553, 12.0735359, 8.6908026, 12.0768414, -3.1938753, 3.1816769
7: -18.5938931, -15.0333471, -18.5958843, -15.0330925, -2.7722154, 2.7693868
8: -1.3725410, 1.5627880, -1.3745317, 1.5630789, -2.6162882, 2.6267176
9: -16.0952225, -12.4757290, -16.1108017, -12.4548302, -3.2011814, 3.2373800

Time for backsubstitution: 14.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 6199
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 6185
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 110

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 515

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6364322, upper bound: 1.6441447
time: 7.74 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6364322, upper bound: 1.6441448
time: 6.97 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -7.1789398, -4.0071845, -7.2091961, -3.9909105, -3.1701632, 3.1491165
1: -13.8528061, -9.6329603, -13.8844852, -9.6144743, -3.7005558, 3.7863989
2: -7.1392069, -3.7904074, -7.1521587, -3.7724624, -2.7504244, 2.8568935
3: -12.8107471, -9.6877441, -12.8378525, -9.6615639, -2.5626097, 2.5093031
4: -6.9474549, -3.4394131, -6.9568315, -3.4183180, -3.0206013, 3.0139194
5: -2.8314984, -0.0231478, -2.8788848, -0.0018923, -2.7400074, 2.7048464
6: 8.7226915, 12.0623493, 8.6669455, 12.0855122, -3.1636076, 3.2106380
7: -18.5852013, -15.0466614, -18.6026421, -15.0263329, -2.7601705, 2.7744026
8: -1.3555503, 1.5521889, -1.3930550, 1.5735393, -2.6403503, 2.6095176
9: -16.0607891, -12.4982014, -16.1037521, -12.4522305, -3.2133312, 3.2066975

Time for backsubstitution: 14.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 6199
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 6185
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 110

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 515

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6433878, upper bound: 1.6375497
time: 6.80 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6433876, upper bound: 1.6375475
time: 17.45 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -7.2046919, -3.9969306, -7.2195101, -3.9877024, -3.1934643, 3.1860771
1: -13.8756647, -9.6077118, -13.8888321, -9.6019115, -3.8253584, 3.8138218
2: -7.1464701, -3.7797627, -7.1580472, -3.7690153, -2.8622031, 2.8667493
3: -12.8321438, -9.6640854, -12.8421059, -9.6484861, -2.5965581, 2.5763721
4: -6.9511590, -3.4253016, -6.9598408, -3.4125898, -3.0589933, 3.0284352
5: -2.8621016, -0.0014534, -2.8851776, 0.0096254, -2.7219305, 2.7115822
6: 8.6948299, 12.0830297, 8.6627493, 12.0971050, -3.2005281, 3.2184868
7: -18.5952644, -15.0260878, -18.6044369, -15.0169191, -2.7805004, 2.7861271
8: -1.3871531, 1.5634265, -1.4084055, 1.5752726, -2.6446600, 2.6355391
9: -16.0999527, -12.4745865, -16.1223850, -12.4504366, -3.2179356, 3.2496862

Time for backsubstitution: 14.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 6199
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 6185
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 110

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 515

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6441447, upper bound: 1.6441430
time: 7.05 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6441445, upper bound: 1.6441426
time: 7.64 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -7.1890874, -3.9958148, -7.1900759, -3.9980049, -3.1634274, 3.1742802
1: -13.8612537, -9.6273785, -13.8698425, -9.6227388, -3.8039427, 3.7769566
2: -7.1495357, -3.7882631, -7.1462431, -3.7966223, -2.8751049, 2.8442054
3: -12.8248272, -9.6742210, -12.8399277, -9.6635380, -2.5675955, 2.6048589
4: -6.9567299, -3.4251997, -6.9546251, -3.4223125, -3.0476599, 3.0180526
5: -2.8525128, -0.0057840, -2.8689876, -0.0013623, -2.7047958, 2.7076035
6: 8.6899529, 12.0782337, 8.6892662, 12.0790615, -3.2056246, 3.1858683
7: -18.5942116, -15.0388641, -18.5960484, -15.0363255, -2.7709670, 2.7648783
8: -1.3735023, 1.5657768, -1.3734841, 1.5638013, -2.6161051, 2.6304026
9: -16.0951099, -12.4741249, -16.1095161, -12.4545975, -3.2024260, 3.2377110

Time for backsubstitution: 14.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 6199
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 6185
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 110

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 515

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6356713, upper bound: 1.6455569
time: 6.21 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6356713, upper bound: 1.6460670
time: 10.23 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -7.2155352, -3.9837952, -7.2004561, -3.9949522, -3.1883574, 3.1988525
1: -13.8878889, -9.6020994, -13.8742847, -9.6101780, -3.8474903, 3.8052502
2: -7.1626101, -3.7770631, -7.1523156, -3.7931864, -2.8972511, 2.8561420
3: -12.8465500, -9.6447067, -12.8441830, -9.6504574, -2.6018791, 2.6172848
4: -6.9640789, -3.4108233, -6.9580765, -3.4165761, -3.0755730, 3.0336819
5: -2.8834760, 0.0179317, -2.8751922, 0.0102305, -2.7516050, 2.7199879
6: 8.6616888, 12.1015368, 8.6851254, 12.0907736, -3.2453651, 3.1974545
7: -18.6049767, -15.0179482, -18.5978050, -15.0268497, -2.7928624, 2.7898140
8: -1.4081616, 1.5770569, -1.3889201, 1.5655012, -2.6494570, 2.6576896
9: -16.1350555, -12.4496260, -16.1282139, -12.4528713, -3.2077227, 3.2820959

Time for backsubstitution: 14.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 6199
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 6185
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 110

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 515

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6364310, upper bound: 1.6521074
time: 6.74 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6364311, upper bound: 1.6524872
time: 9.58 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -7.2004433, -3.9953947, -7.2165151, -3.9864221, -3.1917677, 3.1815243
1: -13.8709717, -9.6260567, -13.8931789, -9.6133356, -3.8252287, 3.7886782
2: -7.1502790, -3.7745314, -7.1572886, -3.7667425, -2.8811088, 2.8689828
3: -12.8263645, -9.6710196, -12.8438511, -9.6545925, -2.5791817, 2.6119056
4: -6.9617805, -3.4235330, -6.9656858, -3.4149120, -3.0605431, 3.0291371
5: -2.8591948, -0.0044124, -2.8839960, 0.0077271, -2.7231874, 2.7210274
6: 8.6853609, 12.0877190, 8.6612396, 12.0993252, -3.2122650, 3.2226801
7: -18.5955830, -15.0316124, -18.6045818, -15.0201569, -2.7792482, 2.7816062
8: -1.3881202, 1.5664244, -1.4073610, 1.5760007, -2.6444578, 2.6392040
9: -16.0998287, -12.4729805, -16.1210899, -12.4502068, -3.2191792, 3.2500095

Time for backsubstitution: 14.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 6199
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 6185
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 110

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 515

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6433865, upper bound: 1.6455591
time: 7.22 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6433863, upper bound: 1.6460654
time: 7.76 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -7.2269087, -3.9833705, -7.2269173, -3.9833665, -3.2167177, 3.2060928
1: -13.8976231, -9.6007843, -13.8976288, -9.6007757, -3.8687782, 3.8168859
2: -7.1633420, -3.7632995, -7.1633472, -3.7633030, -2.9032106, 2.8809528
3: -12.8480968, -9.6415234, -12.8481016, -9.6415157, -2.6134834, 2.6242924
4: -6.9691315, -3.4091840, -6.9691343, -3.4091830, -3.0884333, 3.0447502
5: -2.8902073, 0.0193040, -2.8902092, 0.0193119, -2.7700424, 2.7333312
6: 8.6571178, 12.1110239, 8.6571140, 12.1110334, -3.2519922, 3.2342625
7: -18.6063309, -15.0107155, -18.6063347, -15.0107117, -2.8011508, 2.8065295
8: -1.4227605, 1.5777054, -1.4227712, 1.5777059, -2.6777811, 2.6664858
9: -16.1397495, -12.4484844, -16.1397629, -12.4484854, -3.2244339, 3.2943668

Time for backsubstitution: 14.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 6199
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 6185
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 110

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 515

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6441433, upper bound: 1.6521093
time: 6.80 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6441434, upper bound: 1.6524865
time: 11.08 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 32.95 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 32.95
Output dim: 6, lower bound: -1.6356726, upper bound: 1.6375475
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 32.95
Output dim: 6, lower bound: -1.6356726, upper bound: 1.6375473
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 32.95
Output dim: 6, lower bound: -1.6364322, upper bound: 1.6441447
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 32.95
Output dim: 6, lower bound: -1.6364322, upper bound: 1.6441448
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 32.95
Output dim: 6, lower bound: -1.6433878, upper bound: 1.6375497
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 32.95
Output dim: 6, lower bound: -1.6433876, upper bound: 1.6375475
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 32.95
Output dim: 6, lower bound: -1.6441447, upper bound: 1.6441430
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 32.95
Output dim: 6, lower bound: -1.6441445, upper bound: 1.6441426
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 32.95
Output dim: 6, lower bound: -1.6356713, upper bound: 1.6455569
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 32.95
Output dim: 6, lower bound: -1.6356713, upper bound: 1.6460670
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 32.95
Output dim: 6, lower bound: -1.6364310, upper bound: 1.6521074
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 32.95
Output dim: 6, lower bound: -1.6364311, upper bound: 1.6524872
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 32.95
Output dim: 6, lower bound: -1.6433865, upper bound: 1.6455591
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 32.95
Output dim: 6, lower bound: -1.6433863, upper bound: 1.6460654
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 32.95
Output dim: 6, lower bound: -1.6441433, upper bound: 1.6521093
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 32.95
Output dim: 6, lower bound: -1.6441434, upper bound: 1.6524865

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -7.1676426, -4.0075960, -7.1683364, -4.0107112, -3.1192122, 3.1251240
1: -13.8430605, -9.6342716, -13.8497782, -9.6296558, -3.6713810, 3.6784143
2: -7.1384664, -3.8041675, -7.1320252, -3.8128061, -2.7339849, 2.7336636
3: -12.8091936, -9.6909103, -12.8241634, -9.6833858, -2.4849100, 2.4925194
4: -6.9424005, -3.4410365, -6.9385724, -3.4383142, -2.9839535, 2.9778495
5: -2.8249123, -0.0244980, -2.8411374, -0.0211420, -2.7089815, 2.7347693
6: 8.7273369, 12.0528593, 8.7269154, 12.0523052, -3.1434937, 3.1443253
7: -18.5838280, -15.0539379, -18.5852890, -15.0515537, -2.7478304, 2.7465396
8: -1.3408940, 1.5515518, -1.3393264, 1.5495520, -2.5974240, 2.5979943
9: -16.0560341, -12.4993534, -16.0700378, -12.4802866, -3.1729078, 3.1708689

Time for backsubstitution: 14.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5717
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 110

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5717

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6356726, upper bound: 1.6308240
time: 9.35 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6356726, upper bound: 1.6375475
time: 10.46 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -7.1676426, -4.0075960, -7.1900606, -3.9980059, -3.1460495, 3.1468315
1: -13.8430605, -9.6342716, -13.8698368, -9.6227388, -3.6793022, 3.7685318
2: -7.1384664, -3.8041675, -7.1462312, -3.7966270, -2.7498126, 2.8222637
3: -12.8091936, -9.6909103, -12.8399267, -9.6635485, -2.5443826, 2.5095239
4: -6.9424005, -3.4410365, -6.9546189, -3.4223237, -3.0000639, 3.0021868
5: -2.8249123, -0.0244980, -2.8689840, -0.0013778, -2.7332888, 2.7007875
6: 8.7273369, 12.0528593, 8.6892710, 12.0790548, -3.1718531, 3.1785440
7: -18.5838280, -15.0539379, -18.5960484, -15.0363283, -2.7601514, 2.7572632
8: -1.3408940, 1.5515518, -1.3734722, 1.5637989, -2.6105194, 2.6169820
9: -16.0560341, -12.4993534, -16.1095066, -12.4545956, -3.1983786, 3.2123375

Time for backsubstitution: 14.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5717
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 110

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5717

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6356726, upper bound: 1.6308261
time: 13.20 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6356726, upper bound: 1.6375477
time: 12.85 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -7.1933708, -3.9973497, -7.1783247, -4.0085058, -3.1555910, 3.1647482
1: -13.8659077, -9.6090193, -13.8522902, -9.6170969, -3.7818499, 3.7769356
2: -7.1457314, -3.7935576, -7.1354308, -3.8096924, -2.8362818, 2.8204222
3: -12.8305740, -9.6672316, -12.8281994, -9.6729488, -2.5603547, 2.5502710
4: -6.9461079, -3.4269114, -6.9400983, -3.4326437, -3.0209608, 3.0020266
5: -2.8554649, -0.0028172, -2.8472600, -0.0105238, -2.6893978, 2.6726117
6: 8.6994553, 12.0735359, 8.7228823, 12.0627584, -3.1787586, 3.1499224
7: -18.5938931, -15.0333471, -18.5867138, -15.0422611, -2.7610912, 2.7596316
8: -1.3725410, 1.5627880, -1.3533027, 1.5512381, -2.6042957, 2.6046476
9: -16.0952225, -12.4757290, -16.0883541, -12.4789753, -3.1767941, 3.2140217

Time for backsubstitution: 15.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5717
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 110

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5717

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6364324, upper bound: 1.6374187
time: 6.93 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6364323, upper bound: 1.6441435
time: 7.02 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -7.1933708, -3.9973497, -7.2004471, -3.9949543, -3.1697664, 3.1866245
1: -13.8659077, -9.6090193, -13.8742781, -9.6101789, -3.7901001, 3.7969809
2: -7.1457314, -3.7935576, -7.1523128, -3.7931910, -2.8524246, 2.8335767
3: -12.8305740, -9.6672316, -12.8441820, -9.6504612, -2.5781460, 2.5681472
4: -6.9461079, -3.4269114, -6.9580703, -3.4165783, -3.0371380, 3.0174637
5: -2.8554649, -0.0028172, -2.8751879, 0.0102270, -2.7170506, 2.7072330
6: 8.6994553, 12.0735359, 8.6851273, 12.0907621, -3.2089396, 3.1863098
7: -18.5938931, -15.0333471, -18.5978031, -15.0268517, -2.7806082, 2.7718749
8: -1.3725410, 1.5627880, -1.3889096, 1.5654993, -2.6188078, 2.6431527
9: -16.0952225, -12.4757290, -16.1282043, -12.4528713, -3.2034492, 3.2554359

Time for backsubstitution: 14.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5717
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 110

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5717

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6364324, upper bound: 1.6374202
time: 7.12 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6364323, upper bound: 1.6441448
time: 7.34 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -7.1789398, -4.0071845, -7.1946764, -3.9991434, -3.1475048, 3.1322594
1: -13.8528061, -9.6329603, -13.8731499, -9.6202660, -3.6926289, 3.6900101
2: -7.1392069, -3.7904074, -7.1430688, -3.7828863, -2.7401657, 2.7585616
3: -12.8107471, -9.6877441, -12.8281126, -9.6745071, -2.4963093, 2.4995613
4: -6.9474549, -3.4394131, -6.9496331, -3.4309716, -2.9968853, 2.9888897
5: -2.8314984, -0.0231478, -2.8559725, -0.0120645, -2.7272091, 2.7478919
6: 8.7226915, 12.0623493, 8.6988373, 12.0725756, -3.1501589, 3.1811333
7: -18.5852013, -15.0466614, -18.5938416, -15.0353527, -2.7561741, 2.7633562
8: -1.3555503, 1.5521889, -1.3732290, 1.5617380, -2.6258273, 2.6069427
9: -16.0607891, -12.4982014, -16.0816574, -12.4758949, -3.1896887, 3.1832218

Time for backsubstitution: 14.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5717
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 110

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5717

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6356726, upper bound: 1.6298387
time: 6.03 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6356727, upper bound: 1.6375484
time: 6.28 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -7.1789398, -4.0071845, -7.2164984, -3.9864273, -3.1744204, 3.1540489
1: -13.8528061, -9.6329603, -13.8931713, -9.6133356, -3.7005444, 3.7801466
2: -7.1392069, -3.7904074, -7.1572752, -3.7667487, -2.7559748, 2.8470697
3: -12.8107471, -9.6877441, -12.8438454, -9.6546030, -2.5559826, 2.5165401
4: -6.9474549, -3.4394131, -6.9656801, -3.4149206, -3.0130634, 3.0132246
5: -2.8314984, -0.0231478, -2.8839915, 0.0077136, -2.7515392, 2.7141719
6: 8.7226915, 12.0623493, 8.6612425, 12.0993156, -3.1785173, 3.2153549
7: -18.5852013, -15.0466614, -18.6045780, -15.0201578, -2.7684255, 2.7740798
8: -1.3555503, 1.5521889, -1.4073491, 1.5759993, -2.6389532, 2.6258225
9: -16.0607891, -12.4982014, -16.1210785, -12.4502068, -3.2151566, 3.2246399

Time for backsubstitution: 14.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5717
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 110

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5717

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6356726, upper bound: 1.6298362
time: 7.47 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6356724, upper bound: 1.6375497
time: 8.67 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -7.2046919, -3.9969306, -7.2046938, -3.9969294, -3.1839762, 3.1719055
1: -13.8756647, -9.6077118, -13.8756618, -9.6077042, -3.8031387, 3.7886038
2: -7.1464701, -3.7797627, -7.1464696, -3.7797699, -2.8422546, 2.8452773
3: -12.8321438, -9.6640854, -12.8321438, -9.6640835, -2.5719156, 2.5572791
4: -6.9511590, -3.4253016, -6.9511595, -3.4253006, -3.0338745, 3.0130548
5: -2.8621016, -0.0014534, -2.8621004, -0.0014534, -2.7076912, 2.6859007
6: 8.6948299, 12.0830297, 8.6948280, 12.0830278, -3.1854124, 3.1867247
7: -18.5952644, -15.0260878, -18.5952644, -15.0260849, -2.7694154, 2.7763643
8: -1.3871531, 1.5634265, -1.3871546, 1.5634270, -2.6326494, 2.6135731
9: -16.0999527, -12.4745865, -16.0999546, -12.4745855, -3.1935453, 3.2263536

Time for backsubstitution: 14.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5717
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 110

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5717

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6364324, upper bound: 1.6364324
time: 14.31 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6364323, upper bound: 1.6441429
time: 13.15 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -7.2046919, -3.9969306, -7.2269120, -3.9833684, -3.1981659, 3.1938667
1: -13.8756647, -9.6077118, -13.8976221, -9.6007767, -3.8113985, 3.8086042
2: -7.1464701, -3.7797627, -7.1633463, -3.7633080, -2.8583789, 2.8584337
3: -12.8321438, -9.6640854, -12.8480988, -9.6415224, -2.5897522, 2.5751534
4: -6.9511590, -3.4253016, -6.9691300, -3.4091849, -3.0501194, 3.0284886
5: -2.8621016, -0.0014534, -2.8902054, 0.0193052, -2.7353592, 2.7205544
6: 8.6948299, 12.0830297, 8.6571178, 12.1110249, -3.2155895, 3.2231216
7: -18.5952644, -15.0260878, -18.6063347, -15.0107155, -2.7888937, 2.7886143
8: -1.3871531, 1.5634265, -1.4227619, 1.5777035, -2.6472006, 2.6519642
9: -16.0999527, -12.4745865, -16.1397552, -12.4484835, -3.2201939, 3.2677107

Time for backsubstitution: 14.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5717
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 110

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5717

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6364324, upper bound: 1.6364304
time: 6.55 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6364323, upper bound: 1.6441448
time: 7.38 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 28.64 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 28.64
Output dim: 6, lower bound: -1.6356726, upper bound: 1.6308240
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 28.64
Output dim: 6, lower bound: -1.6356726, upper bound: 1.6375475
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 28.64
Output dim: 6, lower bound: -1.6356726, upper bound: 1.6308261
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 28.64
Output dim: 6, lower bound: -1.6356726, upper bound: 1.6375477
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 28.64
Output dim: 6, lower bound: -1.6364324, upper bound: 1.6374187
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 28.64
Output dim: 6, lower bound: -1.6364323, upper bound: 1.6441435
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 28.64
Output dim: 6, lower bound: -1.6364324, upper bound: 1.6374202
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 28.64
Output dim: 6, lower bound: -1.6364323, upper bound: 1.6441448
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 28.64
Output dim: 6, lower bound: -1.6356726, upper bound: 1.6298387
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 28.64
Output dim: 6, lower bound: -1.6356727, upper bound: 1.6375484
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 28.64
Output dim: 6, lower bound: -1.6356726, upper bound: 1.6298362
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 28.64
Output dim: 6, lower bound: -1.6356724, upper bound: 1.6375497
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 28.64
Output dim: 6, lower bound: -1.6364324, upper bound: 1.6364324
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 28.64
Output dim: 6, lower bound: -1.6364323, upper bound: 1.6441429
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 28.64
Output dim: 6, lower bound: -1.6364324, upper bound: 1.6364304
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 28.64
Output dim: 6, lower bound: -1.6364323, upper bound: 1.6441448
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 28.64
Output dim: 6, lower bound: -1.6356713, upper bound: 1.6455569
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.64
Output dim: 6, lower bound: -1.6356713, upper bound: 1.6460670
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.64
Output dim: 6, lower bound: -1.6364310, upper bound: 1.6521074
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.64
Output dim: 6, lower bound: -1.6364311, upper bound: 1.6524872
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 28.64
Output dim: 6, lower bound: -1.6433865, upper bound: 1.6455591
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.64
Output dim: 6, lower bound: -1.6433863, upper bound: 1.6460654
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.64
Output dim: 6, lower bound: -1.6441433, upper bound: 1.6521093
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.64
Output dim: 6, lower bound: -1.6441434, upper bound: 1.6524865
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=3.269695281982422
rel_dist={6: [-1.652505574709414, 1.6525054132086048]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 5717
type: A, layer: 1, pos: 6199
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 110

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 515

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4925074, upper bound: 1.4864952
time: 10.54 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4930087, upper bound: 1.4930079
time: 9.42 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 20.12 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 20.12
Output dim: 6, lower bound: -1.4925074, upper bound: 1.4864952
IS_A2, status: Status.UNKNOWN, split count: 1, time: 20.12
Output dim: 6, lower bound: -1.4930087, upper bound: 1.4930079

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -7.2047043, -3.9969299, -7.2181811, -3.9885249, -3.1271105, 3.1316500
1: -13.8756742, -9.6076956, -13.8873234, -9.6021099, -3.6897182, 3.6925144
2: -7.1464763, -3.7797542, -7.1570153, -3.7700496, -2.7862515, 2.7864270
3: -12.8321466, -9.6640778, -12.8410110, -9.6498318, -2.4882364, 2.4825559
4: -6.9511647, -3.4252973, -6.9581966, -3.4132159, -2.9456291, 2.9361444
5: -2.8621097, -0.0014484, -2.8842545, 0.0078657, -2.6450710, 2.6573133
6: 8.6948242, 12.0830383, 8.6637917, 12.0945911, -3.1213589, 3.1398649
7: -18.5952663, -15.0260735, -18.6040878, -15.0180473, -2.6637254, 2.6634130
8: -1.3871694, 1.5634274, -1.4057965, 1.5748215, -2.5632601, 2.5708294
9: -16.0999680, -12.4745846, -16.1192131, -12.4508085, -3.1417351, 3.1377850

Time for backsubstitution: 14.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5717
type: B, layer: 1, pos: 6199
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 6185
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 110

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5717

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4867899, upper bound: 1.4864893
time: 13.16 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4925035, upper bound: 1.4864895
time: 8.46 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -7.2269239, -3.9833694, -7.2269278, -3.9833648, -3.1503592, 3.1540394
1: -13.8976297, -9.6007652, -13.8976345, -9.6007652, -3.7315083, 3.6961327
2: -7.1633496, -3.7632909, -7.1633525, -3.7632895, -2.8265476, 2.7982001
3: -12.8480988, -9.6415119, -12.8481035, -9.6415091, -2.5048375, 2.5317235
4: -6.9691348, -3.4091814, -6.9691420, -3.4091804, -2.9737320, 2.9527550
5: -2.8902125, 0.0193090, -2.8902159, 0.0193145, -2.6955967, 2.6799431
6: 8.6571150, 12.1110344, 8.6571102, 12.1110401, -3.1755505, 3.1555567
7: -18.6063347, -15.0107040, -18.6063347, -15.0107021, -2.6858711, 2.6842141
8: -1.4227738, 1.5777054, -1.4227815, 1.5777073, -2.5964689, 2.6047368
9: -16.1397629, -12.4484825, -16.1397705, -12.4484825, -3.1467838, 3.1857567

Time for backsubstitution: 14.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5717
type: B, layer: 1, pos: 6199
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 6185
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 110

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5717

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4873329, upper bound: 1.4930044
time: 5.43 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4930049, upper bound: 1.4930041
time: 8.50 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 28.58 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 28.58
Output dim: 6, lower bound: -1.4867899, upper bound: 1.4864893
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 28.58
Output dim: 6, lower bound: -1.4925035, upper bound: 1.4864895
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 28.58
Output dim: 6, lower bound: -1.4873329, upper bound: 1.4930044
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 28.58
Output dim: 6, lower bound: -1.4930049, upper bound: 1.4930041

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -7.2047009, -3.9969311, -7.2181745, -3.9885256, -3.1268053, 3.1106730
1: -13.8756657, -9.6076965, -13.8873148, -9.6021118, -3.6874743, 3.6789370
2: -7.1464763, -3.7797596, -7.1570153, -3.7700615, -2.7603178, 2.7847290
3: -12.8321466, -9.6640759, -12.8410110, -9.6498308, -2.4887981, 2.4824495
4: -6.9511609, -3.4252975, -6.9581909, -3.4132164, -2.9456234, 2.9333978
5: -2.8621058, -0.0014505, -2.8842473, 0.0078661, -2.6413813, 2.6541190
6: 8.6948242, 12.0830364, 8.6637917, 12.0945845, -3.1027861, 3.1398582
7: -18.5952644, -15.0260773, -18.6040859, -15.0180531, -2.6532574, 2.6610255
8: -1.3871648, 1.5634274, -1.4057879, 1.5748219, -2.5611773, 2.5440903
9: -16.0999660, -12.4745827, -16.1192055, -12.4508076, -3.1463928, 3.1369305

Time for backsubstitution: 14.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6199
type: A, layer: 1, pos: 5717
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 110

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6199

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4916976, upper bound: 1.4813309
time: 11.95 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4925013, upper bound: 1.4864862
time: 5.50 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -7.2134628, -3.9838719, -7.2004600, -3.9949543, -3.1187191, 3.1270437
1: -13.8861046, -9.6023245, -13.8742867, -9.6101694, -3.7057514, 3.6718082
2: -7.1624794, -3.7795901, -7.1523190, -3.7931879, -2.7958956, 2.7690463
3: -12.8462696, -9.6452866, -12.8441849, -9.6504517, -2.4935517, 2.5239983
4: -6.9631557, -3.4111156, -6.9580765, -3.4165738, -2.9598513, 2.9387207
5: -2.8822451, 0.0176809, -2.8751926, 0.0102332, -2.6730776, 2.6641951
6: 8.6625366, 12.0998030, 8.6851206, 12.0907745, -3.1505089, 3.1169987
7: -18.6047287, -15.0192623, -18.5978069, -15.0268459, -2.6678696, 2.6633568
8: -1.4054925, 1.5769367, -1.3889239, 1.5655022, -2.5628219, 2.5708175
9: -16.1342087, -12.4498348, -16.1282177, -12.4528685, -3.1344566, 3.1726952

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6199
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 5717
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 110

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6199

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4866206, upper bound: 1.4880411
time: 6.50 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4873308, upper bound: 1.4930014
time: 17.03 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -7.2269182, -3.9833694, -7.2269192, -3.9833663, -3.1500549, 3.1329613
1: -13.8976278, -9.6007662, -13.8976259, -9.6007652, -3.7292671, 3.6825809
2: -7.1633492, -3.7632973, -7.1633520, -3.7633026, -2.8006115, 2.7965078
3: -12.8480988, -9.6415119, -12.8481016, -9.6415119, -2.5053992, 2.5316172
4: -6.9691339, -3.4091816, -6.9691343, -3.4091799, -2.9737291, 2.9500113
5: -2.8902111, 0.0193093, -2.8902092, 0.0193129, -2.6918917, 2.6768217
6: 8.6571150, 12.1110325, 8.6571169, 12.1110353, -3.1569777, 3.1555490
7: -18.6063347, -15.0107069, -18.6063366, -15.0107059, -2.6754394, 2.6818237
8: -1.4227695, 1.5777063, -1.4227746, 1.5777063, -2.5943909, 2.5780287
9: -16.1397629, -12.4484835, -16.1397667, -12.4484835, -3.1514416, 3.1849012

Time for backsubstitution: 14.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6199
type: A, layer: 1, pos: 5717
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 110

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6199

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4923118, upper bound: 1.4880418
time: 11.59 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4930028, upper bound: 1.4880395
time: 8.44 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 34.76 seconds
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 34.76
Output dim: 6, lower bound: -1.4916976, upper bound: 1.4813309
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 34.76
Output dim: 6, lower bound: -1.4925013, upper bound: 1.4864862
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 34.76
Output dim: 6, lower bound: -1.4866206, upper bound: 1.4880411
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 34.76
Output dim: 6, lower bound: -1.4873308, upper bound: 1.4930014
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 34.76
Output dim: 6, lower bound: -1.4923118, upper bound: 1.4880418
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 34.76
Output dim: 6, lower bound: -1.4930028, upper bound: 1.4880395

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -7.1789417, -4.0071845, -7.2059641, -3.9923778, -3.0933962, 3.0689564
1: -13.8528061, -9.6329613, -13.8821688, -9.6170197, -3.5622683, 3.6500788
2: -7.1392078, -3.7904077, -7.1501222, -3.7741590, -2.6548638, 2.7628579
3: -12.8107471, -9.6877422, -12.8359575, -9.6653280, -2.4476557, 2.3956437
4: -6.9474535, -3.4394131, -6.9548473, -3.4200087, -2.9162674, 2.9105158
5: -2.8314984, -0.0231476, -2.8767450, -0.0057740, -2.6562061, 2.6222744
6: 8.7226925, 12.0623512, 8.6688004, 12.0808668, -3.0635691, 3.1149702
7: -18.5852013, -15.0466652, -18.6019287, -15.0292053, -2.6311297, 2.6494284
8: -1.3555510, 1.5521894, -1.3876033, 1.5727563, -2.5526094, 2.5151272
9: -16.0607853, -12.4982004, -16.0971279, -12.4529600, -3.1033955, 3.0903168

Time for backsubstitution: 14.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 6199
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 6185
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 110

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 515

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4856396, upper bound: 1.4813300
time: 5.53 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4856396, upper bound: 1.4813306
time: 7.96 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -7.2046919, -3.9969320, -7.2181692, -3.9885259, -3.1166582, 3.1079779
1: -13.8756638, -9.6077118, -13.8873119, -9.6021233, -3.6906815, 3.6777840
2: -7.1464686, -3.7797627, -7.1570139, -3.7700651, -2.7657719, 2.7727818
3: -12.8321428, -9.6640863, -12.8410091, -9.6498394, -2.4838576, 2.4621820
4: -6.9511609, -3.4253016, -6.9581909, -3.4132185, -2.9550343, 2.9245853
5: -2.8621025, -0.0014534, -2.8842449, 0.0078638, -2.6413670, 2.6294932
6: 8.6948290, 12.0830269, 8.6637974, 12.0945768, -3.1027803, 3.1226711
7: -18.5952644, -15.0260887, -18.6040859, -15.0180569, -2.6533594, 2.6609774
8: -1.3871536, 1.5634274, -1.4057822, 1.5748210, -2.5572243, 2.5440845
9: -16.0999527, -12.4745855, -16.1191998, -12.4508095, -3.1061554, 3.1369219

Time for backsubstitution: 14.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 6199
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 6185
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 110

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 515

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4864873, upper bound: 1.4864861
time: 31.62 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4864873, upper bound: 1.4864880
time: 6.27 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -7.2134500, -3.9838729, -7.2004528, -3.9949536, -3.1092453, 3.1235132
1: -13.8861008, -9.6023426, -13.8742847, -9.6101818, -3.7089634, 3.6706514
2: -7.1624746, -3.7795930, -7.1523123, -3.7931886, -2.8015046, 2.7588153
3: -12.8462667, -9.6452971, -12.8441839, -9.6504583, -2.4884944, 2.5022926
4: -6.9631529, -3.4111242, -6.9580746, -3.4165795, -2.9694319, 2.9299765
5: -2.8822384, 0.0176764, -2.8751907, 0.0102291, -2.6730156, 2.6396532
6: 8.6625366, 12.0997944, 8.6851263, 12.0907679, -3.1505032, 3.0998116
7: -18.6047249, -15.0192738, -18.5978031, -15.0268517, -2.6679707, 2.6633115
8: -1.4054830, 1.5769362, -1.3889170, 1.5655022, -2.5588703, 2.5708094
9: -16.1341972, -12.4498348, -16.1282120, -12.4528704, -3.0942097, 3.1726875

Time for backsubstitution: 14.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 6199
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 6185
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 110

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 515

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4807546, upper bound: 1.4925020
time: 12.16 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4807546, upper bound: 1.4925004
time: 21.12 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -7.2004433, -3.9953947, -7.2145915, -3.9869990, -3.1155882, 3.1026630
1: -13.8709736, -9.6260576, -13.8923340, -9.6156721, -3.6864405, 3.6518002
2: -7.1502786, -3.7745352, -7.1561956, -3.7673867, -2.7844644, 2.7740154
3: -12.8263636, -9.6710196, -12.8430510, -9.6570044, -2.4638581, 2.4978623
4: -6.9617805, -3.4235361, -6.9650583, -3.4159634, -2.9553394, 2.9259405
5: -2.8591943, -0.0044124, -2.8828297, 0.0055766, -2.6425018, 2.6395702
6: 8.6853619, 12.0877190, 8.6620121, 12.0971565, -3.1149635, 3.1269608
7: -18.5955811, -15.0316124, -18.6042519, -15.0218992, -2.6516972, 2.6565270
8: -1.3881197, 1.5664225, -1.4045062, 1.5756812, -2.5569949, 2.5477567
9: -16.0998268, -12.4729815, -16.1176300, -12.4505301, -3.1075211, 3.1368923

Time for backsubstitution: 14.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 6199
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 6185
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 110

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 515

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4856385, upper bound: 1.4873987
time: 5.19 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4856385, upper bound: 1.4880423
time: 12.43 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -7.2269087, -3.9833713, -7.2269139, -3.9833660, -3.1405754, 3.1294346
1: -13.8976231, -9.6007843, -13.8976269, -9.6007767, -3.7324820, 3.6814251
2: -7.1633410, -3.7632995, -7.1633468, -3.7633040, -2.8061686, 2.7863011
3: -12.8480949, -9.6415262, -12.8480988, -9.6415176, -2.5003538, 2.5098896
4: -6.9691329, -3.4091868, -6.9691324, -3.4091849, -2.9832611, 2.9412346
5: -2.8902066, 0.0193040, -2.8902087, 0.0193100, -2.6918736, 2.6521988
6: 8.6571178, 12.1110229, 8.6571178, 12.1110306, -3.1569738, 3.1383619
7: -18.6063347, -15.0107145, -18.6063328, -15.0107136, -2.6755400, 2.6817784
8: -1.4227600, 1.5777044, -1.4227691, 1.5777063, -2.5904379, 2.5780206
9: -16.1397495, -12.4484863, -16.1397591, -12.4484854, -3.1111965, 3.1848927

Time for backsubstitution: 14.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 6199
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 6185
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 110

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 515

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4864864, upper bound: 1.4925001
time: 12.48 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4864865, upper bound: 1.4925015
time: 6.48 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 33.63 seconds
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 33.63
Output dim: 6, lower bound: -1.4856396, upper bound: 1.4813300
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 33.63
Output dim: 6, lower bound: -1.4856396, upper bound: 1.4813306
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 33.63
Output dim: 6, lower bound: -1.4864873, upper bound: 1.4864861
IS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 33.63
Output dim: 6, lower bound: -1.4864873, upper bound: 1.4864880
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 33.63
Output dim: 6, lower bound: -1.4807546, upper bound: 1.4925020
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 33.63
Output dim: 6, lower bound: -1.4807546, upper bound: 1.4925004
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 33.63
Output dim: 6, lower bound: -1.4856385, upper bound: 1.4873987
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 33.63
Output dim: 6, lower bound: -1.4856385, upper bound: 1.4880423
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 33.63
Output dim: 6, lower bound: -1.4864864, upper bound: 1.4925001
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 33.63
Output dim: 6, lower bound: -1.4864865, upper bound: 1.4925015

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -7.2134500, -3.9838729, -7.1783237, -4.0085058, -3.0986366, 3.1016321
1: -13.8861008, -9.6023426, -13.8522873, -9.6170979, -3.6653538, 3.6505966
2: -7.1624746, -3.7795930, -7.1354299, -3.8096912, -2.7519035, 2.7405572
3: -12.8462667, -9.6452971, -12.8281984, -9.6729488, -2.4658570, 2.4526768
4: -6.9631529, -3.4111242, -6.9400978, -3.4326453, -2.9321280, 2.9145288
5: -2.8822384, 0.0176764, -2.8472586, -0.0105250, -2.6453581, 2.6206760
6: 8.6625366, 12.0997944, 8.7228832, 12.0627575, -3.1203146, 3.0834103
7: -18.6047249, -15.0192738, -18.5867119, -15.0422649, -2.6484489, 2.6527209
8: -1.4054830, 1.5769362, -1.3533001, 1.5512381, -2.5526171, 2.5322971
9: -16.1341972, -12.4498348, -16.0883522, -12.4789762, -3.1065273, 3.1312609

Time for backsubstitution: 14.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 5717
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 110

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 803

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5814

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4770613, upper bound: 1.4910594
time: 9.24 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4807420, upper bound: 1.4924886
time: 9.64 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -7.2134500, -3.9838729, -7.2004461, -3.9949560, -3.1092424, 3.1199150
1: -13.8861008, -9.6023426, -13.8742790, -9.6101818, -3.7089577, 3.7059574
2: -7.1624746, -3.7795930, -7.1523113, -3.7931941, -2.8015003, 2.7871509
3: -12.8462667, -9.6452971, -12.8441782, -9.6504612, -2.5153713, 2.5022821
4: -6.9631529, -3.4111242, -6.9580708, -3.4165792, -2.9694242, 2.9510365
5: -2.8822384, 0.0176764, -2.8751879, 0.0102258, -2.6573582, 2.6396484
6: 8.6625366, 12.0997944, 8.6851282, 12.0907621, -3.1305084, 3.0998096
7: -18.6047249, -15.0192738, -18.5978031, -15.0268536, -2.6663122, 2.6633110
8: -1.4054830, 1.5769362, -1.3889070, 1.5654998, -2.5588675, 2.5625420
9: -16.1341972, -12.4498348, -16.1282005, -12.4528713, -3.0942078, 3.1337109

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 5717
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 110

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 803

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5814

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4770613, upper bound: 1.4917901
time: 5.65 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4807421, upper bound: 1.4924896
time: 6.62 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -7.2269087, -3.9833713, -7.2046919, -3.9969306, -3.1300564, 3.1074705
1: -13.8976231, -9.6007843, -13.8756628, -9.6077051, -3.6888275, 3.6614141
2: -7.1633410, -3.7632995, -7.1464701, -3.7797692, -2.7565823, 2.7680378
3: -12.8480949, -9.6415262, -12.8321457, -9.6640854, -2.4776726, 2.4602776
4: -6.9691329, -3.4091868, -6.9511604, -3.4253018, -2.9460125, 2.9257898
5: -2.8902066, 0.0193040, -2.8620999, -0.0014522, -2.6642008, 2.6331902
6: 8.6571178, 12.1110229, 8.6948299, 12.0830269, -3.1267910, 3.1219540
7: -18.6063347, -15.0107145, -18.5952644, -15.0260849, -2.6560597, 2.6711807
8: -1.4227600, 1.5777044, -1.3871541, 1.5634270, -2.5841441, 2.5396204
9: -16.1397495, -12.4484863, -16.0999546, -12.4745874, -3.1235189, 3.1435280

Time for backsubstitution: 14.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5717
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 110

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5717

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4807546, upper bound: 1.4867884
time: 6.29 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4807546, upper bound: 1.4924996
time: 6.35 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -7.2269087, -3.9833713, -7.2269096, -3.9833682, -3.1405716, 3.1257505
1: -13.8976231, -9.6007843, -13.8976183, -9.6007776, -3.7324753, 3.7167988
2: -7.1633410, -3.7632995, -7.1633468, -3.7633078, -2.8061614, 2.8146372
3: -12.8480949, -9.6415262, -12.8480949, -9.6415205, -2.5272264, 2.5098805
4: -6.9691329, -3.4091868, -6.9691296, -3.4091849, -2.9832544, 2.9622135
5: -2.8902066, 0.0193040, -2.8902054, 0.0193057, -2.6762152, 2.6521950
6: 8.6571178, 12.1110229, 8.6571188, 12.1110258, -3.1369772, 3.1383591
7: -18.6063347, -15.0107145, -18.6063309, -15.0107174, -2.6738815, 2.6817760
8: -1.4227600, 1.5777044, -1.4227610, 1.5777054, -2.5904369, 2.5697532
9: -16.1397495, -12.4484863, -16.1397552, -12.4484844, -3.1111965, 3.1459179

Time for backsubstitution: 14.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5717
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 110

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5717

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4807546, upper bound: 1.4873304
time: 14.55 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4807546, upper bound: 1.4930018
time: 8.78 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 38.02 seconds
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 38.02
Output dim: 6, lower bound: -1.4770613, upper bound: 1.4910594
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 38.02
Output dim: 6, lower bound: -1.4807420, upper bound: 1.4924886
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 38.02
Output dim: 6, lower bound: -1.4770613, upper bound: 1.4917901
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 38.02
Output dim: 6, lower bound: -1.4807421, upper bound: 1.4924896
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 38.02
Output dim: 6, lower bound: -1.4807546, upper bound: 1.4867884
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 38.02
Output dim: 6, lower bound: -1.4807546, upper bound: 1.4924996
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 38.02
Output dim: 6, lower bound: -1.4807546, upper bound: 1.4873304
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 38.02
Output dim: 6, lower bound: -1.4807546, upper bound: 1.4930018

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -7.2134304, -3.9838777, -7.1783118, -4.0085082, -3.0785809, 3.1016159
1: -13.8860970, -9.6023540, -13.8522863, -9.6171064, -3.7011814, 3.6402779
2: -7.1624699, -3.7796116, -7.1354284, -3.8097019, -2.7518911, 2.7038345
3: -12.8462601, -9.6453075, -12.8281975, -9.6729565, -2.4814692, 2.4481707
4: -6.9631500, -3.4111600, -6.9400954, -3.4326653, -2.9276576, 2.8677845
5: -2.8822334, 0.0176718, -2.8472564, -0.0105267, -2.6412086, 2.6350670
6: 8.6625519, 12.0997925, 8.7228889, 12.0627556, -3.1173935, 3.0834007
7: -18.6047268, -15.0192833, -18.5867100, -15.0422697, -2.6484447, 2.6335506
8: -1.4054654, 1.5769362, -1.3532896, 1.5512376, -2.5228167, 2.5322886
9: -16.1341915, -12.4498558, -16.0883503, -12.4789848, -3.1020150, 3.0984874

Time for backsubstitution: 14.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 6199
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 6185
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 110

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 803

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6199

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4755770, upper bound: 1.4916879
time: 8.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4755770, upper bound: 1.4924891
time: 9.87 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -7.1647229, -4.0141506, -7.1768007, -4.0005398, -3.0513668, 3.0628643
1: -13.8579082, -9.6326342, -13.8681688, -9.6264172, -3.6523924, 3.6658125
2: -7.1222825, -3.8240309, -7.1440845, -3.8181210, -2.7372761, 2.7360082
3: -12.8176899, -9.6785717, -12.8372536, -9.6682034, -2.4579906, 2.4623809
4: -6.9121618, -3.4886913, -6.9526634, -3.4604101, -2.8768291, 2.8701720
5: -2.8634558, 0.0017653, -2.8663473, 0.0053818, -2.6353588, 2.6093965
6: 8.6997929, 12.0761280, 8.7025776, 12.0871181, -3.0925293, 3.0602665
7: -18.5808983, -15.0499964, -18.5947495, -15.0437450, -2.6253281, 2.6291051
8: -1.3667104, 1.5556269, -1.3690288, 1.5636339, -2.5175972, 2.5211191
9: -16.0959473, -12.4925737, -16.1226234, -12.4770527, -3.0320539, 3.0863924

Time for backsubstitution: 14.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 6199
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 6185
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 110

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 803

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6199

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4729023, upper bound: 1.4909513
time: 6.10 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4729022, upper bound: 1.4909546
time: 7.49 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -7.2134304, -3.9838777, -7.2004395, -3.9949570, -3.0891838, 3.1198978
1: -13.8860970, -9.6023540, -13.8742762, -9.6101875, -3.7395048, 3.6956406
2: -7.1624699, -3.7796116, -7.1523099, -3.7932010, -2.7990761, 2.7504292
3: -12.8462601, -9.6453075, -12.8441772, -9.6504688, -2.5309801, 2.4977775
4: -6.9631500, -3.4111600, -6.9580693, -3.4165981, -2.9612732, 2.9042931
5: -2.8822334, 0.0176718, -2.8751841, 0.0102229, -2.6532078, 2.6540394
6: 8.6625519, 12.0997925, 8.6851330, 12.0907593, -3.1279058, 3.0998001
7: -18.6047268, -15.0192833, -18.5978012, -15.0268602, -2.6663060, 2.6441402
8: -1.4054654, 1.5769362, -1.3888979, 1.5655003, -2.5290685, 2.5625329
9: -16.1341915, -12.4498558, -16.1282005, -12.4528837, -3.0941963, 3.1081524

Time for backsubstitution: 14.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 6199
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 6185
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 110

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 803

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6199

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4764663, upper bound: 1.4923009
time: 6.64 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4764662, upper bound: 1.4929925
time: 7.53 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -7.2269058, -3.9833689, -7.2046919, -3.9969306, -3.1092997, 3.1074696
1: -13.8976202, -9.6007862, -13.8756628, -9.6077051, -3.6775064, 3.6614132
2: -7.1633406, -3.7633085, -7.1464701, -3.7797692, -2.7565804, 2.7437959
3: -12.8480978, -9.6415272, -12.8321457, -9.6640854, -2.4776716, 2.4609451
4: -6.9691296, -3.4091864, -6.9511604, -3.4253018, -2.9432707, 2.9257884
5: -2.8902040, 0.0193038, -2.8620999, -0.0014522, -2.6647663, 2.6331873
6: 8.6571226, 12.1110220, 8.6948299, 12.0830269, -3.1267891, 3.1033869
7: -18.6063309, -15.0107193, -18.5952644, -15.0260849, -2.6560578, 2.6631336
8: -1.4227562, 1.5777035, -1.3871541, 1.5634270, -2.5595512, 2.5396190
9: -16.1397476, -12.4484854, -16.0999546, -12.4745874, -3.1235142, 3.1474710

Time for backsubstitution: 14.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 6199
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 6185
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 110

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 803

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5814

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4792064, upper bound: 1.4888925
time: 22.76 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4807422, upper bound: 1.4924879
time: 5.19 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -7.2269058, -3.9833689, -7.2269096, -3.9833682, -3.1198006, 3.1257496
1: -13.8976202, -9.6007862, -13.8976183, -9.6007776, -3.7211599, 3.7167978
2: -7.1633406, -3.7633085, -7.1633468, -3.7633078, -2.8061585, 2.7903938
3: -12.8480978, -9.6415272, -12.8480949, -9.6415205, -2.5272264, 2.5105491
4: -6.9691296, -3.4091864, -6.9691296, -3.4091849, -2.9805145, 2.9622107
5: -2.8902040, 0.0193038, -2.8902054, 0.0193057, -2.6767941, 2.6521931
6: 8.6571226, 12.1110220, 8.6571188, 12.1110258, -3.1369753, 3.1197920
7: -18.6063309, -15.0107193, -18.6063309, -15.0107174, -2.6738796, 2.6737351
8: -1.4227562, 1.5777035, -1.4227610, 1.5777054, -2.5658059, 2.5697513
9: -16.1397476, -12.4484854, -16.1397552, -12.4484844, -3.1111908, 3.1514301

Time for backsubstitution: 14.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 6199
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 6185
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 110

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 803

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5814

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4801875, upper bound: 1.4896203
time: 22.89 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4814526, upper bound: 1.4929932
time: 6.67 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 50.46 seconds
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 50.46
Output dim: 6, lower bound: -1.4755770, upper bound: 1.4916879
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 50.46
Output dim: 6, lower bound: -1.4755770, upper bound: 1.4924891
IS_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 50.46
Output dim: 6, lower bound: -1.4729023, upper bound: 1.4909513
IS_A2_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 50.46
Output dim: 6, lower bound: -1.4729022, upper bound: 1.4909546
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 50.46
Output dim: 6, lower bound: -1.4764663, upper bound: 1.4923009
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 50.46
Output dim: 6, lower bound: -1.4764662, upper bound: 1.4929925
IS_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 50.46
Output dim: 6, lower bound: -1.4792064, upper bound: 1.4888925
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 50.46
Output dim: 6, lower bound: -1.4807422, upper bound: 1.4924879
IS_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 50.46
Output dim: 6, lower bound: -1.4801875, upper bound: 1.4896203
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 50.46
Output dim: 6, lower bound: -1.4814526, upper bound: 1.4929932

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -7.2133665, -3.9838815, -7.1525941, -4.0187511, -3.0588217, 3.0729685
1: -13.8860950, -9.6023579, -13.8294487, -9.6423664, -3.6703978, 3.5455961
2: -7.1623850, -3.7796154, -7.1281590, -3.8202832, -2.7259359, 2.6238074
3: -12.8462582, -9.6453571, -12.8068094, -9.6966753, -2.4146042, 2.4430189
4: -6.9631476, -3.4112086, -6.9363866, -3.4468036, -2.8970213, 2.8505192
5: -2.8822274, 0.0176194, -2.8167405, -0.0322130, -2.6193333, 2.6890221
6: 8.6625633, 12.0997887, 8.7507477, 12.0420742, -3.0966444, 3.0663838
7: -18.6047020, -15.0192871, -18.5766468, -15.0628862, -2.6335821, 2.6241117
8: -1.4054627, 1.5769181, -1.3216078, 1.5400019, -2.5167246, 2.5188456
9: -16.1341782, -12.4498568, -16.0491409, -12.5026093, -3.0962276, 3.0567026

Time for backsubstitution: 14.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 5717
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 110

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 803

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=3.1755638122558594
rel_dist={6: [-1.493016446904722, 1.493015957466044]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2417.20 seconds
