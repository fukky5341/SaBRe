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
execution time: IAR + LP analysis = 15.17 + 32.67 = 47.84 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3552.16 seconds, max iter: 100)

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
Binary search time: 213.86 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Individual Split (IS_dual) starts
Time budget: 3338.30 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 5717
type: B, layer: 1, pos: 5717
type: B, layer: 1, pos: 6199
type: A, layer: 1, pos: 6199
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 482
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 6185
type: A, layer: 1, pos: 6185
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 4554
type: B, layer: 1, pos: 4554
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 4645
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 110
type: A, layer: 1, pos: 110

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 515

## Relational analysis of IS_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 515

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0829747, upper bound: 2.0967403
time: 14.05 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0967466, upper bound: 2.0967475
time: 7.27 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 31.51 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 31.51
Output dim: 6, lower bound: -2.0829747, upper bound: 2.0967403
IS_B2, status: Status.UNKNOWN, split count: 1, time: 31.51
Output dim: 6, lower bound: -2.0967466, upper bound: 2.0967475

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -7.2225876, -3.9858620, -7.2047043, -3.9969299, -3.2256577, 3.2188423
1: -13.8924026, -9.6014280, -13.8756742, -9.6076956, -4.2309246, 4.2273808
2: -7.1603260, -3.7666197, -7.1464763, -3.7797542, -3.1608381, 3.1615095
3: -12.8446035, -9.6455002, -12.8321466, -9.6640778, -2.9329705, 2.9382730
4: -6.9636564, -3.4111657, -6.9511647, -3.4252973, -3.3471842, 3.3576903
5: -2.8872907, 0.0136454, -2.8621097, -0.0014484, -2.8858423, 2.8757551
6: 8.6603804, 12.1028891, 8.6948242, 12.0830383, -3.4226580, 3.4080648
7: -18.6052322, -15.0143261, -18.5952663, -15.0260735, -3.1636090, 3.1673884
8: -1.4143906, 1.5762925, -1.3871694, 1.5634274, -2.9276600, 2.9118710
9: -16.1296291, -12.4496050, -16.0999680, -12.4745846, -3.5855265, 3.5800323

Time for backsubstitution: 14.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5717
type: B, layer: 1, pos: 5717
type: B, layer: 1, pos: 6199
type: A, layer: 1, pos: 6199
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 482
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 6185
type: A, layer: 1, pos: 6185
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 4554
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 4645
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 110
type: A, layer: 1, pos: 110

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5717

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0829285, upper bound: 2.0836094
time: 7.83 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0829656, upper bound: 2.0967328
time: 10.69 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -7.2269306, -3.9833639, -7.2269239, -3.9833694, -3.2435613, 3.2435601
1: -13.8976383, -9.6007633, -13.8976297, -9.6007652, -4.2376575, 4.2759209
2: -7.1633511, -3.7632847, -7.1633496, -3.7632909, -3.1760488, 3.2047677
3: -12.8481064, -9.6415071, -12.8480988, -9.6415119, -2.9837723, 2.9565306
4: -6.9691429, -3.4091766, -6.9691348, -3.4091814, -3.3627138, 3.3911848
5: -2.8902194, 0.0193186, -2.8902125, 0.0193090, -2.9095285, 2.9095311
6: 8.6571083, 12.1110477, 8.6571150, 12.1110344, -3.4539261, 3.4539328
7: -18.6063347, -15.0106993, -18.6063347, -15.0107040, -3.1831875, 3.1845336
8: -1.4227905, 1.5777082, -1.4227738, 1.5777054, -2.9517937, 2.9450903
9: -16.1397781, -12.4484825, -16.1397629, -12.4484825, -3.6227150, 3.5911579

Time for backsubstitution: 14.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5717
type: B, layer: 1, pos: 5717
type: B, layer: 1, pos: 6199
type: A, layer: 1, pos: 6199
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 482
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 6185
type: B, layer: 1, pos: 6185
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 4554
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5717

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0967027, upper bound: 2.0836906
time: 7.79 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0967374, upper bound: 2.0967375
time: 7.46 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 29.98 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 29.98
Output dim: 6, lower bound: -2.0829285, upper bound: 2.0836094
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 29.98
Output dim: 6, lower bound: -2.0829656, upper bound: 2.0967328
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 29.98
Output dim: 6, lower bound: -2.0967027, upper bound: 2.0836906
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 29.98
Output dim: 6, lower bound: -2.0967374, upper bound: 2.0967375

## BFS IS instance: IS_B1_A1

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

Time for backsubstitution: 15.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6199
type: A, layer: 1, pos: 6199
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 5717
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 482
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 6185
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 4554
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 4645
type: A, layer: 1, pos: 4645
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 110
type: A, layer: 1, pos: 110

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 6199

## Relational analysis of IS_B1_A1_B1

### Relational analysis result of IS_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0712038, upper bound: 2.0827745
time: 7.65 seconds

## Relational analysis of IS_B1_A1_B2

### Relational analysis result of IS_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0829216, upper bound: 2.0836052
time: 8.17 seconds

## BFS IS instance: IS_B1_A2

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

Time for backsubstitution: 15.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6199
type: A, layer: 1, pos: 6199
type: B, layer: 1, pos: 5717
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 482
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 6185
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 110
type: A, layer: 1, pos: 110

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 6199

## Relational analysis of IS_B1_A2_B1

### Relational analysis result of IS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0712453, upper bound: 2.0959124
time: 5.79 seconds

## Relational analysis of IS_B1_A2_B2

### Relational analysis result of IS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0829586, upper bound: 2.0967259
time: 7.84 seconds

## BFS IS instance: IS_B2_A1

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

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6199
type: A, layer: 1, pos: 6199
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 5717
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 482
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 6185
type: B, layer: 1, pos: 6185
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 4554
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 4645
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6199

## Relational analysis of IS_B2_A1_B1

### Relational analysis result of IS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0850995, upper bound: 2.0828863
time: 6.91 seconds

## Relational analysis of IS_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0966974, upper bound: 2.0836862
time: 14.94 seconds

## BFS IS instance: IS_B2_A2

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

Time for backsubstitution: 14.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6199
type: A, layer: 1, pos: 6199
type: B, layer: 1, pos: 5717
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 482
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 6185
type: B, layer: 1, pos: 6185
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 4645
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6199

## Relational analysis of IS_B2_A2_B1

### Relational analysis result of IS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0851324, upper bound: 2.0959388
time: 18.88 seconds

## Relational analysis of IS_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0967321, upper bound: 2.0967341
time: 6.26 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 39.96 seconds
IS_B1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 39.96
Output dim: 6, lower bound: -2.0712038, upper bound: 2.0827745
IS_B1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 39.96
Output dim: 6, lower bound: -2.0829216, upper bound: 2.0836052
IS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 39.96
Output dim: 6, lower bound: -2.0712453, upper bound: 2.0959124
IS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 39.96
Output dim: 6, lower bound: -2.0829586, upper bound: 2.0967259
IS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 39.96
Output dim: 6, lower bound: -2.0850995, upper bound: 2.0828863
IS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 39.96
Output dim: 6, lower bound: -2.0966974, upper bound: 2.0836862
IS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 39.96
Output dim: 6, lower bound: -2.0851324, upper bound: 2.0959388
IS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 39.96
Output dim: 6, lower bound: -2.0967321, upper bound: 2.0967341

## BFS IS instance: IS_B1_A1_B1

### Backsubstitution after applying IS history:
0: -7.1902299, -3.9992306, -7.1721148, -4.0074310, -3.1827989, 3.1728842
1: -13.8665514, -9.6180191, -13.8469219, -9.6337500, -4.1801538, 4.0970125
2: -7.1458216, -3.7985034, -7.1387625, -3.7987132, -3.1193819, 3.0257096
3: -12.8382444, -9.6619301, -12.8098087, -9.6896496, -2.8428636, 2.8941708
4: -6.9506874, -3.4218416, -6.9444051, -3.4403882, -3.3127756, 3.3216448
5: -2.8687208, -0.0020638, -2.8275211, -0.0239601, -2.8447607, 2.8254573
6: 8.6908083, 12.0759459, 8.7254829, 12.0566225, -3.3658142, 3.3504629
7: -18.5956764, -15.0359173, -18.5843716, -15.0510559, -3.1357746, 3.1343122
8: -1.3716896, 1.5631151, -1.3467014, 1.5518050, -2.8748422, 2.8816857
9: -16.1073551, -12.4550028, -16.0579166, -12.4988928, -3.5383167, 3.5270786

Time for backsubstitution: 14.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5717
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 6199
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 6185
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 4554
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 4645
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 110
type: A, layer: 1, pos: 110

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5717

## Relational analysis of IS_B1_A1_B1_B1

### Relational analysis result of IS_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0579829, upper bound: 2.0827742
time: 5.63 seconds

## Relational analysis of IS_B1_A1_B1_B2

### Relational analysis result of IS_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0579828, upper bound: 2.0827751
time: 5.70 seconds

## BFS IS instance: IS_B1_A1_B2

### Backsubstitution after applying IS history:
0: -7.1961541, -3.9974456, -7.1978512, -3.9971850, -3.1989691, 3.2004056
1: -13.8690434, -9.6108313, -13.8697739, -9.6084986, -4.2068634, 4.2126226
2: -7.1492887, -3.7965369, -7.1460252, -3.7880893, -3.1290579, 3.1409345
3: -12.8406754, -9.6544256, -12.8311958, -9.6659784, -2.9124370, 2.9229450
4: -6.9525909, -3.4185443, -6.9481096, -3.4262710, -3.3284655, 3.3595610
5: -2.8722863, 0.0045512, -2.8580945, -0.0022721, -2.8700142, 2.8626456
6: 8.6884203, 12.0826149, 8.6976080, 12.0773001, -3.3888798, 3.3850069
7: -18.5966969, -15.0304966, -18.5944405, -15.0304689, -3.1482439, 3.1501894
8: -1.3805139, 1.5640936, -1.3783293, 1.5630417, -2.8940725, 2.8852363
9: -16.1180496, -12.4539967, -16.0970974, -12.4752731, -3.5729675, 3.5374966

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 5717
type: A, layer: 1, pos: 6199
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 482
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 6185
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 4554
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 4645
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 515

## Relational analysis of IS_B1_A1_B2_A1

### Relational analysis result of IS_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0829217, upper bound: 2.0697588
time: 8.91 seconds

## Relational analysis of IS_B1_A1_B2_A2

### Relational analysis result of IS_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0829216, upper bound: 2.0836068
time: 7.30 seconds

## BFS IS instance: IS_B1_A2_B1

### Backsubstitution after applying IS history:
0: -7.2166409, -3.9876523, -7.1789446, -4.0071812, -3.2094598, 3.1912923
1: -13.8898973, -9.6086187, -13.8528099, -9.6329603, -4.1946554, 4.1134872
2: -7.1568637, -3.7686005, -7.1392078, -3.7904024, -3.1384225, 3.0359321
3: -12.8421736, -9.6530056, -12.8107500, -9.6877422, -2.8486500, 2.9052243
4: -6.9617453, -3.4144542, -6.9474559, -3.4394119, -3.3235292, 3.3324909
5: -2.8837194, 0.0070343, -2.8315008, -0.0231483, -2.8605711, 2.8385351
6: 8.6627626, 12.0962124, 8.7226906, 12.0623531, -3.3995905, 3.3735218
7: -18.6042252, -15.0197411, -18.5852013, -15.0466604, -3.1487999, 3.1449361
8: -1.4055705, 1.5753098, -1.3555539, 1.5521889, -2.8885403, 2.9030738
9: -16.1189384, -12.4506121, -16.0607872, -12.4982014, -3.5508442, 3.5426359

Time for backsubstitution: 14.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5717
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 6199
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 6185
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 6185
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 4554
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 4645
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 110
type: A, layer: 1, pos: 110

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5717

## Relational analysis of IS_B1_A2_B1_B1

### Relational analysis result of IS_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0579702, upper bound: 2.0958750
time: 5.91 seconds

## Relational analysis of IS_B1_A2_B1_B2

### Relational analysis result of IS_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0579700, upper bound: 2.0959124
time: 5.86 seconds

## BFS IS instance: IS_B1_A2_B2

### Backsubstitution after applying IS history:
0: -7.2225761, -3.9858627, -7.2046928, -3.9969311, -3.2256451, 3.2188301
1: -13.8923922, -9.6014347, -13.8756695, -9.6077118, -4.2213707, 4.2291183
2: -7.1603231, -3.7666340, -7.1464672, -3.7797561, -3.1481280, 3.1510210
3: -12.8445997, -9.6455030, -12.8321438, -9.6640854, -2.9182072, 2.9340091
4: -6.9636502, -3.4111674, -6.9511614, -3.4253018, -3.3392076, 3.3703785
5: -2.8872833, 0.0136437, -2.8621035, -0.0014546, -2.8858287, 2.8757472
6: 8.6603851, 12.1028805, 8.6948280, 12.0830307, -3.4226456, 3.4080524
7: -18.6052322, -15.0143356, -18.5952663, -15.0260849, -3.1612005, 3.1608171
8: -1.4143784, 1.5762920, -1.3871570, 1.5634270, -2.9077511, 2.9065909
9: -16.1296196, -12.4496059, -16.0999527, -12.4745846, -3.5854788, 3.5530405

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5717
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 6199
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 482
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 6185
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 4645
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5717

## Relational analysis of IS_B1_A2_B2_B1

### Relational analysis result of IS_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0697562, upper bound: 2.0966927
time: 5.74 seconds

## Relational analysis of IS_B1_A2_B2_B2

### Relational analysis result of IS_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0697561, upper bound: 2.0967257
time: 17.07 seconds

## BFS IS instance: IS_B2_A1_B1

### Backsubstitution after applying IS history:
0: -7.1945047, -3.9966874, -7.1935844, -3.9956496, -3.1988552, 3.1968970
1: -13.8717632, -9.6173611, -13.8651066, -9.6268501, -4.1862335, 4.2232389
2: -7.1488085, -3.7951479, -7.1498327, -3.7828190, -3.1339083, 3.1605535
3: -12.8417587, -9.6579571, -12.8254356, -9.6729441, -2.9475365, 2.9117928
4: -6.9560843, -3.4198694, -6.9587307, -3.4245367, -3.3275270, 3.3647542
5: -2.8716559, 0.0035977, -2.8551610, -0.0052352, -2.8664207, 2.8587587
6: 8.6874847, 12.0840721, 8.6881208, 12.0819979, -3.3945131, 3.3959513
7: -18.5968094, -15.0322838, -18.5947552, -15.0359907, -3.1436481, 3.1500320
8: -1.3800821, 1.5645342, -1.3792951, 1.5660353, -2.8978271, 2.8852797
9: -16.1175079, -12.4538536, -16.0969810, -12.4736691, -3.5742073, 3.5383768

Time for backsubstitution: 14.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 5717
type: A, layer: 1, pos: 6199
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 482
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 6185
type: B, layer: 1, pos: 6185
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 4554
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 4645
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 515

## Relational analysis of IS_B2_A1_B1_A1

### Relational analysis result of IS_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0712038, upper bound: 2.0688837
time: 8.12 seconds

## Relational analysis of IS_B2_A1_B1_A2

### Relational analysis result of IS_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0712038, upper bound: 2.0688856
time: 5.87 seconds

## BFS IS instance: IS_B2_A1_B2

### Backsubstitution after applying IS history:
0: -7.2004614, -3.9949512, -7.2200403, -3.9836249, -3.2168365, 3.2250891
1: -13.8742895, -9.6101751, -13.8917475, -9.6015749, -4.2135868, 4.2611723
2: -7.1523170, -3.7931831, -7.1629004, -3.7716074, -3.1451120, 3.1843472
3: -12.8441877, -9.6504517, -12.8471632, -9.6434422, -2.9617343, 2.9410882
4: -6.9580784, -3.4165738, -6.9660807, -3.4101689, -3.3440065, 3.3931818
5: -2.8751934, 0.0102358, -2.8861444, 0.0184827, -2.8936760, 2.8963802
6: 8.6851215, 12.0907755, 8.6598644, 12.1053019, -3.4201803, 3.4309111
7: -18.5978088, -15.0268459, -18.6055164, -15.0150814, -3.1678381, 3.1673498
8: -1.3889272, 1.5655031, -1.4139483, 1.5773153, -2.9182320, 2.9184847
9: -16.1282234, -12.4528685, -16.1369171, -12.4491711, -3.6101770, 3.5486441

Time for backsubstitution: 14.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 5717
type: A, layer: 1, pos: 6199
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 482
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 6185
type: B, layer: 1, pos: 6185
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 4554
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 4645
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 4645
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 515

## Relational analysis of IS_B2_A1_B2_A1

### Relational analysis result of IS_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0829217, upper bound: 2.0697565
time: 9.25 seconds

## Relational analysis of IS_B2_A1_B2_A2

### Relational analysis result of IS_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0829215, upper bound: 2.0697564
time: 7.91 seconds

## BFS IS instance: IS_B2_A2_B1

### Backsubstitution after applying IS history:
0: -7.2209539, -3.9851046, -7.2004461, -3.9953942, -3.2255597, 3.2153416
1: -13.8951025, -9.6079550, -13.8709784, -9.6260567, -4.2007599, 4.2397299
2: -7.1598473, -3.7652674, -7.1502810, -3.7745266, -3.1529365, 3.1705990
3: -12.8456764, -9.6490145, -12.8263645, -9.6710176, -2.9533119, 2.9228640
4: -6.9671454, -3.4124751, -6.9617844, -3.4235325, -3.3383036, 3.3755488
5: -2.8866656, 0.0126843, -2.8591969, -0.0044103, -2.8822553, 2.8718812
6: 8.6594706, 12.1043377, 8.6853600, 12.0877247, -3.4282541, 3.4189777
7: -18.6053371, -15.0161276, -18.5955811, -15.0316105, -3.1565962, 3.1606598
8: -1.4139490, 1.5767345, -1.3881240, 1.5664244, -2.9114847, 2.9066248
9: -16.1290722, -12.4494610, -16.0998306, -12.4729795, -3.5867138, 3.5539131

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5717
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 6199
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 482
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 4554
type: B, layer: 1, pos: 6185
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 4645
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5717

## Relational analysis of IS_B2_A2_B1_B1

### Relational analysis result of IS_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0720550, upper bound: 2.0959047
time: 7.77 seconds

## Relational analysis of IS_B2_A2_B1_B2

### Relational analysis result of IS_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0720549, upper bound: 2.0959385
time: 7.09 seconds

## BFS IS instance: IS_B2_A2_B2

### Backsubstitution after applying IS history:
0: -7.2269206, -3.9833646, -7.2269120, -3.9833694, -3.2435513, 3.2435474
1: -13.8976326, -9.6007690, -13.8976278, -9.6007833, -4.2281160, 4.2776651
2: -7.1633520, -3.7633004, -7.1633425, -3.7632954, -3.1641607, 3.1943412
3: -12.8481045, -9.6415119, -12.8480949, -9.6415262, -2.9674783, 2.9521618
4: -6.9691386, -3.4091797, -6.9691315, -3.4091866, -3.3547745, 3.4039478
5: -2.8902116, 0.0193160, -2.8902092, 0.0193055, -2.9095170, 2.9095252
6: 8.6571131, 12.1110401, 8.6571169, 12.1110277, -3.4539146, 3.4539232
7: -18.6063366, -15.0107079, -18.6063309, -15.0107126, -3.1807795, 3.1779814
8: -1.4227786, 1.5777068, -1.4227629, 1.5777044, -2.9318833, 2.9398141
9: -16.1397705, -12.4484835, -16.1397514, -12.4484863, -3.6226673, 3.5641603

Time for backsubstitution: 14.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5717
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 6199
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 482
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 6185
type: B, layer: 1, pos: 6185
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5717

## Relational analysis of IS_B2_A2_B2_B1

### Relational analysis result of IS_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0836846, upper bound: 2.0966981
time: 7.68 seconds

## Relational analysis of IS_B2_A2_B2_B2

### Relational analysis result of IS_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0836846, upper bound: 2.0967329
time: 8.35 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 30.76 seconds
IS_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 30.76
Output dim: 6, lower bound: -2.0579829, upper bound: 2.0827742
IS_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 30.76
Output dim: 6, lower bound: -2.0579828, upper bound: 2.0827751
IS_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 30.76
Output dim: 6, lower bound: -2.0829217, upper bound: 2.0697588
IS_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 30.76
Output dim: 6, lower bound: -2.0829216, upper bound: 2.0836068
IS_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 30.76
Output dim: 6, lower bound: -2.0579702, upper bound: 2.0958750
IS_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 30.76
Output dim: 6, lower bound: -2.0579700, upper bound: 2.0959124
IS_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 30.76
Output dim: 6, lower bound: -2.0697562, upper bound: 2.0966927
IS_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 30.76
Output dim: 6, lower bound: -2.0697561, upper bound: 2.0967257
IS_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 30.76
Output dim: 6, lower bound: -2.0712038, upper bound: 2.0688837
IS_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 30.76
Output dim: 6, lower bound: -2.0712038, upper bound: 2.0688856
IS_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 30.76
Output dim: 6, lower bound: -2.0829217, upper bound: 2.0697565
IS_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 30.76
Output dim: 6, lower bound: -2.0829215, upper bound: 2.0697564
IS_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 30.76
Output dim: 6, lower bound: -2.0720550, upper bound: 2.0959047
IS_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 30.76
Output dim: 6, lower bound: -2.0720549, upper bound: 2.0959385
IS_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 30.76
Output dim: 6, lower bound: -2.0836846, upper bound: 2.0966981
IS_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 30.76
Output dim: 6, lower bound: -2.0836846, upper bound: 2.0967329

## BFS IS instance: IS_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -7.1902299, -3.9992306, -7.1526065, -4.0187483, -3.1714816, 3.1533759
1: -13.8665514, -9.6180191, -13.8294525, -9.6423588, -4.1694841, 4.0815620
2: -7.1458216, -3.7985034, -7.1281624, -3.8202734, -3.0985937, 3.0142856
3: -12.8382444, -9.6619301, -12.8068113, -9.6966715, -2.8355412, 2.8913379
4: -6.9506874, -3.4218416, -6.9363890, -3.4467807, -3.3061609, 3.3128924
5: -2.8687208, -0.0020638, -2.8167439, -0.0322106, -2.8365102, 2.8146801
6: 8.6908083, 12.0759459, 8.7507381, 12.0420761, -3.3512678, 3.3252077
7: -18.5956764, -15.0359173, -18.5766468, -15.0628777, -3.1248808, 3.1254597
8: -1.3716896, 1.5631151, -1.3216162, 1.5400038, -2.8624754, 2.8591018
9: -16.1073551, -12.4550028, -16.0491428, -12.5025978, -3.5331717, 3.5192490

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 6199
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 6185
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 4554
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 4645
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 110
type: A, layer: 1, pos: 110

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 515

## Relational analysis of IS_B1_A1_B1_B1_A1

### Relational analysis result of IS_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0579829, upper bound: 2.0688864
time: 5.12 seconds

## Relational analysis of IS_B1_A1_B1_B1_A2

### Relational analysis result of IS_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0579829, upper bound: 2.0827742
time: 5.58 seconds

## BFS IS instance: IS_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -7.1902299, -3.9992306, -7.1789365, -4.0071821, -3.1830478, 3.1797059
1: -13.8665514, -9.6180191, -13.8528023, -9.6329603, -4.1787119, 4.1042385
2: -7.1458216, -3.7985034, -7.1392078, -3.7904155, -3.1281385, 3.0245662
3: -12.8382444, -9.6619301, -12.8107491, -9.6877460, -2.8446269, 2.8950992
4: -6.9506874, -3.4218416, -6.9474525, -3.4394155, -3.3137989, 3.3248410
5: -2.8687208, -0.0020638, -2.8314934, -0.0231476, -2.8455732, 2.8294296
6: 8.6908083, 12.0759459, 8.7226944, 12.0623465, -3.3715382, 3.3532515
7: -18.5956764, -15.0359173, -18.5851974, -15.0466671, -3.1415329, 3.1326418
8: -1.3716896, 1.5631151, -1.3555462, 1.5521870, -2.8731308, 2.8923783
9: -16.1073551, -12.4550028, -16.0607853, -12.4982052, -3.5371552, 3.5313396

Time for backsubstitution: 14.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 6199
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 6185
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 4554
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 4645
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 110
type: A, layer: 1, pos: 110

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 515

## Relational analysis of IS_B1_A1_B1_B2_A1

### Relational analysis result of IS_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0579829, upper bound: 2.0688867
time: 12.77 seconds

## Relational analysis of IS_B1_A1_B1_B2_A2

### Relational analysis result of IS_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0579828, upper bound: 2.0827747
time: 6.26 seconds

## BFS IS instance: IS_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -7.1783285, -4.0085044, -7.1978512, -3.9971850, -3.1811435, 3.1893468
1: -13.8522902, -9.6170921, -13.8697739, -9.6084986, -4.1803808, 4.1896257
2: -7.1354332, -3.8096910, -7.1460252, -3.7880893, -3.1075897, 3.1196518
3: -12.8281994, -9.6729469, -12.8311958, -9.6659784, -2.8911901, 2.8964729
4: -6.9400978, -3.4326441, -6.9481096, -3.4262710, -3.3123283, 3.3329411
5: -2.8472619, -0.0105228, -2.8580945, -0.0022721, -2.8449898, 2.8475716
6: 8.7228823, 12.0627613, 8.6976080, 12.0773001, -3.3544178, 3.3651533
7: -18.5867138, -15.0422583, -18.5944405, -15.0304689, -3.1374292, 3.1355767
8: -1.3533051, 1.5512390, -1.3783293, 1.5630417, -2.8651781, 2.8721728
9: -16.0883560, -12.4789753, -16.0970974, -12.4752731, -3.5420971, 3.5121508

Time for backsubstitution: 15.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5717
type: A, layer: 1, pos: 6199
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 482
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 6185
type: B, layer: 1, pos: 6185
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 4554
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 4645
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 5717

## Relational analysis of IS_B1_A1_B2_A1_B1

### Relational analysis result of IS_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0697687, upper bound: 2.0697588
time: 10.94 seconds

## Relational analysis of IS_B1_A1_B2_A1_B2

### Relational analysis result of IS_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0697689, upper bound: 2.0697595
time: 6.95 seconds

## BFS IS instance: IS_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -7.2004533, -3.9949546, -7.1978512, -3.9971850, -3.2032683, 3.2028966
1: -13.8742809, -9.6101770, -13.8697739, -9.6084986, -4.2004261, 4.1978741
2: -7.1523123, -3.7931917, -7.1460252, -3.7880893, -3.1193094, 3.1357937
3: -12.8441830, -9.6504574, -12.8311958, -9.6659784, -2.9090657, 2.9142666
4: -6.9580717, -3.4165759, -6.9481096, -3.4262710, -3.3277645, 3.3491197
5: -2.8751900, 0.0102272, -2.8580945, -0.0022721, -2.8729179, 2.8683217
6: 8.6851244, 12.0907640, 8.6976080, 12.0773001, -3.3921757, 3.3931561
7: -18.5978031, -15.0268488, -18.5944405, -15.0304689, -3.1496716, 3.1550937
8: -1.3889117, 1.5654993, -1.3783293, 1.5630417, -2.9036837, 2.8866844
9: -16.1282082, -12.4528713, -16.0970974, -12.4752731, -3.5835114, 3.5388069

Time for backsubstitution: 14.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5717
type: A, layer: 1, pos: 6199
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 482
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 6185
type: B, layer: 1, pos: 6185
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 4554
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5717

## Relational analysis of IS_B1_A1_B2_A2_B1

### Relational analysis result of IS_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0697687, upper bound: 2.0836046
time: 9.30 seconds

## Relational analysis of IS_B1_A1_B2_A2_B2

### Relational analysis result of IS_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0697689, upper bound: 2.0836047
time: 9.29 seconds

## BFS IS instance: IS_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -7.2166409, -3.9876523, -7.1526065, -4.0187483, -3.1978927, 3.1649542
1: -13.8898973, -9.6086187, -13.8294525, -9.6423588, -4.1923323, 4.0908012
2: -7.1568637, -3.7686005, -7.1281624, -3.8202734, -3.1088686, 3.0440297
3: -12.8421736, -9.6530056, -12.8068113, -9.6966715, -2.8393579, 2.9005771
4: -6.9617453, -3.4144542, -6.9363890, -3.4467807, -3.3181076, 3.3205395
5: -2.8837194, 0.0070343, -2.8167439, -0.0322106, -2.8515089, 2.8237782
6: 8.6627626, 12.0962124, 8.7507381, 12.0420761, -3.3793135, 3.3454742
7: -18.6042252, -15.0197411, -18.5766468, -15.0628777, -3.1321392, 3.1420794
8: -1.4055705, 1.5753098, -1.3216162, 1.5400038, -2.8957090, 2.8697858
9: -16.1189384, -12.4506121, -16.0491428, -12.5025978, -3.5451860, 3.5232220

Time for backsubstitution: 14.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 6199
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 6185
type: B, layer: 1, pos: 4554
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 4645
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 110
type: A, layer: 1, pos: 110

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 515

## Relational analysis of IS_B1_A2_B1_B1_A1

### Relational analysis result of IS_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0579700, upper bound: 2.0820669
time: 4.94 seconds

## Relational analysis of IS_B1_A2_B1_B1_A2

### Relational analysis result of IS_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0579702, upper bound: 2.0958750
time: 5.90 seconds

## BFS IS instance: IS_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -7.2166409, -3.9876523, -7.1789365, -4.0071821, -3.2094588, 3.1912842
1: -13.8898973, -9.6086187, -13.8528023, -9.6329603, -4.1946497, 4.1066351
2: -7.1568637, -3.7686005, -7.1392078, -3.7904155, -3.1200390, 3.0359302
3: -12.8421736, -9.6530056, -12.8107491, -9.6877460, -2.8493276, 2.9052229
4: -6.9617453, -3.4144542, -6.9474525, -3.4394155, -3.3235254, 3.3302689
5: -2.8837194, 0.0070343, -2.8314934, -0.0231476, -2.8605719, 2.8385277
6: 8.6627626, 12.0962124, 8.7226944, 12.0623465, -3.3995838, 3.3735180
7: -18.6042252, -15.0197411, -18.5851974, -15.0466671, -3.1444931, 3.1449332
8: -1.4055705, 1.5753098, -1.3555462, 1.5521870, -2.8885384, 2.8853035
9: -16.1189384, -12.4506121, -16.0607853, -12.4982052, -3.5564842, 3.5426283

Time for backsubstitution: 14.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 6199
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 6185
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 4554
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 4645
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 110
type: A, layer: 1, pos: 110

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 515

## Relational analysis of IS_B1_A2_B1_B2_A1

### Relational analysis result of IS_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0579701, upper bound: 2.0821089
time: 5.88 seconds

## Relational analysis of IS_B1_A2_B1_B2_A2

### Relational analysis result of IS_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0579700, upper bound: 2.0959124
time: 5.89 seconds

## BFS IS instance: IS_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -7.2225761, -3.9858627, -7.1783190, -4.0085063, -3.2140698, 3.1924562
1: -13.8923922, -9.6014347, -13.8522882, -9.6171055, -4.2190456, 4.2063246
2: -7.1603231, -3.7666340, -7.1354303, -3.8096936, -3.1184845, 3.1591201
3: -12.8445997, -9.6455030, -12.8281956, -9.6729507, -2.9087009, 2.9293551
4: -6.9636502, -3.4111674, -6.9400973, -3.4326475, -3.3338099, 3.3584261
5: -2.8872833, 0.0136437, -2.8472562, -0.0105257, -2.8767576, 2.8608999
6: 8.6603851, 12.1028805, 8.7228842, 12.0627527, -3.4023676, 3.3799963
7: -18.6052322, -15.0143356, -18.5867138, -15.0422668, -3.1445770, 3.1579304
8: -1.4143784, 1.5762920, -1.3532925, 1.5512385, -2.9149170, 2.8733506
9: -16.1296196, -12.4496059, -16.0883484, -12.4789753, -3.5798254, 3.5336790

Time for backsubstitution: 14.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 6199
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 482
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 6185
type: A, layer: 1, pos: 6185
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 4554
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 515

## Relational analysis of IS_B1_A2_B2_B1_A1

### Relational analysis result of IS_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0697561, upper bound: 2.0829249
time: 6.04 seconds

## Relational analysis of IS_B1_A2_B2_B1_A2

### Relational analysis result of IS_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0697560, upper bound: 2.0966925
time: 6.17 seconds

## BFS IS instance: IS_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -7.2225761, -3.9858627, -7.2046881, -3.9969335, -3.2256427, 3.2188253
1: -13.8923922, -9.6014347, -13.8756618, -9.6077156, -4.2213678, 4.2222509
2: -7.1603231, -3.7666340, -7.1464696, -3.7797716, -3.1297445, 3.1510162
3: -12.8445997, -9.6455030, -12.8321409, -9.6640873, -2.9188852, 2.9340067
4: -6.9636502, -3.4111674, -6.9511566, -3.4253027, -3.3392067, 3.3681550
5: -2.8872833, 0.0136437, -2.8620968, -0.0014555, -2.8858278, 2.8757405
6: 8.6603851, 12.1028805, 8.6948318, 12.0830231, -3.4226379, 3.4080486
7: -18.6052322, -15.0143356, -18.5952625, -15.0260887, -3.1569118, 3.1608143
8: -1.4143784, 1.5762920, -1.3871498, 1.5634265, -2.9077487, 2.8888254
9: -16.1296196, -12.4496059, -16.0999508, -12.4745846, -3.5911169, 3.5530329

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 6199
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 482
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 6185
type: A, layer: 1, pos: 6185
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 4554
type: A, layer: 1, pos: 4554
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 515

## Relational analysis of IS_B1_A2_B2_B2_A1

### Relational analysis result of IS_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0697561, upper bound: 2.0829249
time: 7.30 seconds

## Relational analysis of IS_B1_A2_B2_B2_A2

### Relational analysis result of IS_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0697560, upper bound: 2.0967283
time: 18.18 seconds

## BFS IS instance: IS_B2_A1_B1_A1

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

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5717
type: A, layer: 1, pos: 6199
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 482
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 6185
type: B, layer: 1, pos: 6185
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 4554
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 4645
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 110
type: A, layer: 1, pos: 110

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5717

## Relational analysis of IS_B2_A1_B1_A1_B1

### Relational analysis result of IS_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0579829, upper bound: 2.0688839
time: 6.85 seconds

## Relational analysis of IS_B2_A1_B1_A1_B2

### Relational analysis result of IS_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0579829, upper bound: 2.0688837
time: 5.46 seconds

## BFS IS instance: IS_B2_A1_B1_A2

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

Time for backsubstitution: 14.52 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=3.453947067260742
rel_dist={6: [-2.09676497388166, 2.096767128916582]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 5717
type: B, layer: 1, pos: 5717
type: A, layer: 1, pos: 6199
type: B, layer: 1, pos: 6199
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 482
type: A, layer: 1, pos: 6185
type: B, layer: 1, pos: 6185
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 4554
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 4645
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 515

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6521165, upper bound: 1.6441523
time: 6.26 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6524956, upper bound: 1.6524946
time: 4.93 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 11.37 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 11.37
Output dim: 6, lower bound: -1.6521165, upper bound: 1.6441523
IS_A2, status: Status.UNKNOWN, split count: 1, time: 11.37
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

Time for backsubstitution: 14.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5717
type: A, layer: 1, pos: 5717
type: A, layer: 1, pos: 6199
type: B, layer: 1, pos: 6199
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 482
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 6185
type: B, layer: 1, pos: 6185
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 4554
type: B, layer: 1, pos: 4554
type: A, layer: 1, pos: 4645
type: B, layer: 1, pos: 4645
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5717

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6444017, upper bound: 1.6441466
time: 7.77 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6521112, upper bound: 1.6441472
time: 5.72 seconds

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

Time for backsubstitution: 14.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5717
type: A, layer: 1, pos: 5717
type: A, layer: 1, pos: 6199
type: B, layer: 1, pos: 6199
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 6185
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 4554
type: A, layer: 1, pos: 4554
type: B, layer: 1, pos: 4645
type: A, layer: 1, pos: 4645
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 110
type: A, layer: 1, pos: 110

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5717

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6448064, upper bound: 1.6524895
time: 24.50 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6524903, upper bound: 1.6524895
time: 4.99 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 44.15 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 44.15
Output dim: 6, lower bound: -1.6444017, upper bound: 1.6441466
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 44.15
Output dim: 6, lower bound: -1.6521112, upper bound: 1.6441472
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 44.15
Output dim: 6, lower bound: -1.6448064, upper bound: 1.6524895
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 44.15
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

Time for backsubstitution: 14.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6199
type: B, layer: 1, pos: 6199
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 5717
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 482
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 6185
type: B, layer: 1, pos: 6185
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 4554
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 4554
type: A, layer: 1, pos: 4645
type: B, layer: 1, pos: 4645
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6199

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6436737, upper bound: 1.6375477
time: 5.58 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6443992, upper bound: 1.6441425
time: 11.02 seconds

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

Time for backsubstitution: 14.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6199
type: B, layer: 1, pos: 6199
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 5717
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 482
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 6185
type: B, layer: 1, pos: 6185
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 4554
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6199

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6513902, upper bound: 1.6375477
time: 9.84 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6521086, upper bound: 1.6441430
time: 6.29 seconds

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

Time for backsubstitution: 14.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6199
type: B, layer: 1, pos: 6199
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 5717
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 6185
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 4554
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 110
type: A, layer: 1, pos: 110

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6199

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6441558, upper bound: 1.6460646
time: 8.91 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6448039, upper bound: 1.6524867
time: 9.36 seconds

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

Time for backsubstitution: 14.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6199
type: B, layer: 1, pos: 6199
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 5717
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 6185
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 4645
type: B, layer: 1, pos: 4645
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 110
type: A, layer: 1, pos: 110

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6199

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6518485, upper bound: 1.6460650
time: 6.62 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6524878, upper bound: 1.6524863
time: 7.53 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 28.83 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 28.83
Output dim: 6, lower bound: -1.6436737, upper bound: 1.6375477
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 28.83
Output dim: 6, lower bound: -1.6443992, upper bound: 1.6441425
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 28.83
Output dim: 6, lower bound: -1.6513902, upper bound: 1.6375477
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 28.83
Output dim: 6, lower bound: -1.6521086, upper bound: 1.6441430
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 28.83
Output dim: 6, lower bound: -1.6441558, upper bound: 1.6460646
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 28.83
Output dim: 6, lower bound: -1.6448039, upper bound: 1.6524867
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 28.83
Output dim: 6, lower bound: -1.6518485, upper bound: 1.6460650
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 28.83
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

Time for backsubstitution: 14.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 5717
type: B, layer: 1, pos: 482
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 6185
type: B, layer: 1, pos: 6185
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6199
type: A, layer: 1, pos: 4645
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 4554
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 4554
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 803

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 803

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5717

## Relational analysis of IS_A1_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6436738, upper bound: 1.6308260
time: 7.19 seconds

## Relational analysis of IS_A1_B1_A1_A2

### Relational analysis result of IS_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6436738, upper bound: 1.6375476
time: 5.87 seconds

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

Time for backsubstitution: 14.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 5717
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 6199
type: A, layer: 1, pos: 6185
type: B, layer: 1, pos: 6185
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 4645
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 110
type: A, layer: 1, pos: 110

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 803

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 803

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5717

## Relational analysis of IS_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6443990, upper bound: 1.6374183
time: 7.23 seconds

## Relational analysis of IS_A1_B1_A2_A2

### Relational analysis result of IS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6443991, upper bound: 1.6441426
time: 8.38 seconds

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

Time for backsubstitution: 14.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 482
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 6185
type: B, layer: 1, pos: 6185
type: A, layer: 1, pos: 5717
type: A, layer: 1, pos: 4645
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 6199
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 4554
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 4645
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 803

## Relational analysis of IS_A1_B2_A1_A1

### Relational analysis result of IS_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6508642, upper bound: 1.6280293
time: 8.57 seconds

## Relational analysis of IS_A1_B2_A1_A2

### Relational analysis result of IS_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6508642, upper bound: 1.6370293
time: 5.71 seconds

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

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 5717
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 6199
type: A, layer: 1, pos: 6185
type: B, layer: 1, pos: 6185
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 110
type: A, layer: 1, pos: 110

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 803

## Relational analysis of IS_A1_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6515427, upper bound: 1.6345932
time: 7.52 seconds

## Relational analysis of IS_A1_B2_A2_A2

### Relational analysis result of IS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6515426, upper bound: 1.6435915
time: 9.62 seconds

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

Time for backsubstitution: 14.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 5717
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 6199
type: B, layer: 1, pos: 6185
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 4554
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 4645
type: A, layer: 1, pos: 4645
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 110
type: A, layer: 1, pos: 110

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 803

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 803

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5717

## Relational analysis of IS_A2_B1_A1_A1

### Relational analysis result of IS_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6441559, upper bound: 1.6393517
time: 8.65 seconds

## Relational analysis of IS_A2_B1_A1_A2

### Relational analysis result of IS_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6441559, upper bound: 1.6460644
time: 9.90 seconds

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

Time for backsubstitution: 14.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 5717
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 6199
type: B, layer: 1, pos: 6185
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 4554
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 4645
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 110
type: A, layer: 1, pos: 110

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 803

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 803

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5717

## Relational analysis of IS_A2_B1_A2_A1

### Relational analysis result of IS_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6448038, upper bound: 1.6457876
time: 19.19 seconds

## Relational analysis of IS_A2_B1_A2_A2

### Relational analysis result of IS_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6448039, upper bound: 1.6524863
time: 10.21 seconds

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

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 5717
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 6199
type: A, layer: 1, pos: 6185
type: B, layer: 1, pos: 6185
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 803

## Relational analysis of IS_A2_B2_A1_A1

### Relational analysis result of IS_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6512563, upper bound: 1.6365157
time: 13.22 seconds

## Relational analysis of IS_A2_B2_A1_A2

### Relational analysis result of IS_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6512563, upper bound: 1.6370289
time: 7.53 seconds

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

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 5717
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 6199
type: B, layer: 1, pos: 6185
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 4645
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 110
type: A, layer: 1, pos: 110

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 803

## Relational analysis of IS_A2_B2_A2_A1

### Relational analysis result of IS_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6518956, upper bound: 1.6429321
time: 9.58 seconds

## Relational analysis of IS_A2_B2_A2_A2

### Relational analysis result of IS_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6518956, upper bound: 1.6518945
time: 7.04 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 31.33 seconds
IS_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 31.33
Output dim: 6, lower bound: -1.6436738, upper bound: 1.6308260
IS_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 31.33
Output dim: 6, lower bound: -1.6436738, upper bound: 1.6375476
IS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 31.33
Output dim: 6, lower bound: -1.6443990, upper bound: 1.6374183
IS_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 31.33
Output dim: 6, lower bound: -1.6443991, upper bound: 1.6441426
IS_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 31.33
Output dim: 6, lower bound: -1.6508642, upper bound: 1.6280293
IS_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 31.33
Output dim: 6, lower bound: -1.6508642, upper bound: 1.6370293
IS_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 31.33
Output dim: 6, lower bound: -1.6515427, upper bound: 1.6345932
IS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 31.33
Output dim: 6, lower bound: -1.6515426, upper bound: 1.6435915
IS_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 31.33
Output dim: 6, lower bound: -1.6441559, upper bound: 1.6393517
IS_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 31.33
Output dim: 6, lower bound: -1.6441559, upper bound: 1.6460644
IS_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 31.33
Output dim: 6, lower bound: -1.6448038, upper bound: 1.6457876
IS_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 31.33
Output dim: 6, lower bound: -1.6448039, upper bound: 1.6524863
IS_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 31.33
Output dim: 6, lower bound: -1.6512563, upper bound: 1.6365157
IS_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 31.33
Output dim: 6, lower bound: -1.6512563, upper bound: 1.6370289
IS_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 31.33
Output dim: 6, lower bound: -1.6518956, upper bound: 1.6429321
IS_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 31.33
Output dim: 6, lower bound: -1.6518956, upper bound: 1.6518945

## BFS IS instance: IS_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -7.1526065, -4.0187483, -7.1828203, -4.0024843, -3.1317282, 3.1299343
1: -13.8294525, -9.6423588, -13.8611345, -9.6238728, -3.6686258, 3.7645655
2: -7.1281624, -3.8202734, -7.1411190, -3.8023782, -2.7331634, 2.8170657
3: -12.8068113, -9.6966715, -12.8339128, -9.6704731, -2.5488439, 2.4962516
4: -6.9363890, -3.4467807, -6.9457722, -3.4256976, -3.0010290, 2.9969416
5: -2.8167439, -0.0322106, -2.8639059, -0.0110002, -2.7182598, 2.6812954
6: 8.7507381, 12.0420761, 8.6950226, 12.0652466, -3.1336575, 3.1630011
7: -18.5766468, -15.0628777, -18.5940876, -15.0425301, -2.7435122, 2.7504783
8: -1.3216162, 1.5400038, -1.3591542, 1.5613499, -2.5963945, 2.5885496
9: -16.0491428, -12.5025978, -16.0921497, -12.4566326, -3.1906462, 3.1895771

Time for backsubstitution: 14.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 482
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 6185
type: B, layer: 1, pos: 6185
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6199
type: A, layer: 1, pos: 4645
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 4554
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 4554
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 803

## Relational analysis of IS_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 803

## Relational analysis of IS_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 482

## Relational analysis of IS_A1_B1_A1_A1_B1

### Relational analysis result of IS_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6436709, upper bound: 1.6280328
time: 6.46 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2

### Relational analysis result of IS_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6436709, upper bound: 1.6308231
time: 7.29 seconds

## BFS IS instance: IS_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -7.1789365, -4.0071821, -7.1828203, -4.0024843, -3.1578999, 3.1421795
1: -13.8528023, -9.6329603, -13.8611345, -9.6238728, -3.6913023, 3.7737932
2: -7.1392078, -3.7904155, -7.1411190, -3.8023782, -2.7434449, 2.8466105
3: -12.8107491, -9.6877460, -12.8339128, -9.6704731, -2.5526061, 2.5053372
4: -6.9474525, -3.4394155, -6.9457722, -3.4256976, -3.0129776, 3.0045800
5: -2.8314934, -0.0231476, -2.8639059, -0.0110002, -2.7321291, 2.6891499
6: 8.7226944, 12.0623465, 8.6950226, 12.0652466, -3.1609888, 3.1833019
7: -18.5851974, -15.0466671, -18.5940876, -15.0425301, -2.7506933, 2.7671299
8: -1.3555462, 1.5521870, -1.3591542, 1.5613499, -2.6296721, 2.5992055
9: -16.0607853, -12.4982052, -16.0921497, -12.4566326, -3.2027378, 3.1935616

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 482
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 6185
type: B, layer: 1, pos: 6185
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 4645
type: B, layer: 1, pos: 6199
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 4554
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 803

## Relational analysis of IS_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 803

## Relational analysis of IS_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 482

## Relational analysis of IS_A1_B1_A1_A2_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6436709, upper bound: 1.6347563
time: 5.65 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2

### Relational analysis result of IS_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6436709, upper bound: 1.6375449
time: 7.68 seconds

## BFS IS instance: IS_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -7.1783190, -4.0085063, -7.1931095, -3.9992795, -3.1549816, 3.1667776
1: -13.8522882, -9.6171055, -13.8654737, -9.6113052, -3.7933607, 3.7919807
2: -7.1354303, -3.8096936, -7.1470127, -3.7989352, -2.8450971, 2.8268347
3: -12.8281956, -9.6729507, -12.8381691, -9.6573915, -2.5827527, 2.5631843
4: -6.9400973, -3.4326475, -6.9487805, -3.4199622, -3.0394421, 3.0114813
5: -2.8472562, -0.0105257, -2.8701854, 0.0005252, -2.7000132, 2.6879721
6: 8.7228842, 12.0627527, 8.6908026, 12.0768414, -3.1705828, 3.1708527
7: -18.5867138, -15.0422668, -18.5958843, -15.0330925, -2.7638221, 2.7623067
8: -1.3532925, 1.5512385, -1.3745317, 1.5630789, -2.6007576, 2.6145959
9: -16.0883484, -12.4789753, -16.1108017, -12.4548302, -3.1952953, 3.2325983

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 6199
type: A, layer: 1, pos: 6185
type: B, layer: 1, pos: 6185
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 4554
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 4645
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 110
type: A, layer: 1, pos: 110

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 803

## Relational analysis of IS_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 803

## Relational analysis of IS_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 515

## Relational analysis of IS_A1_B1_A2_A1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6364324, upper bound: 1.6374187
time: 6.59 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2

### Relational analysis result of IS_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6364324, upper bound: 1.6374202
time: 6.86 seconds

## BFS IS instance: IS_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -7.2046881, -3.9969335, -7.1931095, -3.9992795, -3.1811991, 3.1790380
1: -13.8756618, -9.6077156, -13.8654737, -9.6113052, -3.8161440, 3.8011971
2: -7.1464696, -3.7797716, -7.1470127, -3.7989352, -2.8553710, 2.8564687
3: -12.8321409, -9.6640873, -12.8381691, -9.6573915, -2.5865216, 2.5724840
4: -6.9511566, -3.4253027, -6.9487805, -3.4199622, -3.0513878, 3.0190964
5: -2.8620968, -0.0014555, -2.8701854, 0.0005252, -2.7140255, 2.6958361
6: 8.6948318, 12.0830231, 8.6908026, 12.0768414, -3.1979084, 3.1911507
7: -18.5952625, -15.0260887, -18.5958843, -15.0330925, -2.7710137, 2.7789235
8: -1.3871498, 1.5634265, -1.3745317, 1.5630789, -2.6339884, 2.6252403
9: -16.0999508, -12.4745846, -16.1108017, -12.4548302, -3.2073383, 3.2365789

Time for backsubstitution: 14.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 6199
type: A, layer: 1, pos: 6185
type: B, layer: 1, pos: 6185
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 4554
type: B, layer: 1, pos: 4645
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 110
type: A, layer: 1, pos: 110

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 803

## Relational analysis of IS_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 803

## Relational analysis of IS_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 515

## Relational analysis of IS_A1_B1_A2_A2_B1

### Relational analysis result of IS_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6364323, upper bound: 1.6441435
time: 6.74 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2

### Relational analysis result of IS_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6364323, upper bound: 1.6441448
time: 7.10 seconds

## BFS IS instance: IS_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -7.1730933, -4.0073500, -7.2062945, -3.9909942, -3.1622944, 3.1480703
1: -13.8491316, -9.6334362, -13.8826694, -9.6147070, -3.6957712, 3.7849894
2: -7.1389689, -3.7961087, -7.1520424, -3.7752855, -2.7462411, 2.8504500
3: -12.8102703, -9.6887341, -12.8376160, -9.6620522, -2.5611734, 2.5077362
4: -6.9458532, -3.4400454, -6.9560404, -3.4186335, -3.0183697, 3.0123425
5: -2.8279850, -0.0237408, -2.8771300, -0.0021901, -2.7348385, 2.7028532
6: 8.7244625, 12.0571861, 8.6678104, 12.0829611, -3.1595421, 3.2047377
7: -18.5847549, -15.0494032, -18.6024227, -15.0276852, -2.7587461, 2.7706523
8: -1.3482547, 1.5520487, -1.3894567, 1.5734682, -2.6310844, 2.6072845
9: -16.0594311, -12.4983740, -16.1030922, -12.4523182, -3.2117844, 3.2058353

Time for backsubstitution: 14.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 482
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 6185
type: B, layer: 1, pos: 6185
type: A, layer: 1, pos: 5717
type: A, layer: 1, pos: 4645
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6199
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 4554
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 4645
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 482

## Relational analysis of IS_A1_B2_A1_A1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6508612, upper bound: 1.6252402
time: 6.49 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2

### Relational analysis result of IS_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6508614, upper bound: 1.6280270
time: 8.67 seconds

## BFS IS instance: IS_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -7.1808052, -4.0001297, -7.2091851, -3.9909108, -3.1661310, 3.1524334
1: -13.8544197, -9.6290188, -13.8844709, -9.6144753, -3.6987410, 3.7821398
2: -7.1443806, -3.7895513, -7.1521549, -3.7724652, -2.7500629, 2.8531094
3: -12.8110018, -9.6866608, -12.8378477, -9.6615639, -2.5618925, 2.5104289
4: -6.9476542, -3.4374175, -6.9568319, -3.4183209, -3.0204315, 3.0165009
5: -2.8321500, -0.0183144, -2.8788807, -0.0018921, -2.7358627, 2.6995902
6: 8.7112570, 12.0628414, 8.6669512, 12.0855093, -3.1753597, 3.2080727
7: -18.5882111, -15.0460396, -18.6026382, -15.0263386, -2.7558928, 2.7717867
8: -1.3578815, 1.5591812, -1.3930531, 1.5735350, -2.6351280, 2.6067977
9: -16.0617695, -12.4977570, -16.1037521, -12.4522352, -3.2139530, 3.2071648

Time for backsubstitution: 14.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 482
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 5717
type: B, layer: 1, pos: 6185
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6199
type: B, layer: 1, pos: 5814
type: B, layer: 1, pos: 4554
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 4554
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 4645
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 482

## Relational analysis of IS_A1_B2_A1_A2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6508614, upper bound: 1.6342378
time: 5.59 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6508614, upper bound: 1.6370263
time: 5.65 seconds

## BFS IS instance: IS_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -7.1988335, -3.9971023, -7.2166071, -3.9877841, -3.1855745, 3.1849899
1: -13.8719845, -9.6081848, -13.8870182, -9.6021461, -3.8205395, 3.8124170
2: -7.1462297, -3.7854791, -7.1579285, -3.7718387, -2.8580546, 2.8602848
3: -12.8316565, -9.6650696, -12.8418713, -9.6489792, -2.5951118, 2.5748205
4: -6.9495602, -3.4259303, -6.9590478, -3.4129059, -3.0567665, 3.0268617
5: -2.8585777, -0.0020528, -2.8834255, 0.0093277, -2.7167349, 2.7095795
6: 8.6965866, 12.0778656, 8.6636066, 12.0945568, -3.1964693, 3.2125893
7: -18.5948219, -15.0288172, -18.6042213, -15.0182686, -2.7790747, 2.7823901
8: -1.3798761, 1.5632858, -1.4048097, 1.5752006, -2.6354094, 2.6333075
9: -16.0986099, -12.4747572, -16.1217270, -12.4505215, -3.2164011, 3.2488308

Time for backsubstitution: 14.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5717
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 6199
type: A, layer: 1, pos: 6185
type: B, layer: 1, pos: 6185
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 110
type: A, layer: 1, pos: 110

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5717

## Relational analysis of IS_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=3.269695281982422
rel_dist={6: [-1.652505574709414, 1.6525054132086048]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 5717
type: A, layer: 1, pos: 5717
type: B, layer: 1, pos: 6199
type: A, layer: 1, pos: 6199
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 6185
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 4645
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 4554
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 515

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4925074, upper bound: 1.4864952
time: 10.27 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4930087, upper bound: 1.4930079
time: 9.13 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 19.58 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 19.58
Output dim: 6, lower bound: -1.4925074, upper bound: 1.4864952
IS_A2, status: Status.UNKNOWN, split count: 1, time: 19.58
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

Time for backsubstitution: 14.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5717
type: A, layer: 1, pos: 5717
type: B, layer: 1, pos: 6199
type: A, layer: 1, pos: 6199
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 482
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 6185
type: B, layer: 1, pos: 6185
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 4554
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 4645
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5717

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4867899, upper bound: 1.4864893
time: 12.82 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4925035, upper bound: 1.4864895
time: 8.27 seconds

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

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5717
type: A, layer: 1, pos: 5717
type: A, layer: 1, pos: 6199
type: B, layer: 1, pos: 6199
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 6185
type: A, layer: 1, pos: 6185
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 4554
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 4645
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 110
type: A, layer: 1, pos: 110

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5717

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4873329, upper bound: 1.4930044
time: 5.34 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4930049, upper bound: 1.4930041
time: 8.28 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 28.33 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 28.33
Output dim: 6, lower bound: -1.4867899, upper bound: 1.4864893
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 28.33
Output dim: 6, lower bound: -1.4925035, upper bound: 1.4864895
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 28.33
Output dim: 6, lower bound: -1.4873329, upper bound: 1.4930044
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 28.33
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

Time for backsubstitution: 14.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6199
type: A, layer: 1, pos: 6199
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 482
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 6185
type: B, layer: 1, pos: 6185
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 5717
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6199

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4874001, upper bound: 1.4856385
time: 7.68 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4925008, upper bound: 1.4864885
time: 8.32 seconds

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

Time for backsubstitution: 14.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6199
type: B, layer: 1, pos: 6199
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 482
type: A, layer: 1, pos: 5717
type: B, layer: 1, pos: 6185
type: A, layer: 1, pos: 6185
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 4554
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 4554
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 110
type: A, layer: 1, pos: 110

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6199

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4866206, upper bound: 1.4880411
time: 6.39 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4873308, upper bound: 1.4930014
time: 16.51 seconds

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

Time for backsubstitution: 14.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6199
type: B, layer: 1, pos: 6199
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 6185
type: A, layer: 1, pos: 6185
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 5717
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 4645
type: B, layer: 1, pos: 4645
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 4554
type: B, layer: 1, pos: 110
type: A, layer: 1, pos: 110

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6199

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4923118, upper bound: 1.4880418
time: 11.24 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4930028, upper bound: 1.4880395
time: 8.19 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 34.13 seconds
IS_A1_B2_B1, status: Status.VERIFIED, split count: 3, time: 34.13
Output dim: 6, lower bound: -1.4874001, upper bound: 1.4856385
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 34.13
Output dim: 6, lower bound: -1.4925008, upper bound: 1.4864885
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 34.13
Output dim: 6, lower bound: -1.4866206, upper bound: 1.4880411
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 34.13
Output dim: 6, lower bound: -1.4873308, upper bound: 1.4930014
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 34.13
Output dim: 6, lower bound: -1.4923118, upper bound: 1.4880418
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 34.13
Output dim: 6, lower bound: -1.4930028, upper bound: 1.4880395

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -7.2046957, -3.9969306, -7.2181625, -3.9885261, -3.1237688, 3.1000710
1: -13.8756657, -9.6077061, -13.8873100, -9.6021290, -3.6863165, 3.6821489
2: -7.1464720, -3.7797623, -7.1570091, -3.7700667, -2.7471323, 2.7902527
3: -12.8321438, -9.6640844, -12.8410082, -9.6498451, -2.4677224, 2.4775028
4: -6.9511604, -3.4253004, -6.9581885, -3.4132237, -2.9367399, 2.9428854
5: -2.8621018, -0.0014520, -2.8842442, 0.0078592, -2.6167555, 2.6541080
6: 8.6948290, 12.0830288, 8.6637974, 12.0945740, -3.0855980, 3.1398544
7: -18.5952682, -15.0260820, -18.6040840, -15.0180645, -2.6532135, 2.6611247
8: -1.3871596, 1.5634284, -1.4057770, 1.5748205, -2.5611701, 2.5401363
9: -16.0999546, -12.4745855, -16.1191959, -12.4508104, -3.1463890, 3.0966816

Time for backsubstitution: 14.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 482
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 6185
type: B, layer: 1, pos: 6185
type: A, layer: 1, pos: 5717
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 4554
type: A, layer: 1, pos: 6199
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 4645
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 803

## Relational analysis of IS_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4919244, upper bound: 1.4792824
time: 8.09 seconds

## Relational analysis of IS_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4919243, upper bound: 1.4859266
time: 5.92 seconds

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

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 482
type: A, layer: 1, pos: 5717
type: B, layer: 1, pos: 6185
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 6199
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 4554
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 4645
type: B, layer: 1, pos: 4554
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 110
type: A, layer: 1, pos: 110

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 803

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 803

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 482

## Relational analysis of IS_A2_B1_A2_A1

### Relational analysis result of IS_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4852485, upper bound: 1.4929991
time: 16.43 seconds

## Relational analysis of IS_A2_B1_A2_A2

### Relational analysis result of IS_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4873287, upper bound: 1.4929994
time: 14.38 seconds

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

Time for backsubstitution: 14.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 6185
type: A, layer: 1, pos: 6185
type: B, layer: 1, pos: 6199
type: A, layer: 1, pos: 5717
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 803

## Relational analysis of IS_A2_B2_A1_A1

### Relational analysis result of IS_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4917280, upper bound: 1.4808302
time: 7.03 seconds

## Relational analysis of IS_A2_B2_A1_A2

### Relational analysis result of IS_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4917281, upper bound: 1.4874705
time: 8.30 seconds

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

Time for backsubstitution: 14.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 6185
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 5717
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 6199
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 4645
type: B, layer: 1, pos: 4645
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 4554
type: B, layer: 1, pos: 110
type: A, layer: 1, pos: 110

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 803

## Relational analysis of IS_A2_B2_A2_A1

### Relational analysis result of IS_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4923760, upper bound: 1.4857753
time: 5.93 seconds

## Relational analysis of IS_A2_B2_A2_A2

### Relational analysis result of IS_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4923760, upper bound: 1.4923756
time: 5.40 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 26.04 seconds
IS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 26.04
Output dim: 6, lower bound: -1.4919244, upper bound: 1.4792824
IS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 26.04
Output dim: 6, lower bound: -1.4919243, upper bound: 1.4859266
IS_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 26.04
Output dim: 6, lower bound: -1.4852485, upper bound: 1.4929991
IS_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 26.04
Output dim: 6, lower bound: -1.4873287, upper bound: 1.4929994
IS_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 26.04
Output dim: 6, lower bound: -1.4917280, upper bound: 1.4808302
IS_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 26.04
Output dim: 6, lower bound: -1.4917281, upper bound: 1.4874705
IS_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 26.04
Output dim: 6, lower bound: -1.4923760, upper bound: 1.4857753
IS_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 26.04
Output dim: 6, lower bound: -1.4923760, upper bound: 1.4923756

## BFS IS instance: IS_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -7.1988382, -3.9971013, -7.2146654, -3.9886289, -3.1158600, 3.0987988
1: -13.8719873, -9.6081781, -13.8851204, -9.6024132, -3.6814547, 3.6806440
2: -7.1462326, -3.7854812, -7.1568689, -3.7734709, -2.7428846, 2.7837853
3: -12.8316574, -9.6650658, -12.8407240, -9.6504307, -2.4661722, 2.4758620
4: -6.9495578, -3.4259274, -6.9572344, -3.4136007, -2.9344492, 2.9411182
5: -2.8585773, -0.0020518, -2.8821282, 0.0074999, -2.6115017, 2.6520400
6: 8.6965857, 12.0778694, 8.6648369, 12.0914974, -3.0810127, 3.1338005
7: -18.5948219, -15.0288143, -18.6038208, -15.0196905, -2.6517410, 2.6573491
8: -1.3798819, 1.5632863, -1.4014411, 1.5747337, -2.5519028, 2.5377522
9: -16.0986137, -12.4747562, -16.1184006, -12.4509125, -3.1448402, 3.0956774

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 482
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 6185
type: B, layer: 1, pos: 6185
type: A, layer: 1, pos: 5717
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 6199
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 4645
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 482

## Relational analysis of IS_A1_B2_B2_A1_B1

### Relational analysis result of IS_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4919223, upper bound: 1.4771950
time: 10.58 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2

### Relational analysis result of IS_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4919223, upper bound: 1.4792802
time: 8.11 seconds

## BFS IS instance: IS_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -7.2065468, -3.9898825, -7.2181501, -3.9885287, -3.1192102, 3.1027336
1: -13.8772755, -9.6037626, -13.8872957, -9.6021290, -3.6841040, 3.6772003
2: -7.1516418, -3.7789116, -7.1570034, -3.7700698, -2.7465420, 2.7859058
3: -12.8323851, -9.6630068, -12.8410015, -9.6498432, -2.4669929, 2.4786701
4: -6.9513588, -3.4232974, -6.9581871, -3.4132254, -2.9365320, 2.9455023
5: -2.8627410, 0.0033748, -2.8842392, 0.0078621, -2.6121101, 2.6483932
6: 8.6833811, 12.0835218, 8.6638012, 12.0945711, -3.0973577, 3.1368809
7: -18.5982780, -15.0254602, -18.6040802, -15.0180655, -2.6486130, 2.6581597
8: -1.3894980, 1.5704174, -1.4057746, 1.5748148, -2.5551620, 2.5371141
9: -16.1009464, -12.4741383, -16.1191978, -12.4508133, -3.1470222, 3.0971489

Time for backsubstitution: 14.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 482
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 6185
type: B, layer: 1, pos: 6185
type: A, layer: 1, pos: 5717
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 6199
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 4645
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 4554
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 482

## Relational analysis of IS_A1_B2_B2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4919222, upper bound: 1.4838433
time: 6.38 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2

### Relational analysis result of IS_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4919222, upper bound: 1.4859245
time: 6.19 seconds

## BFS IS instance: IS_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -7.1987801, -4.0051112, -7.1982222, -4.0064068, -3.0853558, 3.0996208
1: -13.8673229, -9.6302509, -13.8723679, -9.6264420, -3.6726561, 3.6391935
2: -7.1524854, -3.7928758, -7.1465502, -3.7995374, -2.7856297, 2.7387228
3: -12.8326025, -9.6469450, -12.8382301, -9.6512194, -2.4748664, 2.4940543
4: -6.9593873, -3.4178810, -6.9572754, -3.4194598, -2.9590588, 2.9175086
5: -2.8680432, 0.0141559, -2.8690145, 0.0086851, -2.6533318, 2.6254139
6: 8.6778994, 12.0952168, 8.6930122, 12.0903988, -3.1334705, 3.0846481
7: -18.5993919, -15.0304928, -18.5972767, -15.0317421, -2.6534576, 2.6451197
8: -1.3964894, 1.5726004, -1.3850214, 1.5642281, -2.5478544, 2.5615745
9: -16.1181049, -12.4690075, -16.1255970, -12.4635353, -3.0613098, 3.1487951

Time for backsubstitution: 14.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 5717
type: B, layer: 1, pos: 6185
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 6199
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 4554
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 4645
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 4554
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 110
type: A, layer: 1, pos: 110

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 803

## Relational analysis of IS_A2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 803

## Relational analysis of IS_A2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5717

## Relational analysis of IS_A2_B1_A2_A1_A1

### Relational analysis result of IS_A2_B1_A2_A1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4852486, upper bound: 1.4882947
time: 8.22 seconds

## Relational analysis of IS_A2_B1_A2_A1_A2

### Relational analysis result of IS_A2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4852486, upper bound: 1.4929991
time: 12.57 seconds

## BFS IS instance: IS_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -7.2134480, -3.9838786, -7.2004528, -3.9949555, -3.1092386, 3.1069431
1: -13.8861027, -9.6023474, -13.8742809, -9.6101847, -3.7089577, 3.6445589
2: -7.1624727, -3.7795978, -7.1523108, -3.7931921, -2.8018417, 2.7575994
3: -12.8462610, -9.6452999, -12.8441820, -9.6504602, -2.4892139, 2.5014043
4: -6.9631524, -3.4111259, -6.9580736, -3.4165778, -2.9683542, 2.9288573
5: -2.8822355, 0.0176756, -2.8751867, 0.0102293, -2.6730375, 2.6396484
6: 8.6625433, 12.0997944, 8.6851301, 12.0907707, -3.1425934, 3.0977468
7: -18.6047249, -15.0192757, -18.5978050, -15.0268517, -2.6679668, 2.6599059
8: -1.4054813, 1.5769358, -1.3889143, 1.5655012, -2.5579176, 2.5738673
9: -16.1341934, -12.4498425, -16.1282082, -12.4528732, -3.0898046, 3.1544857

Time for backsubstitution: 14.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 5717
type: B, layer: 1, pos: 6185
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 6199
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 5814
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 4554
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 4645
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 4554
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 482
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 110
type: A, layer: 1, pos: 110

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 803

## Relational analysis of IS_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 803

## Relational analysis of IS_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5717

## Relational analysis of IS_A2_B1_A2_A2_A1

### Relational analysis result of IS_A2_B1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4873288, upper bound: 1.4882926
time: 35.71 seconds

## Relational analysis of IS_A2_B1_A2_A2_A2

### Relational analysis result of IS_A2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4873288, upper bound: 1.4929992
time: 13.31 seconds

## BFS IS instance: IS_A2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -7.1945672, -3.9955668, -7.2110882, -3.9871035, -3.1077118, 3.1013975
1: -13.8673086, -9.6265297, -13.8901443, -9.6159515, -3.6815701, 3.6503010
2: -7.1500425, -3.7802238, -7.1560531, -3.7707856, -2.7802358, 2.7675557
3: -12.8258972, -9.6720152, -12.8427687, -9.6575956, -2.4623122, 2.4962139
4: -6.9601798, -3.4241841, -6.9641056, -3.4163465, -2.9530487, 2.9241500
5: -2.8556509, -0.0050147, -2.8807106, 0.0052166, -2.6372328, 2.6374874
6: 8.6871052, 12.0825577, 8.6630440, 12.0940771, -3.1103897, 3.1209135
7: -18.5951366, -15.0343437, -18.6039906, -15.0235252, -2.6502252, 2.6527581
8: -1.3808410, 1.5662789, -1.4001667, 1.5755959, -2.5477409, 2.5453730
9: -16.0984879, -12.4731503, -16.1168365, -12.4506311, -3.1059761, 3.1358862

Time for backsubstitution: 14.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 6185
type: A, layer: 1, pos: 6185
type: B, layer: 1, pos: 6199
type: A, layer: 1, pos: 5717
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 482

## Relational analysis of IS_A2_B2_A1_A1_A1

### Relational analysis result of IS_A2_B2_A1_A1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4896455, upper bound: 1.4808285
time: 10.19 seconds

## Relational analysis of IS_A2_B2_A1_A1_A2

### Relational analysis result of IS_A2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4917259, upper bound: 1.4808282
time: 7.53 seconds

## BFS IS instance: IS_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -7.2022901, -3.9883437, -7.2145786, -3.9869995, -3.1110830, 3.1053257
1: -13.8725939, -9.6221161, -13.8923187, -9.6156702, -3.6842356, 3.6468554
2: -7.1554489, -3.7736650, -7.1561885, -3.7673914, -2.7838745, 2.7696843
3: -12.8266258, -9.6699390, -12.8430462, -9.6570053, -2.4631324, 2.4990339
4: -6.9619856, -3.4215488, -6.9650593, -3.4159670, -2.9551325, 2.9285412
5: -2.8598294, 0.0004091, -2.8828249, 0.0055764, -2.6378536, 2.6338520
6: 8.6738873, 12.0882120, 8.6620178, 12.0971508, -3.1267414, 3.1239891
7: -18.5985947, -15.0309830, -18.6042461, -15.0219030, -2.6470985, 2.6535735
8: -1.3904650, 1.5734119, -1.4044995, 1.5756760, -2.5510159, 2.5447493
9: -16.1008186, -12.4725342, -16.1176262, -12.4505348, -3.1081610, 3.1373596

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 6185
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 5717
type: B, layer: 1, pos: 6199
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 4645
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 4554
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 110
type: A, layer: 1, pos: 110

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 482

## Relational analysis of IS_A2_B2_A1_A2_A1

### Relational analysis result of IS_A2_B2_A1_A2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4896455, upper bound: 1.4874687
time: 12.58 seconds

## Relational analysis of IS_A2_B2_A1_A2_A2

### Relational analysis result of IS_A2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4917260, upper bound: 1.4874688
time: 58.11 seconds

## BFS IS instance: IS_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -7.2210274, -3.9835443, -7.2234044, -3.9834704, -3.1326952, 3.1281614
1: -13.8939524, -9.6012583, -13.8954372, -9.6010580, -3.7276096, 3.6799278
2: -7.1631050, -3.7690058, -7.1632051, -3.7667055, -2.8019681, 2.7798276
3: -12.8476200, -9.6425180, -12.8478184, -9.6421089, -2.4987917, 2.5082459
4: -6.9675303, -3.4098217, -6.9681802, -3.4095633, -2.9809771, 2.9394484
5: -2.8866558, 0.0187013, -2.8880901, 0.0189497, -2.6865826, 2.6501255
6: 8.6588449, 12.1058626, 8.6581430, 12.1079550, -3.1524038, 3.1323156
7: -18.6058960, -15.0134420, -18.6060753, -15.0123377, -2.6740680, 2.6780124
8: -1.4154918, 1.5775595, -1.4184346, 1.5776196, -2.5811968, 2.5756397
9: -16.1384220, -12.4486542, -16.1389694, -12.4485855, -3.1096659, 3.1838942

Time for backsubstitution: 14.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 6185
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 5717
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6199
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 4645
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 4554
type: B, layer: 1, pos: 110
type: A, layer: 1, pos: 110

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 482

## Relational analysis of IS_A2_B2_A2_A1_A1

### Relational analysis result of IS_A2_B2_A2_A1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4902936, upper bound: 1.4857729
time: 26.36 seconds

## Relational analysis of IS_A2_B2_A2_A1_A2

### Relational analysis result of IS_A2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4923739, upper bound: 1.4857732
time: 5.61 seconds

## BFS IS instance: IS_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -7.2287469, -3.9763215, -7.2269006, -3.9833679, -3.1360636, 3.1321030
1: -13.8992386, -9.5968437, -13.8976088, -9.6007748, -3.7302742, 3.6764708
2: -7.1685152, -3.7624364, -7.1633415, -3.7633076, -2.8055754, 2.7819662
3: -12.8483467, -9.6404505, -12.8480949, -9.6415205, -2.4996076, 2.5110588
4: -6.9693336, -3.4071898, -6.9691315, -3.4091866, -2.9830542, 2.9438400
5: -2.8908315, 0.0241265, -2.8902044, 0.0193107, -2.6872034, 2.6464791
6: 8.6456251, 12.1115150, 8.6571198, 12.1110268, -3.1625714, 3.1353893
7: -18.6093502, -15.0100832, -18.6063251, -15.0107155, -2.6709452, 2.6788197
8: -1.4251108, 1.5846930, -1.4227653, 1.5777001, -2.5844550, 2.5749750
9: -16.1407509, -12.4480371, -16.1397591, -12.4484882, -3.1118422, 3.1853638

Time for backsubstitution: 14.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 6185
type: A, layer: 1, pos: 6185
type: A, layer: 1, pos: 5717
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6199
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 4645
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 110
type: A, layer: 1, pos: 110

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 482

## Relational analysis of IS_A2_B2_A2_A2_A1

### Relational analysis result of IS_A2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4902936, upper bound: 1.4923747
time: 6.66 seconds

## Relational analysis of IS_A2_B2_A2_A2_A2

### Relational analysis result of IS_A2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4923739, upper bound: 1.4923722
time: 4.77 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 26.15 seconds
IS_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 26.15
Output dim: 6, lower bound: -1.4919223, upper bound: 1.4771950
IS_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 26.15
Output dim: 6, lower bound: -1.4919223, upper bound: 1.4792802
IS_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 26.15
Output dim: 6, lower bound: -1.4919222, upper bound: 1.4838433
IS_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 26.15
Output dim: 6, lower bound: -1.4919222, upper bound: 1.4859245
IS_A2_B1_A2_A1_A1, status: Status.VERIFIED, split count: 5, time: 26.15
Output dim: 6, lower bound: -1.4852486, upper bound: 1.4882947
IS_A2_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 26.15
Output dim: 6, lower bound: -1.4852486, upper bound: 1.4929991
IS_A2_B1_A2_A2_A1, status: Status.VERIFIED, split count: 5, time: 26.15
Output dim: 6, lower bound: -1.4873288, upper bound: 1.4882926
IS_A2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 26.15
Output dim: 6, lower bound: -1.4873288, upper bound: 1.4929992
IS_A2_B2_A1_A1_A1, status: Status.VERIFIED, split count: 5, time: 26.15
Output dim: 6, lower bound: -1.4896455, upper bound: 1.4808285
IS_A2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 26.15
Output dim: 6, lower bound: -1.4917259, upper bound: 1.4808282
IS_A2_B2_A1_A2_A1, status: Status.VERIFIED, split count: 5, time: 26.15
Output dim: 6, lower bound: -1.4896455, upper bound: 1.4874687
IS_A2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 26.15
Output dim: 6, lower bound: -1.4917260, upper bound: 1.4874688
IS_A2_B2_A2_A1_A1, status: Status.VERIFIED, split count: 5, time: 26.15
Output dim: 6, lower bound: -1.4902936, upper bound: 1.4857729
IS_A2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 26.15
Output dim: 6, lower bound: -1.4923739, upper bound: 1.4857732
IS_A2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 26.15
Output dim: 6, lower bound: -1.4902936, upper bound: 1.4923747
IS_A2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 26.15
Output dim: 6, lower bound: -1.4923739, upper bound: 1.4923722

## BFS IS instance: IS_A1_B2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -7.1966157, -4.0085487, -7.1999536, -4.0098872, -3.0919781, 3.0749712
1: -13.8700933, -9.6244268, -13.8663416, -9.6303177, -3.6500120, 3.6443033
2: -7.1404753, -3.7918067, -7.1468759, -3.7867341, -2.7227879, 2.7679367
3: -12.8256760, -9.6658010, -12.8270378, -9.6520576, -2.4579115, 2.4622083
4: -6.9487772, -3.4288049, -6.9534731, -3.4203262, -2.9220505, 2.9307384
5: -2.8524551, -0.0036011, -2.8680153, 0.0039732, -2.5972834, 2.6324191
6: 8.7044668, 12.0774956, 8.6801710, 12.0869217, -3.0658731, 3.1168070
7: -18.5942936, -15.0337324, -18.5984840, -15.0309563, -2.6335430, 2.6428146
8: -1.3759823, 1.5620060, -1.3924403, 1.5703955, -2.5426664, 2.5266953
9: -16.0960140, -12.4854202, -16.1022949, -12.4700813, -3.1209507, 3.0627451

Time for backsubstitution: 14.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6185
type: B, layer: 1, pos: 6185
type: A, layer: 1, pos: 5717
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 6199
type: B, layer: 1, pos: 4554
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 4645
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 482
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 4645
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 4554
type: A, layer: 1, pos: 110
type: B, layer: 1, pos: 110

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6185

## Relational analysis of IS_A1_B2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4916368, upper bound: 1.4751209
time: 19.76 seconds

## Relational analysis of IS_A1_B2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4919205, upper bound: 1.4771929
time: 7.56 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -7.1988373, -3.9971042, -7.2146645, -3.9886341, -3.0992775, 3.0987930
1: -13.8719845, -9.6081820, -13.8851175, -9.6024189, -3.6553593, 3.6806364
2: -7.1462321, -3.7854829, -7.1568651, -3.7734780, -2.7416468, 2.7841201
3: -12.8316555, -9.6650648, -12.8407183, -9.6504316, -2.4652815, 2.4765577
4: -6.9495597, -3.4259305, -6.9572344, -3.4136024, -2.9333334, 2.9400406
5: -2.8585746, -0.0020516, -2.8821216, 0.0074990, -2.6114931, 2.6520596
6: 8.6965904, 12.0778675, 8.6648417, 12.0915012, -3.0789480, 3.1258917
7: -18.5948219, -15.0288181, -18.6038227, -15.0196934, -2.6483316, 2.6573443
8: -1.3798802, 1.5632858, -1.4014378, 1.5747337, -2.5549588, 2.5368004
9: -16.0986137, -12.4747610, -16.1183968, -12.4509153, -3.1268654, 3.0912800

Time for backsubstitution: 14.54 seconds
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=3.1755638122558594
rel_dist={6: [-1.493016446904722, 1.493015957466044]}

## Binary Search with IS_dual Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2426.17 seconds
