## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 0.8979959898
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-9.0963173, -5.8753343, -9.0963173, -5.8753343, -3.2209830, 3.2209830)
1: (-14.3965416, -11.0441065, -14.3965416, -11.0441065, -3.3503251, 3.3503256)
2: (6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.9031539, 2.9031539)
3: (-5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.6605895, 2.6605895)
4: (-11.1295519, -7.9597182, -11.1295519, -7.9597182, -3.1698337, 3.1698337)
5: (-10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.7234845, 2.7234845)
6: (-13.6051750, -9.5625849, -13.6051750, -9.5625849, -4.0156889, 4.0156889)
7: (-4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.4775901, 2.4775901)
8: (-2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.3622570, 2.3622570)
9: (-9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.8990235, 2.8990231)

## BASE Result
execution time: IAR + LP analysis = 15.30 + 33.19 = 48.50 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3551.50 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.5954461097717285
rel_dist={2: [-1.3415004962338442, 1.3415005921146497]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=2.384212017059326
rel_dist={2: [-0.9012986819155433, 0.9012981480281343]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=2.24338960647583
rel_dist={2: [-0.56420718700224, 0.5642064145019043]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=2.3138012886047363
rel_dist={2: [-0.7378083117025334, 0.7378082346906965]}

## Binary Search Result
Binary search time: 202.31 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Individual Split (IS_dual) starts
Time budget: 3349.19 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4639
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 6184
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 470
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4639

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4790352, upper bound: 1.4751890
time: 4.58 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4790338, upper bound: 1.4790332
time: 4.20 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 8.99 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 8.99
Output dim: 2, lower bound: -1.4790352, upper bound: 1.4751890
IS_A2, status: Status.UNKNOWN, split count: 1, time: 8.99
Output dim: 2, lower bound: -1.4790338, upper bound: 1.4790332

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -9.0940342, -5.8776608, -9.0963173, -5.8753343, -2.8109708, 2.8108101
1: -14.3950214, -11.0544968, -14.3965416, -11.0441065, -2.7538819, 2.7453823
2: 6.4066234, 9.3040161, 6.4048781, 9.3080320, -2.6641221, 2.6618080
3: -5.2118101, -2.5546470, -5.2136140, -2.5530245, -2.6587856, 2.6589670
4: -11.1238737, -7.9606109, -11.1295519, -7.9597182, -2.8259230, 2.8303978
5: -10.7155838, -7.9956589, -10.7184896, -7.9950051, -2.3504505, 2.3539891
6: -13.6023312, -9.5642033, -13.6051750, -9.5625849, -3.3467340, 3.3477883
7: -4.3414259, -1.8684752, -4.3419952, -1.8644050, -2.2839217, 2.2804055
8: -2.1576006, 0.1959448, -2.1592803, 0.2029767, -2.3605773, 2.3552251
9: -9.3674126, -6.3416343, -9.3684855, -6.3403826, -2.4099269, 2.4098961

Time for backsubstitution: 14.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4639
type: A, layer: 1, pos: 6184
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 4639

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4751891, upper bound: 1.4751890
time: 4.34 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4751891, upper bound: 1.4751890
time: 4.50 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -9.1746120, -5.8598013, -9.0963106, -5.8753409, -2.8947659, 2.8308692
1: -14.4910040, -11.0261898, -14.3965378, -11.0441427, -2.8124709, 2.7922797
2: 6.3232183, 9.3151550, 6.4048867, 9.3080187, -2.7533560, 2.6822948
3: -5.2276077, -2.4510708, -5.2136068, -2.5530319, -2.6745758, 2.7625360
4: -11.1550636, -7.9222455, -11.1295328, -7.9597197, -2.8766913, 2.8663430
5: -10.7290831, -7.9215641, -10.7184811, -7.9950094, -2.3804173, 2.4236841
6: -13.7222738, -9.5588531, -13.6051674, -9.5625858, -3.4611697, 3.3556635
7: -4.3688583, -1.8302293, -4.3419943, -1.8644159, -2.3047614, 2.3300364
8: -2.3144202, 0.2082949, -2.1592755, 0.2029600, -2.4810572, 2.3675704
9: -9.3729439, -6.3013988, -9.3684797, -6.3403835, -2.4157043, 2.4593284

Time for backsubstitution: 14.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 6184
type: A, layer: 1, pos: 6184
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 470
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 4639

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4751891, upper bound: 1.4790338
time: 4.51 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4751891, upper bound: 1.4790336
time: 4.47 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 23.80 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 23.80
Output dim: 2, lower bound: -1.4751891, upper bound: 1.4751890
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 23.80
Output dim: 2, lower bound: -1.4751891, upper bound: 1.4751890
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 23.80
Output dim: 2, lower bound: -1.4751891, upper bound: 1.4790338
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 23.80
Output dim: 2, lower bound: -1.4751891, upper bound: 1.4790336

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -9.0940342, -5.8776608, -9.0940342, -5.8776608, -2.8085237, 2.8085232
1: -14.3950214, -11.0544968, -14.3950214, -11.0544968, -2.7436900, 2.7436895
2: 6.4066234, 9.3040161, 6.4066234, 9.3040161, -2.6600723, 2.6600728
3: -5.2118101, -2.5546470, -5.2118101, -2.5546470, -2.6571631, 2.6571631
4: -11.1238737, -7.9606109, -11.1238737, -7.9606109, -2.8249812, 2.8249812
5: -10.7155838, -7.9956589, -10.7155838, -7.9956589, -2.3495750, 2.3495750
6: -13.6023312, -9.5642033, -13.6023312, -9.5642033, -3.3452673, 3.3452678
7: -4.3414259, -1.8684752, -4.3414259, -1.8684752, -2.2798982, 2.2798984
8: -2.1576006, 0.1959448, -2.1576006, 0.1959448, -2.3535454, 2.3535454
9: -9.3674126, -6.3416343, -9.3674126, -6.3416343, -2.4084682, 2.4084675

Time for backsubstitution: 14.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6184
type: B, layer: 1, pos: 6184
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 470
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6184

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4716084, upper bound: 1.4749731
time: 4.04 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4751844, upper bound: 1.4751822
time: 4.11 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -9.0940342, -5.8776608, -9.1746120, -5.8598013, -2.8285875, 2.8918972
1: -14.3950214, -11.0544968, -14.4910040, -11.0261898, -2.7805624, 2.8022671
2: 6.4066234, 9.3040161, 6.3232183, 9.3151550, -2.6769857, 2.7492547
3: -5.2118101, -2.5546470, -5.2276077, -2.4510708, -2.7607393, 2.6729608
4: -11.1238737, -7.9606109, -11.1550636, -7.9222455, -2.8608947, 2.8711259
5: -10.7155838, -7.9956589, -10.7290831, -7.9215641, -2.4192762, 2.3640103
6: -13.6023312, -9.5642033, -13.7222738, -9.5588531, -3.3515968, 3.4596558
7: -4.3414259, -1.8684752, -4.3688583, -1.8302293, -2.3216944, 2.3008094
8: -2.1576006, 0.1959448, -2.3144202, 0.2082949, -2.3658955, 2.4730356
9: -9.3674126, -6.3416343, -9.3729439, -6.3013988, -2.4543915, 2.4142480

Time for backsubstitution: 14.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6184
type: B, layer: 1, pos: 6184
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 470
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6184

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4716084, upper bound: 1.4749733
time: 4.49 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4751844, upper bound: 1.4751821
time: 4.45 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -9.1746120, -5.8598013, -9.0940342, -5.8776608, -2.8918972, 2.8285890
1: -14.4910040, -11.0261898, -14.3950214, -11.0544968, -2.8022671, 2.7805614
2: 6.3232183, 9.3151550, 6.4066234, 9.3040161, -2.7492537, 2.6769857
3: -5.2276077, -2.4510708, -5.2118101, -2.5546470, -2.6729608, 2.7607393
4: -11.1550636, -7.9222455, -11.1238737, -7.9606109, -2.8711257, 2.8608947
5: -10.7290831, -7.9215641, -10.7155838, -7.9956589, -2.3640108, 2.4192762
6: -13.7222738, -9.5588531, -13.6023312, -9.5642033, -3.4596562, 3.3515959
7: -4.3688583, -1.8302293, -4.3414259, -1.8684752, -2.3008094, 2.3216946
8: -2.3144202, 0.2082949, -2.1576006, 0.1959448, -2.4730358, 2.3658955
9: -9.3729439, -6.3013988, -9.3674126, -6.3416343, -2.4142475, 2.4543915

Time for backsubstitution: 14.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6184
type: A, layer: 1, pos: 6184
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 470
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6184

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4749730, upper bound: 1.4754537
time: 4.22 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4751822, upper bound: 1.4790260
time: 4.22 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -9.1746120, -5.8598013, -9.1746120, -5.8598013, -2.9125605, 2.9125600
1: -14.4910040, -11.0261898, -14.4910040, -11.0261898, -2.8386440, 2.8386436
2: 6.3232183, 9.3151550, 6.3232183, 9.3151550, -2.7523584, 2.7523584
3: -5.2276077, -2.4510708, -5.2276077, -2.4510708, -2.7765369, 2.7765369
4: -11.1550636, -7.9222455, -11.1550636, -7.9222455, -2.8993955, 2.8993957
5: -10.7290831, -7.9215641, -10.7290831, -7.9215641, -2.4341111, 2.4341116
6: -13.7222738, -9.5588531, -13.7222738, -9.5588531, -3.4658999, 3.4659004
7: -4.3688583, -1.8302293, -4.3688583, -1.8302293, -2.3429966, 2.3429954
8: -2.3144202, 0.2082949, -2.3144202, 0.2082949, -2.4870820, 2.4870820
9: -9.3729439, -6.3013988, -9.3729439, -6.3013988, -2.4640622, 2.4640622

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6184
type: B, layer: 1, pos: 6184
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 470
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6184

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4716059, upper bound: 1.4788177
time: 4.35 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4751820, upper bound: 1.4790268
time: 4.43 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 23.53 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 23.53
Output dim: 2, lower bound: -1.4716084, upper bound: 1.4749731
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 23.53
Output dim: 2, lower bound: -1.4751844, upper bound: 1.4751822
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 23.53
Output dim: 2, lower bound: -1.4716084, upper bound: 1.4749733
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 23.53
Output dim: 2, lower bound: -1.4751844, upper bound: 1.4751821
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 23.53
Output dim: 2, lower bound: -1.4749730, upper bound: 1.4754537
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 23.53
Output dim: 2, lower bound: -1.4751822, upper bound: 1.4790260
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 23.53
Output dim: 2, lower bound: -1.4716059, upper bound: 1.4788177
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 23.53
Output dim: 2, lower bound: -1.4751820, upper bound: 1.4790268

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -9.0790291, -5.8923688, -9.0901155, -5.8797503, -2.7914586, 2.7848873
1: -14.3744278, -11.0618801, -14.3901968, -11.0558624, -2.7180486, 2.7276106
2: 6.4282508, 9.2934217, 6.4113121, 9.3025360, -2.6341801, 2.6400905
3: -5.1993518, -2.5736549, -5.2084608, -2.5574360, -2.6419158, 2.6348059
4: -11.0901146, -7.9849434, -11.1197834, -7.9675741, -2.7671690, 2.7951260
5: -10.6980047, -8.0034914, -10.7114620, -7.9964523, -2.3305469, 2.3371091
6: -13.5517426, -9.6218510, -13.5999289, -9.5802526, -3.2762833, 3.2814665
7: -4.3116722, -1.9036618, -4.3397017, -1.8781223, -2.2281466, 2.2407427
8: -2.1009564, 0.1413708, -2.1418509, 0.1922970, -2.2932534, 2.2832217
9: -9.3548565, -6.3507552, -9.3647327, -6.3429728, -2.3895254, 2.3903546

Time for backsubstitution: 14.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 6184
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4616

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4716032, upper bound: 1.4677540
time: 4.21 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4716031, upper bound: 1.4749645
time: 3.99 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -9.0940247, -5.8776684, -9.0940332, -5.8776617, -2.8120480, 2.8033018
1: -14.3950043, -11.0544987, -14.3950167, -11.0544958, -2.7398911, 2.7468581
2: 6.4066410, 9.3040104, 6.4066262, 9.3040123, -2.6636400, 2.6540980
3: -5.2118006, -2.5546553, -5.2118082, -2.5546496, -2.6571510, 2.6571529
4: -11.1238613, -7.9606323, -11.1238718, -7.9606147, -2.8147869, 2.8201845
5: -10.7155752, -7.9956598, -10.7155828, -7.9956594, -2.3434849, 2.3495708
6: -13.6023264, -9.5642567, -13.6023312, -9.5642118, -3.3251066, 3.2880554
7: -4.3414202, -1.8685040, -4.3414249, -1.8684797, -2.2708473, 2.2666364
8: -2.1575346, 0.1959357, -2.1575861, 0.1959438, -2.3210869, 2.3529918
9: -9.3674040, -6.3416371, -9.3674097, -6.3416362, -2.4039917, 2.4163442

Time for backsubstitution: 14.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 470
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4616

## Relational analysis of IS_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4679622, upper bound: 1.4751736
time: 4.08 seconds

## Relational analysis of IS_A1_B1_A2_A2

### Relational analysis result of IS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4751733, upper bound: 1.4751736
time: 4.34 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -9.0790291, -5.8923688, -9.1706543, -5.8618903, -2.8115377, 2.8682451
1: -14.3744278, -11.0618801, -14.4861917, -11.0275059, -2.7548990, 2.7861896
2: 6.4282508, 9.2934217, 6.3278675, 9.3136978, -2.6510925, 2.7292051
3: -5.1993518, -2.5736549, -5.2242374, -2.4538844, -2.7454674, 2.6505826
4: -11.0901146, -7.9849434, -11.1509829, -7.9292054, -2.8027301, 2.8412714
5: -10.6980047, -8.0034914, -10.7249517, -7.9223604, -2.4001503, 2.3515363
6: -13.5517426, -9.6218510, -13.7198553, -9.5749016, -3.2826152, 3.3958004
7: -4.3116722, -1.9036618, -4.3671169, -1.8398681, -2.2699046, 2.2615280
8: -2.1009564, 0.1413708, -2.2986314, 0.2046614, -2.3056178, 2.3922646
9: -9.3548565, -6.3507552, -9.3702707, -6.3027482, -2.4354429, 2.3961360

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 6184
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4616

## Relational analysis of IS_A1_B2_A1_A1

### Relational analysis result of IS_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4682614, upper bound: 1.4749584
time: 4.38 seconds

## Relational analysis of IS_A1_B2_A1_A2

### Relational analysis result of IS_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4754459, upper bound: 1.4749587
time: 4.12 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -9.0940247, -5.8776684, -9.1746082, -5.8598046, -2.8321738, 2.8827236
1: -14.3950043, -11.0544987, -14.4910002, -11.0261898, -2.7767639, 2.8025613
2: 6.4066410, 9.3040104, 6.3232212, 9.3151550, -2.6806278, 2.7396321
3: -5.2118006, -2.5546553, -5.2276039, -2.4510722, -2.7607284, 2.6729486
4: -11.1238613, -7.9606323, -11.1550608, -7.9222512, -2.8359613, 2.8664253
5: -10.7155752, -7.9956598, -10.7290792, -7.9215655, -2.4129014, 2.3640065
6: -13.6023264, -9.5642567, -13.7222729, -9.5588636, -3.3313513, 3.4011140
7: -4.3414202, -1.8685040, -4.3688574, -1.8302352, -2.3135185, 2.2782793
8: -2.1575346, 0.1959357, -2.3144054, 0.2082939, -2.3337412, 2.4346588
9: -9.3674040, -6.3416371, -9.3729439, -6.3014021, -2.4499149, 2.4221139

Time for backsubstitution: 14.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 470
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 6184
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4616

## Relational analysis of IS_A1_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4717996, upper bound: 1.4751678
time: 4.53 seconds

## Relational analysis of IS_A1_B2_A2_A2

### Relational analysis result of IS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4790129, upper bound: 1.4751677
time: 4.47 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -9.1706543, -5.8618903, -9.0790291, -5.8923688, -2.8682451, 2.8115373
1: -14.4861917, -11.0275059, -14.3744278, -11.0618801, -2.7861896, 2.7549000
2: 6.3278675, 9.3136978, 6.4282508, 9.2934217, -2.7292051, 2.6510925
3: -5.2242374, -2.4538844, -5.1993518, -2.5736549, -2.6505826, 2.7454674
4: -11.1509829, -7.9292054, -11.0901146, -7.9849434, -2.8412714, 2.8027301
5: -10.7249517, -7.9223604, -10.6980047, -8.0034914, -2.3515368, 2.4001498
6: -13.7198553, -9.5749016, -13.5517426, -9.6218510, -3.3958011, 3.2826152
7: -4.3671169, -1.8398681, -4.3116722, -1.9036618, -2.2615280, 2.2699046
8: -2.2986314, 0.2046614, -2.1009564, 0.1413708, -2.3922644, 2.3056178
9: -9.3702707, -6.3027482, -9.3548565, -6.3507552, -2.3961363, 2.4354429

Time for backsubstitution: 14.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 470
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 6184
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4616

## Relational analysis of IS_A2_B1_B1_B1

### Relational analysis result of IS_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4749586, upper bound: 1.4682613
time: 6.31 seconds

## Relational analysis of IS_A2_B1_B1_B2

### Relational analysis result of IS_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4749586, upper bound: 1.4754460
time: 4.20 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -9.1746082, -5.8598046, -9.0940247, -5.8776684, -2.8827233, 2.8321738
1: -14.4910002, -11.0261898, -14.3950043, -11.0544987, -2.8025613, 2.7767644
2: 6.3232212, 9.3151550, 6.4066410, 9.3040104, -2.7396317, 2.6806283
3: -5.2276039, -2.4510722, -5.2118006, -2.5546553, -2.6729486, 2.7607284
4: -11.1550608, -7.9222512, -11.1238613, -7.9606323, -2.8664255, 2.8359613
5: -10.7290792, -7.9215655, -10.7155752, -7.9956598, -2.3640065, 2.4129009
6: -13.7222729, -9.5588636, -13.6023264, -9.5642567, -3.4011140, 3.3313503
7: -4.3688574, -1.8302352, -4.3414202, -1.8685040, -2.2782793, 2.3135188
8: -2.3144054, 0.2082939, -2.1575346, 0.1959357, -2.4346585, 2.3337412
9: -9.3729439, -6.3014021, -9.3674040, -6.3416371, -2.4221144, 2.4499149

Time for backsubstitution: 14.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 470
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 6184
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4616

## Relational analysis of IS_A2_B1_B2_B1

### Relational analysis result of IS_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4751677, upper bound: 1.4717996
time: 4.38 seconds

## Relational analysis of IS_A2_B1_B2_B2

### Relational analysis result of IS_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4751678, upper bound: 1.4790128
time: 4.17 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -9.1595240, -5.8745661, -9.1706543, -5.8618903, -2.8955050, 2.8889322
1: -14.4704914, -11.0335121, -14.4861917, -11.0275059, -2.8132491, 2.8225150
2: 6.3447671, 9.3046398, 6.3278675, 9.3136978, -2.7261696, 2.7322397
3: -5.2150340, -2.4702408, -5.2242374, -2.4538844, -2.7611496, 2.7539966
4: -11.1213703, -7.9465780, -11.1509829, -7.9292054, -2.8415761, 2.8695350
5: -10.7114553, -7.9293756, -10.7249517, -7.9223604, -2.4150624, 2.4214029
6: -13.6717319, -9.6164742, -13.7198553, -9.5749016, -3.3868752, 3.4020648
7: -4.3390226, -1.8654317, -4.3671169, -1.8398681, -2.2910643, 2.3035083
8: -2.2577052, 0.1537914, -2.2986314, 0.2046614, -2.4269385, 2.4063256
9: -9.3604164, -6.3105922, -9.3702707, -6.3027482, -2.4451265, 2.4459035

Time for backsubstitution: 14.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 6184
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4616

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4715971, upper bound: 1.4715902
time: 3.98 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4715970, upper bound: 1.4788027
time: 4.12 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9.1746006, -5.8598084, -9.1746082, -5.8598046, -2.9111757, 2.9033871
1: -14.4909878, -11.0261936, -14.4910002, -11.0261898, -2.8346949, 2.8398623
2: 6.3232346, 9.3151512, 6.3232212, 9.3151550, -2.7559991, 2.7463303
3: -5.2275963, -2.4510787, -5.2276039, -2.4510722, -2.7765241, 2.7765253
4: -11.1550512, -7.9222689, -11.1550608, -7.9222512, -2.8821921, 2.8927259
5: -10.7290764, -7.9215646, -10.7290792, -7.9215655, -2.4280224, 2.4306870
6: -13.7222681, -9.5589075, -13.7222729, -9.5588636, -3.4278383, 3.4073577
7: -4.3688531, -1.8302599, -4.3688574, -1.8302352, -2.3177748, 2.3215787
8: -2.3143556, 0.2082834, -2.3144054, 0.2082939, -2.4277802, 2.4487052
9: -9.3729362, -6.3014021, -9.3729439, -6.3014021, -2.4595747, 2.4719267

Time for backsubstitution: 14.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 470
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4616

## Relational analysis of IS_A2_B2_A2_A1

### Relational analysis result of IS_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4679563, upper bound: 1.4790124
time: 4.37 seconds

## Relational analysis of IS_A2_B2_A2_A2

### Relational analysis result of IS_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4751674, upper bound: 1.4790119
time: 4.41 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 23.56 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.56
Output dim: 2, lower bound: -1.4716032, upper bound: 1.4677540
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.56
Output dim: 2, lower bound: -1.4716031, upper bound: 1.4749645
IS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 23.56
Output dim: 2, lower bound: -1.4679622, upper bound: 1.4751736
IS_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 23.56
Output dim: 2, lower bound: -1.4751733, upper bound: 1.4751736
IS_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 23.56
Output dim: 2, lower bound: -1.4682614, upper bound: 1.4749584
IS_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 23.56
Output dim: 2, lower bound: -1.4754459, upper bound: 1.4749587
IS_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 23.56
Output dim: 2, lower bound: -1.4717996, upper bound: 1.4751678
IS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 23.56
Output dim: 2, lower bound: -1.4790129, upper bound: 1.4751677
IS_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 23.56
Output dim: 2, lower bound: -1.4749586, upper bound: 1.4682613
IS_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 23.56
Output dim: 2, lower bound: -1.4749586, upper bound: 1.4754460
IS_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 23.56
Output dim: 2, lower bound: -1.4751677, upper bound: 1.4717996
IS_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 23.56
Output dim: 2, lower bound: -1.4751678, upper bound: 1.4790128
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.56
Output dim: 2, lower bound: -1.4715971, upper bound: 1.4715902
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.56
Output dim: 2, lower bound: -1.4715970, upper bound: 1.4788027
IS_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 23.56
Output dim: 2, lower bound: -1.4679563, upper bound: 1.4790124
IS_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 23.56
Output dim: 2, lower bound: -1.4751674, upper bound: 1.4790119

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -9.0787601, -5.8966846, -9.0873947, -5.9023066, -2.7685337, 2.7778091
1: -14.3676062, -11.0623188, -14.3551044, -11.0621185, -2.7045150, 2.6909175
2: 6.4337678, 9.2930622, 6.4402657, 9.2972298, -2.6231246, 2.6109085
3: -5.1986008, -2.5766153, -5.2024417, -2.5731673, -2.6254334, 2.6258264
4: -11.0878315, -7.9857807, -11.1059017, -7.9721966, -2.7588043, 2.7718830
5: -10.6961212, -8.0038128, -10.7012825, -7.9990125, -2.3260159, 2.3260550
6: -13.5506544, -9.6324825, -13.5857506, -9.6356678, -3.2179441, 3.2492578
7: -4.3104210, -1.9044650, -4.3319731, -1.8832761, -2.2176619, 2.2283430
8: -2.0894680, 0.1402907, -2.0820923, 0.1779847, -2.2674527, 2.2223830
9: -9.3544006, -6.3576403, -9.3566113, -6.3785028, -2.3537645, 2.3715525

Time for backsubstitution: 14.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 6184
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 470

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4693765, upper bound: 1.4672800
time: 3.96 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4716013, upper bound: 1.4677518
time: 4.22 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -9.0790291, -5.8923688, -9.0901146, -5.8797565, -2.7903013, 2.7848868
1: -14.3744278, -11.0618801, -14.3901939, -11.0558643, -2.7180467, 2.7146235
2: 6.4282508, 9.2934217, 6.4113178, 9.3025351, -2.6341786, 2.6251111
3: -5.1993518, -2.5736549, -5.2084579, -2.5574386, -2.6419132, 2.6348031
4: -11.0901146, -7.9849434, -11.1197805, -7.9675760, -2.7948008, 2.7920427
5: -10.6980047, -8.0034914, -10.7114553, -7.9964533, -2.3305473, 2.3333774
6: -13.5517426, -9.6218510, -13.5999260, -9.5802660, -3.2388439, 3.2814646
7: -4.3116722, -1.9036618, -4.3396993, -1.8781228, -2.2534375, 2.2379458
8: -2.1009564, 0.1413708, -2.1418326, 0.1922956, -2.2932520, 2.2624843
9: -9.3548565, -6.3507552, -9.3647356, -6.3429770, -2.3754206, 2.3892531

Time for backsubstitution: 14.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 6184
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 470

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4693763, upper bound: 1.4744769
time: 4.04 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4716011, upper bound: 1.4749627
time: 4.22 seconds

## BFS IS instance: IS_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -9.0912952, -5.9002075, -9.0937529, -5.8819523, -2.8049622, 2.7804728
1: -14.3597765, -11.0607643, -14.3882351, -11.0549440, -2.7036781, 2.7332883
2: 6.4356031, 9.2987013, 6.4121456, 9.3036518, -2.6344938, 2.6431317
3: -5.2057519, -2.5703812, -5.2110291, -2.5575166, -2.6482353, 2.6406479
4: -11.1100311, -7.9652405, -11.1216097, -7.9614439, -2.7918382, 2.8118360
5: -10.7053547, -7.9982185, -10.7136669, -7.9959784, -2.3323908, 2.3450108
6: -13.5881834, -9.6196985, -13.6012745, -9.5748577, -3.2749100, 3.2297096
7: -4.3337994, -1.8736566, -4.3402414, -1.8692874, -2.2585154, 2.2562547
8: -2.0977821, 0.1817436, -2.1460900, 0.1949635, -2.2591944, 2.3008740
9: -9.3592720, -6.3771706, -9.3669367, -6.3485107, -2.3851671, 2.3805745

Time for backsubstitution: 14.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 470
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 470

## Relational analysis of IS_A1_B1_A2_A1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4674882, upper bound: 1.4729482
time: 4.30 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2

### Relational analysis result of IS_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4679604, upper bound: 1.4751719
time: 4.43 seconds

## BFS IS instance: IS_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -9.0940237, -5.8776746, -9.0940332, -5.8776617, -2.8120470, 2.8021450
1: -14.3949976, -11.0545006, -14.3950167, -11.0544958, -2.7269158, 2.7468567
2: 6.4066491, 9.3040085, 6.4066262, 9.3040123, -2.6486497, 2.6540976
3: -5.2118006, -2.5546591, -5.2118082, -2.5546496, -2.6571510, 2.6571491
4: -11.1238585, -7.9606347, -11.1238718, -7.9606147, -2.8116951, 2.8477559
5: -10.7155714, -7.9956598, -10.7155828, -7.9956594, -2.3397551, 2.3495703
6: -13.6023254, -9.5642681, -13.6023312, -9.5642118, -3.3171368, 3.2506170
7: -4.3414192, -1.8685054, -4.3414249, -1.8684797, -2.2676578, 2.2919283
8: -2.1575167, 0.1959338, -2.1575861, 0.1959438, -2.2779465, 2.3444426
9: -9.3674049, -6.3416443, -9.3674097, -6.3416362, -2.4028912, 2.4020576

Time for backsubstitution: 14.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 470
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 470

## Relational analysis of IS_A1_B1_A2_A2_B1

### Relational analysis result of IS_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4746881, upper bound: 1.4729481
time: 4.56 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2

### Relational analysis result of IS_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4751716, upper bound: 1.4751719
time: 4.21 seconds

## BFS IS instance: IS_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -9.0763741, -5.9151163, -9.1703691, -5.8661690, -2.8044186, 2.8450766
1: -14.3388357, -11.0680933, -14.4793348, -11.0279322, -2.7180037, 2.7607489
2: 6.4572568, 9.2881012, 6.3333769, 9.3133507, -2.6216335, 2.7082524
3: -5.1934748, -2.5897779, -5.2234802, -2.4567430, -2.7367318, 2.6337023
4: -11.0762119, -7.9895926, -11.1487207, -7.9300370, -2.7795000, 2.8329084
5: -10.6879616, -8.0060501, -10.7230453, -7.9226837, -2.3892980, 2.3469687
6: -13.5373802, -9.6772423, -13.7187786, -9.5855350, -3.2400169, 3.3375230
7: -4.3037243, -1.9087962, -4.3658895, -1.8406609, -2.2574677, 2.2511654
8: -2.0411971, 0.1265268, -2.2870371, 0.2036595, -2.2448566, 2.3396871
9: -9.3468189, -6.3863430, -9.3697939, -6.3096251, -2.4166775, 2.3602891

Time for backsubstitution: 14.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 470

## Relational analysis of IS_A1_B2_A1_A1_A1

### Relational analysis result of IS_A1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4660705, upper bound: 1.4744709
time: 4.19 seconds

## Relational analysis of IS_A1_B2_A1_A1_A2

### Relational analysis result of IS_A1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4682595, upper bound: 1.4749566
time: 4.47 seconds

## BFS IS instance: IS_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -9.0790291, -5.8923759, -9.1706543, -5.8618903, -2.8115368, 2.8670883
1: -14.3744221, -11.0618811, -14.4861917, -11.0275059, -2.7419262, 2.7824359
2: 6.4282541, 9.2934217, 6.3278675, 9.3136978, -2.6361299, 2.7260513
3: -5.1993499, -2.5736589, -5.2242374, -2.4538844, -2.7454655, 2.6505785
4: -11.0901136, -7.9849477, -11.1509829, -7.9292054, -2.7990212, 2.8690796
5: -10.6979990, -8.0034924, -10.7249517, -7.9223604, -2.3964176, 2.3515363
6: -13.5517406, -9.6218596, -13.7198553, -9.5749016, -3.2823067, 3.3583128
7: -4.3116698, -1.9036620, -4.3671169, -1.8398681, -2.2670999, 2.2829440
8: -2.1009421, 0.1413703, -2.2986314, 0.2046614, -2.2869778, 2.3836932
9: -9.3548555, -6.3507590, -9.3702707, -6.3027482, -2.4343243, 2.3819466

Time for backsubstitution: 14.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 470
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 470

## Relational analysis of IS_A1_B2_A1_A2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4749551, upper bound: 1.4727288
time: 3.86 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4754441, upper bound: 1.4749567
time: 3.97 seconds

## BFS IS instance: IS_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -9.0912952, -5.9002075, -9.1743164, -5.8640747, -2.8250937, 2.8596869
1: -14.3597765, -11.0607643, -14.4841557, -11.0266209, -2.7405691, 2.7769041
2: 6.4356031, 9.2987013, 6.3287354, 9.3148088, -2.6514878, 2.7186427
3: -5.2057519, -2.5703812, -5.2268429, -2.4539256, -2.7518263, 2.6564617
4: -11.1100311, -7.9652405, -11.1528101, -7.9230790, -2.8128562, 2.8580935
5: -10.7053547, -7.9982185, -10.7271643, -7.9218860, -2.4018545, 2.3594480
6: -13.5881834, -9.6196985, -13.7212019, -9.5695076, -3.2811971, 3.3427100
7: -4.3337994, -1.8736566, -4.3676500, -1.8310283, -2.3011823, 2.2678659
8: -2.0977821, 0.1817436, -2.3028083, 0.2073131, -2.2718449, 2.3823638
9: -9.3592720, -6.3771706, -9.3724651, -6.3082767, -2.4310937, 2.3863368

Time for backsubstitution: 14.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 470
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 470

## Relational analysis of IS_A1_B2_A2_A1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4713257, upper bound: 1.4729421
time: 4.34 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2

### Relational analysis result of IS_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4717978, upper bound: 1.4751662
time: 4.72 seconds

## BFS IS instance: IS_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -9.0940237, -5.8776746, -9.1746082, -5.8598046, -2.8321729, 2.8815668
1: -14.3949976, -11.0545006, -14.4910002, -11.0261898, -2.7637882, 2.7988052
2: 6.4066491, 9.3040085, 6.3232212, 9.3151550, -2.6656375, 2.7364759
3: -5.2118006, -2.5546591, -5.2276039, -2.4510722, -2.7607284, 2.6729448
4: -11.1238585, -7.9606347, -11.1550608, -7.9222512, -2.8322496, 2.8942795
5: -10.7155714, -7.9956598, -10.7290792, -7.9215655, -2.4091692, 2.3640065
6: -13.6023254, -9.5642681, -13.7222729, -9.5588636, -3.3233805, 3.3636255
7: -4.3414192, -1.8685054, -4.3688574, -1.8302352, -2.3103290, 2.2997875
8: -2.1575167, 0.1959338, -2.3144054, 0.2082939, -2.2905998, 2.4261093
9: -9.3674049, -6.3416443, -9.3729439, -6.3014021, -2.4488144, 2.4077778

Time for backsubstitution: 14.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 470
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 470

## Relational analysis of IS_A1_B2_A2_A2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4785305, upper bound: 1.4729420
time: 4.44 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2

### Relational analysis result of IS_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4790111, upper bound: 1.4751658
time: 4.14 seconds

## BFS IS instance: IS_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -9.1703691, -5.8661690, -9.0763741, -5.9151163, -2.8450766, 2.8044181
1: -14.4793348, -11.0279322, -14.3388357, -11.0680933, -2.7607489, 2.7180033
2: 6.3333769, 9.3133507, 6.4572568, 9.2881012, -2.7082520, 2.6216335
3: -5.2234802, -2.4567430, -5.1934748, -2.5897779, -2.6337023, 2.7367318
4: -11.1487207, -7.9300370, -11.0762119, -7.9895926, -2.8329077, 2.7795000
5: -10.7230453, -7.9226837, -10.6879616, -8.0060501, -2.3469687, 2.3892984
6: -13.7187786, -9.5855350, -13.5373802, -9.6772423, -3.3375225, 3.2400172
7: -4.3658895, -1.8406609, -4.3037243, -1.9087962, -2.2511654, 2.2574673
8: -2.2870371, 0.2036595, -2.0411971, 0.1265268, -2.3396869, 2.2448566
9: -9.3697939, -6.3096251, -9.3468189, -6.3863430, -2.3602891, 2.4166775

Time for backsubstitution: 14.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 470
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 470

## Relational analysis of IS_A2_B1_B1_B1_B1

### Relational analysis result of IS_A2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4744710, upper bound: 1.4660704
time: 4.29 seconds

## Relational analysis of IS_A2_B1_B1_B1_B2

### Relational analysis result of IS_A2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4749568, upper bound: 1.4682611
time: 5.13 seconds

## BFS IS instance: IS_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -9.1706543, -5.8618903, -9.0790291, -5.8923759, -2.8670878, 2.8115363
1: -14.4861917, -11.0275059, -14.3744221, -11.0618811, -2.7824359, 2.7419262
2: 6.3278675, 9.3136978, 6.4282541, 9.2934217, -2.7260504, 2.6361294
3: -5.2242374, -2.4538844, -5.1993499, -2.5736589, -2.6505785, 2.7454655
4: -11.1509829, -7.9292054, -11.0901136, -7.9849477, -2.8690796, 2.7990208
5: -10.7249517, -7.9223604, -10.6979990, -8.0034924, -2.3515358, 2.3964176
6: -13.7198553, -9.5749016, -13.5517406, -9.6218596, -3.3583136, 3.2823062
7: -4.3671169, -1.8398681, -4.3116698, -1.9036620, -2.2829442, 2.2670989
8: -2.2986314, 0.2046614, -2.1009421, 0.1413703, -2.3836932, 2.2869782
9: -9.3702707, -6.3027482, -9.3548555, -6.3507590, -2.3819470, 2.4343243

Time for backsubstitution: 14.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 470
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 470

## Relational analysis of IS_A2_B1_B1_B2_A1

### Relational analysis result of IS_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4727288, upper bound: 1.4749552
time: 4.28 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2

### Relational analysis result of IS_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4749567, upper bound: 1.4754441
time: 4.31 seconds

## BFS IS instance: IS_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -9.1743164, -5.8640747, -9.0912952, -5.9002075, -2.8596869, 2.8250933
1: -14.4841557, -11.0266209, -14.3597765, -11.0607643, -2.7769046, 2.7405691
2: 6.3287354, 9.3148088, 6.4356031, 9.2987013, -2.7186432, 2.6514874
3: -5.2268429, -2.4539256, -5.2057519, -2.5703812, -2.6564617, 2.7518263
4: -11.1528101, -7.9230790, -11.1100311, -7.9652405, -2.8580942, 2.8128560
5: -10.7271643, -7.9218860, -10.7053547, -7.9982185, -2.3594484, 2.4018543
6: -13.7212019, -9.5695076, -13.5881834, -9.6196985, -3.3427095, 3.2811968
7: -4.3676500, -1.8310283, -4.3337994, -1.8736566, -2.2678661, 2.3011823
8: -2.3028083, 0.2073131, -2.0977821, 0.1817436, -2.3823643, 2.2718449
9: -9.3724651, -6.3082767, -9.3592720, -6.3771706, -2.3863363, 2.4310939

Time for backsubstitution: 14.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 470
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 470

## Relational analysis of IS_A2_B1_B2_B1_A1

### Relational analysis result of IS_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4729421, upper bound: 1.4713254
time: 4.85 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2

### Relational analysis result of IS_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4751658, upper bound: 1.4717977
time: 4.24 seconds

## BFS IS instance: IS_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -9.1746082, -5.8598046, -9.0940237, -5.8776746, -2.8815670, 2.8321729
1: -14.4910002, -11.0261898, -14.3949976, -11.0545006, -2.7988052, 2.7637882
2: 6.3232212, 9.3151550, 6.4066491, 9.3040085, -2.7364759, 2.6656375
3: -5.2276039, -2.4510722, -5.2118006, -2.5546591, -2.6729448, 2.7607284
4: -11.1550608, -7.9222512, -11.1238585, -7.9606347, -2.8942795, 2.8322496
5: -10.7290792, -7.9215655, -10.7155714, -7.9956598, -2.3640060, 2.4091699
6: -13.7222729, -9.5588636, -13.6023254, -9.5642681, -3.3636255, 3.3233802
7: -4.3688574, -1.8302352, -4.3414192, -1.8685054, -2.2997875, 2.3103292
8: -2.3144054, 0.2082939, -2.1575167, 0.1959338, -2.4261093, 2.2905998
9: -9.3729439, -6.3014021, -9.3674049, -6.3416443, -2.4077778, 2.4488144

Time for backsubstitution: 14.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 470
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 470

## Relational analysis of IS_A2_B1_B2_B2_A1

### Relational analysis result of IS_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4729421, upper bound: 1.4785304
time: 4.17 seconds

## Relational analysis of IS_A2_B1_B2_B2_A2

### Relational analysis result of IS_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4751659, upper bound: 1.4790110
time: 4.33 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -9.1592522, -5.8788586, -9.1679058, -5.8844457, -2.8724670, 2.8765349
1: -14.4636126, -11.0339355, -14.4506741, -11.0336342, -2.7892923, 2.7856956
2: 6.3502836, 9.3042984, 6.3568006, 9.3084726, -2.7151327, 2.7030044
3: -5.2142987, -2.4731882, -5.2183104, -2.4695468, -2.7447519, 2.7451222
4: -11.1190958, -7.9474130, -11.1371794, -7.9338336, -2.8332129, 2.8463900
5: -10.7095718, -7.9297018, -10.7147617, -7.9249287, -2.4087319, 2.4104221
6: -13.6706333, -9.6271057, -13.7056599, -9.6302910, -3.3285446, 3.3523247
7: -4.3377476, -1.8662193, -4.3592553, -1.8449430, -2.2805271, 2.2909489
8: -2.2461128, 0.1527090, -2.2383800, 0.1903458, -2.3745341, 2.3440773
9: -9.3599586, -6.3174834, -9.3621197, -6.3383002, -2.4092889, 2.4270709

Time for backsubstitution: 14.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 6184
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 470

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4693704, upper bound: 1.4711168
time: 4.42 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4715952, upper bound: 1.4715883
time: 4.13 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -9.1595240, -5.8745661, -9.1706543, -5.8618951, -2.8943472, 2.8877578
1: -14.4704914, -11.0335121, -14.4861870, -11.0275059, -2.8107862, 2.8095279
2: 6.3447671, 9.3046398, 6.3278723, 9.3136959, -2.7261677, 2.7172565
3: -5.2150340, -2.4702408, -5.2242360, -2.4538858, -2.7611482, 2.7539952
4: -11.1213703, -7.9465780, -11.1509819, -7.9292092, -2.8694601, 2.8664138
5: -10.7114553, -7.9293756, -10.7249479, -7.9223614, -2.4150629, 2.4176717
6: -13.6717319, -9.6164742, -13.7198544, -9.5749168, -3.3493886, 3.3941286
7: -4.3390226, -1.8654317, -4.3671150, -1.8398699, -2.3125367, 2.3007331
8: -2.2577052, 0.1537914, -2.2986126, 0.2046614, -2.4183755, 2.3626289
9: -9.3604164, -6.3105922, -9.3702707, -6.3027534, -2.4308758, 2.4448097

Time for backsubstitution: 14.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 6184
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 470

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4693702, upper bound: 1.4783181
time: 4.29 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4715950, upper bound: 1.4788008
time: 4.29 seconds

## BFS IS instance: IS_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -9.1718264, -5.8823490, -9.1743164, -5.8640747, -2.8986645, 2.8803577
1: -14.4553337, -11.0323257, -14.4841557, -11.0266209, -2.7982817, 2.8143058
2: 6.3521786, 9.3099270, 6.3287354, 9.3148088, -2.7267118, 2.7353597
3: -5.2216382, -2.4667366, -5.2268429, -2.4539256, -2.7677126, 2.7601063
4: -11.1412945, -7.9268775, -11.1528101, -7.9230790, -2.8591251, 2.8843582
5: -10.7188444, -7.9241304, -10.7271643, -7.9218860, -2.4169340, 2.4232666
6: -13.7081041, -9.6143274, -13.7212019, -9.5695076, -3.3779769, 3.3489738
7: -4.3610983, -1.8353283, -4.3676500, -1.8310283, -2.3053088, 2.3111730
8: -2.2541103, 0.1940913, -2.3028083, 0.2073131, -2.3655753, 2.3964229
9: -9.3647776, -6.3369589, -9.3724651, -6.3082767, -2.4407187, 2.4360774

Time for backsubstitution: 14.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 470
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 470

## Relational analysis of IS_A2_B2_A2_A1_B1

### Relational analysis result of IS_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4674823, upper bound: 1.4767929
time: 4.29 seconds

## Relational analysis of IS_A2_B2_A2_A1_B2

### Relational analysis result of IS_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4679545, upper bound: 1.4790100
time: 4.03 seconds

## BFS IS instance: IS_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -9.1746006, -5.8598156, -9.1746082, -5.8598046, -2.9100046, 2.9022293
1: -14.4909830, -11.0261927, -14.4910002, -11.0261898, -2.8217201, 2.8361168
2: 6.3232422, 9.3151503, 6.3232212, 9.3151550, -2.7410040, 2.7463284
3: -5.2275963, -2.4510798, -5.2276039, -2.4510722, -2.7765241, 2.7765241
4: -11.1550503, -7.9222684, -11.1550608, -7.9222512, -2.8784819, 2.9174905
5: -10.7290716, -7.9215655, -10.7290792, -7.9215655, -2.4242916, 2.4298472
6: -13.7222652, -9.5589199, -13.7222729, -9.5588636, -3.4199038, 3.3698697
7: -4.3688517, -1.8302602, -4.3688574, -1.8302352, -2.3145857, 2.3431053
8: -2.3143339, 0.2082834, -2.3144054, 0.2082939, -2.3840656, 2.4401531
9: -9.3729353, -6.3014088, -9.3729439, -6.3014021, -2.4584808, 2.4574962

Time for backsubstitution: 14.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 470
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 470

## Relational analysis of IS_A2_B2_A2_A2_B1

### Relational analysis result of IS_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4746820, upper bound: 1.4767922
time: 3.91 seconds

## Relational analysis of IS_A2_B2_A2_A2_B2

### Relational analysis result of IS_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4751656, upper bound: 1.4790100
time: 4.05 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 22.73 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.73
Output dim: 2, lower bound: -1.4693765, upper bound: 1.4672800
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.73
Output dim: 2, lower bound: -1.4716013, upper bound: 1.4677518
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.73
Output dim: 2, lower bound: -1.4693763, upper bound: 1.4744769
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.73
Output dim: 2, lower bound: -1.4716011, upper bound: 1.4749627
IS_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 22.73
Output dim: 2, lower bound: -1.4674882, upper bound: 1.4729482
IS_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 22.73
Output dim: 2, lower bound: -1.4679604, upper bound: 1.4751719
IS_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 22.73
Output dim: 2, lower bound: -1.4746881, upper bound: 1.4729481
IS_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 22.73
Output dim: 2, lower bound: -1.4751716, upper bound: 1.4751719
IS_A1_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 22.73
Output dim: 2, lower bound: -1.4660705, upper bound: 1.4744709
IS_A1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 22.73
Output dim: 2, lower bound: -1.4682595, upper bound: 1.4749566
IS_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 22.73
Output dim: 2, lower bound: -1.4749551, upper bound: 1.4727288
IS_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 22.73
Output dim: 2, lower bound: -1.4754441, upper bound: 1.4749567
IS_A1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 22.73
Output dim: 2, lower bound: -1.4713257, upper bound: 1.4729421
IS_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 22.73
Output dim: 2, lower bound: -1.4717978, upper bound: 1.4751662
IS_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 22.73
Output dim: 2, lower bound: -1.4785305, upper bound: 1.4729420
IS_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 22.73
Output dim: 2, lower bound: -1.4790111, upper bound: 1.4751658
IS_A2_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 22.73
Output dim: 2, lower bound: -1.4744710, upper bound: 1.4660704
IS_A2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 22.73
Output dim: 2, lower bound: -1.4749568, upper bound: 1.4682611
IS_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.73
Output dim: 2, lower bound: -1.4727288, upper bound: 1.4749552
IS_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.73
Output dim: 2, lower bound: -1.4749567, upper bound: 1.4754441
IS_A2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.73
Output dim: 2, lower bound: -1.4729421, upper bound: 1.4713254
IS_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.73
Output dim: 2, lower bound: -1.4751658, upper bound: 1.4717977
IS_A2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.73
Output dim: 2, lower bound: -1.4729421, upper bound: 1.4785304
IS_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.73
Output dim: 2, lower bound: -1.4751659, upper bound: 1.4790110
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.73
Output dim: 2, lower bound: -1.4693704, upper bound: 1.4711168
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.73
Output dim: 2, lower bound: -1.4715952, upper bound: 1.4715883
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.73
Output dim: 2, lower bound: -1.4693702, upper bound: 1.4783181
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.73
Output dim: 2, lower bound: -1.4715950, upper bound: 1.4788008
IS_A2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 22.73
Output dim: 2, lower bound: -1.4674823, upper bound: 1.4767929
IS_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 22.73
Output dim: 2, lower bound: -1.4679545, upper bound: 1.4790100
IS_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 22.73
Output dim: 2, lower bound: -1.4746820, upper bound: 1.4767922
IS_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 22.73
Output dim: 2, lower bound: -1.4751656, upper bound: 1.4790100

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -9.0656796, -5.8999939, -9.0840330, -5.9031219, -2.7547407, 2.7713375
1: -14.3494606, -11.0634336, -14.3504715, -11.0624027, -2.6846023, 2.6820450
2: 6.4420567, 9.2914944, 6.4424305, 9.2968321, -2.6127787, 2.6060786
3: -5.1951294, -2.5784214, -5.2015204, -2.5736248, -2.6215045, 2.6230991
4: -11.0628214, -7.9868484, -11.0994854, -7.9724684, -2.7306643, 2.7636220
5: -10.6931820, -8.0098858, -10.7005215, -8.0005703, -2.3203344, 2.3185458
6: -13.5477848, -9.6344719, -13.5849991, -9.6361809, -3.2137480, 3.2457867
7: -4.3096762, -1.9128118, -4.3317766, -1.8854579, -2.2144427, 2.2189806
8: -2.0883398, 0.1314659, -2.0818067, 0.1757164, -2.2640562, 2.2132726
9: -9.3525543, -6.3609729, -9.3561316, -6.3793535, -2.3496470, 2.3665423

Time for backsubstitution: 14.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6184
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6184

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4693765, upper bound: 1.4639460
time: 4.29 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4693765, upper bound: 1.4672800
time: 4.11 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -9.0807695, -5.8850775, -9.0873909, -5.9023056, -2.7672257, 2.7895031
1: -14.3687572, -11.0517406, -14.3551025, -11.0621204, -2.7034397, 2.6990633
2: 6.4318523, 9.2958660, 6.4402637, 9.2972279, -2.6241288, 2.6138630
3: -5.2016439, -2.5738883, -5.2024441, -2.5731673, -2.6284766, 2.6285558
4: -11.0898819, -7.9667773, -11.1058979, -7.9721985, -2.7547617, 2.7908611
5: -10.7012291, -8.0028887, -10.7012825, -7.9990139, -2.3307209, 2.3257656
6: -13.5527983, -9.6294117, -13.5857496, -9.6356659, -3.2197647, 3.2498629
7: -4.3161087, -1.9024872, -4.3319731, -1.8832781, -2.2234716, 2.2289615
8: -2.0942993, 0.1412449, -2.0820925, 0.1779823, -2.2722816, 2.2233374
9: -9.3579025, -6.3569660, -9.3566132, -6.3785009, -2.3561363, 2.3717489

Time for backsubstitution: 14.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6184
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.18 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=2.6658577919006348
rel_dist={2: [-1.4790870249746977, 1.4790871137342352]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4639
type: A, layer: 1, pos: 4639
type: B, layer: 1, pos: 6184
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 470
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4639

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0504956, upper bound: 1.0555283
time: 4.18 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0555274, upper bound: 1.0555279
time: 4.75 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 9.12 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 9.12
Output dim: 2, lower bound: -1.0504956, upper bound: 1.0555283
IS_B2, status: Status.UNKNOWN, split count: 1, time: 9.12
Output dim: 2, lower bound: -1.0555274, upper bound: 1.0555279

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -9.0963173, -5.8753343, -9.0940342, -5.8776608, -2.4620795, 2.4622393
1: -14.3965416, -11.0441065, -14.3950214, -11.0544968, -2.3885312, 2.3970308
2: 6.4048781, 9.3080320, 6.4066234, 9.3040161, -2.4505739, 2.4528885
3: -5.2136140, -2.5530245, -5.2118101, -2.5546470, -2.5066452, 2.5060444
4: -11.1295519, -7.9597182, -11.1238737, -7.9606109, -2.5097823, 2.5053062
5: -10.7184896, -7.9950051, -10.7155838, -7.9956589, -2.1113539, 2.1078153
6: -13.6051750, -9.5625849, -13.6023312, -9.5642033, -2.9479275, 2.9468727
7: -4.3419952, -1.8644050, -4.3414259, -1.8684752, -2.0889134, 2.0924296
8: -2.1592803, 0.2029767, -2.1576006, 0.1959448, -2.1701264, 2.1755223
9: -9.3684855, -6.3403826, -9.3674126, -6.3416343, -2.1172962, 2.1173275

Time for backsubstitution: 14.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4639
type: B, layer: 1, pos: 6184
type: A, layer: 1, pos: 6184
type: B, layer: 1, pos: 470
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4639

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0504955, upper bound: 1.0504952
time: 4.21 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0504956, upper bound: 1.0555297
time: 4.32 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -9.0963058, -5.8753428, -9.1746120, -5.8598013, -2.4821329, 2.5400825
1: -14.3965349, -11.0441704, -14.4910040, -11.0261898, -2.4278955, 2.4494376
2: 6.4048896, 9.3080044, 6.3232183, 9.3151550, -2.4683695, 2.5382977
3: -5.2136040, -2.5530334, -5.2276077, -2.4510708, -2.6134415, 2.5252209
4: -11.1295176, -7.9597225, -11.1550636, -7.9222455, -2.5432234, 2.5525806
5: -10.7184725, -7.9950113, -10.7290831, -7.9215641, -2.1782751, 2.1343865
6: -13.6051636, -9.5625916, -13.7222738, -9.5588531, -2.9546270, 3.0592542
7: -4.3419929, -1.8644251, -4.3688583, -1.8302293, -2.1366196, 2.1100247
8: -2.1592720, 0.2029467, -2.3144202, 0.2082949, -2.1953993, 2.2689393
9: -9.3684769, -6.3403883, -9.3729439, -6.3013988, -2.1663470, 2.1231008

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6184
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 470
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 4639
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6184

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0532850, upper bound: 1.0549928
time: 4.70 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0555233, upper bound: 1.0555240
time: 4.91 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 24.36 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 24.36
Output dim: 2, lower bound: -1.0504955, upper bound: 1.0504952
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 24.36
Output dim: 2, lower bound: -1.0504956, upper bound: 1.0555297
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 24.36
Output dim: 2, lower bound: -1.0532850, upper bound: 1.0549928
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 24.36
Output dim: 2, lower bound: -1.0555233, upper bound: 1.0555240

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -9.0940342, -5.8776608, -9.0940342, -5.8776608, -2.4597917, 2.4597921
1: -14.3950214, -11.0544968, -14.3950214, -11.0544968, -2.3868384, 2.3868389
2: 6.4066234, 9.3040161, 6.4066234, 9.3040161, -2.4488392, 2.4488387
3: -5.2118101, -2.5546470, -5.2118101, -2.5546470, -2.5047817, 2.5047822
4: -11.1238737, -7.9606109, -11.1238737, -7.9606109, -2.5043654, 2.5043650
5: -10.7155838, -7.9956589, -10.7155838, -7.9956589, -2.1069403, 2.1069398
6: -13.6023312, -9.5642033, -13.6023312, -9.5642033, -2.9454069, 2.9454069
7: -4.3414259, -1.8684752, -4.3414259, -1.8684752, -2.0884070, 2.0884070
8: -2.1576006, 0.1959448, -2.1576006, 0.1959448, -2.1675429, 2.1675429
9: -9.3674126, -6.3416343, -9.3674126, -6.3416343, -2.1158671, 2.1158674

Time for backsubstitution: 14.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6184
type: B, layer: 1, pos: 6184
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 470
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6184

## Relational analysis of IS_B1_A1_A1

### Relational analysis result of IS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0482675, upper bound: 1.0499742
time: 4.34 seconds

## Relational analysis of IS_B1_A1_A2

### Relational analysis result of IS_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0504915, upper bound: 1.0504929
time: 4.64 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -9.1740541, -5.8600497, -9.0940342, -5.8776608, -2.5368605, 2.4795599
1: -14.4906120, -11.0263929, -14.3950214, -11.0544968, -2.4388266, 2.4234085
2: 6.3238883, 9.3150768, 6.4066234, 9.3040161, -2.5335689, 2.4655619
3: -5.2275581, -2.4522831, -5.2118101, -2.5546470, -2.5239143, 2.6107154
4: -11.1547117, -7.9222574, -11.1238737, -7.9606109, -2.5499892, 2.5377383
5: -10.7289677, -7.9221430, -10.7155838, -7.9956589, -2.1212454, 2.1733141
6: -13.7209873, -9.5588531, -13.6023312, -9.5642033, -3.0566630, 2.9517355
7: -4.3688583, -1.8304369, -4.3414259, -1.8684752, -2.1060758, 2.1299739
8: -2.3126140, 0.2082953, -2.1576006, 0.1959448, -2.2595029, 2.1801436
9: -9.3729429, -6.3019805, -9.3674126, -6.3416343, -2.1216407, 2.1611161

Time for backsubstitution: 14.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6184
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6184

## Relational analysis of IS_B1_A2_B1

### Relational analysis result of IS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0499742, upper bound: 1.0532849
time: 4.24 seconds

## Relational analysis of IS_B1_A2_B2

### Relational analysis result of IS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0504917, upper bound: 1.0555259
time: 4.38 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -9.0812969, -5.8900633, -9.1677475, -5.8634562, -2.4632387, 2.5149908
1: -14.3759298, -11.0515528, -14.4825630, -11.0284843, -2.4010048, 2.4298553
2: 6.4265270, 9.2974167, 6.3313026, 9.3126087, -2.4411240, 2.5154457
3: -5.2011495, -2.5720482, -5.2217565, -2.4560113, -2.5850110, 2.4898729
4: -11.0957651, -7.9840603, -11.1479378, -7.9343619, -2.4837523, 2.5186417
5: -10.7008829, -8.0028458, -10.7219162, -7.9229560, -2.1584864, 2.1187563
6: -13.5545740, -9.6202326, -13.7180443, -9.5867825, -2.8729601, 2.9933000
7: -4.3122330, -1.8996066, -4.3658018, -1.8470030, -2.0787201, 2.0689173
8: -2.1026177, 0.1483655, -2.2869325, 0.2019262, -2.1325712, 2.1764736
9: -9.3559237, -6.3495035, -9.3682976, -6.3037548, -2.1461515, 2.1028223

Time for backsubstitution: 14.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 470
type: A, layer: 1, pos: 4639
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 6184
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 470

## Relational analysis of IS_B2_A1_A1

### Relational analysis result of IS_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0515788, upper bound: 1.0542495
time: 4.36 seconds

## Relational analysis of IS_B2_A1_A2

### Relational analysis result of IS_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0532834, upper bound: 1.0549913
time: 4.46 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -9.0962944, -5.8753486, -9.1746063, -5.8598042, -2.4829936, 2.5309072
1: -14.3965187, -11.0441742, -14.4909954, -11.0261927, -2.4231997, 2.4487195
2: 6.4049072, 9.3080044, 6.3232260, 9.3151531, -2.4696703, 2.5279789
3: -5.2135968, -2.5530422, -5.2276030, -2.4510734, -2.6067667, 2.5071261
4: -11.1295033, -7.9597421, -11.1550579, -7.9222579, -2.5182896, 2.5404963
5: -10.7184649, -7.9950132, -10.7290792, -7.9215646, -2.1705494, 2.1343808
6: -13.6051579, -9.5626411, -13.7222748, -9.5588779, -2.9323397, 2.9894009
7: -4.3419876, -1.8644542, -4.3688564, -1.8302432, -2.1205745, 2.0796566
8: -2.1592064, 0.2029362, -2.3143859, 0.2082901, -2.1274257, 2.2305610
9: -9.3684692, -6.3403888, -9.3729401, -6.3014011, -2.1604238, 2.1293483

Time for backsubstitution: 14.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 470
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 4639
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 6184
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4616

## Relational analysis of IS_B2_A2_A1

### Relational analysis result of IS_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0513708, upper bound: 1.0555147
time: 4.46 seconds

## Relational analysis of IS_B2_A2_A2

### Relational analysis result of IS_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0555149, upper bound: 1.0555157
time: 4.55 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 23.76 seconds
IS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 23.76
Output dim: 2, lower bound: -1.0482675, upper bound: 1.0499742
IS_B1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 23.76
Output dim: 2, lower bound: -1.0504915, upper bound: 1.0504929
IS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 23.76
Output dim: 2, lower bound: -1.0499742, upper bound: 1.0532849
IS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 23.76
Output dim: 2, lower bound: -1.0504917, upper bound: 1.0555259
IS_B2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 23.76
Output dim: 2, lower bound: -1.0515788, upper bound: 1.0542495
IS_B2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 23.76
Output dim: 2, lower bound: -1.0532834, upper bound: 1.0549913
IS_B2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 23.76
Output dim: 2, lower bound: -1.0513708, upper bound: 1.0555147
IS_B2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 23.76
Output dim: 2, lower bound: -1.0555149, upper bound: 1.0555157

## BFS IS instance: IS_B1_A1_A1

### Backsubstitution after applying IS history:
0: -9.0790291, -5.8923688, -9.0872250, -5.8813038, -2.4408770, 2.4347110
1: -14.3744278, -11.0618801, -14.3866816, -11.0568686, -2.3599939, 2.3671222
2: 6.4282508, 9.2934217, 6.4147768, 9.3014336, -2.4216113, 2.4260869
3: -5.1993518, -2.5736549, -5.2059951, -2.5595474, -2.4762917, 2.4693861
4: -11.0901146, -7.9849434, -11.1167278, -7.9727330, -2.4449577, 2.4704170
5: -10.6980047, -8.0034914, -10.7084322, -7.9970446, -2.0872622, 2.0912976
6: -13.5517426, -9.6218510, -13.5981293, -9.5921364, -2.8637352, 2.8794641
7: -4.3116722, -1.9036618, -4.3383975, -1.8852580, -2.0307021, 2.0474730
8: -2.1009564, 0.1413708, -2.1301837, 0.1895509, -2.1046839, 2.0830278
9: -9.3548565, -6.3507552, -9.3627605, -6.3439746, -2.0956807, 2.0955851

Time for backsubstitution: 14.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 6184
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 470

## Relational analysis of IS_B1_A1_A1_A1

### Relational analysis result of IS_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0465652, upper bound: 1.0492301
time: 4.76 seconds

## Relational analysis of IS_B1_A1_A1_A2

### Relational analysis result of IS_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0482705, upper bound: 1.0499759
time: 4.75 seconds

## BFS IS instance: IS_B1_A1_A2

### Backsubstitution after applying IS history:
0: -9.0940247, -5.8776684, -9.0940313, -5.8776655, -2.4605961, 2.4545660
1: -14.3950043, -11.0544987, -14.3950109, -11.0544987, -2.3821435, 2.3889999
2: 6.4066410, 9.3040104, 6.4066324, 9.3040123, -2.4500656, 2.4422688
3: -5.2118006, -2.5546553, -5.2118063, -2.5546520, -2.5108590, 2.4866860
4: -11.1238613, -7.9606323, -11.1238689, -7.9606199, -2.4941635, 2.4921863
5: -10.7155752, -7.9956598, -10.7155790, -7.9956598, -2.0997767, 2.1069326
6: -13.6023264, -9.5642567, -13.6023302, -9.5642252, -2.9231896, 2.8781052
7: -4.3414202, -1.8685040, -4.3414240, -1.8684896, -2.0761113, 2.0673223
8: -2.1575346, 0.1959357, -2.1575670, 0.1959410, -2.0998926, 2.1408746
9: -9.3674040, -6.3416371, -9.3674068, -6.3416371, -2.1099477, 2.1221271

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 470
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 6184
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4616

## Relational analysis of IS_B1_A1_A2_A1

### Relational analysis result of IS_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0463571, upper bound: 1.0504870
time: 4.66 seconds

## Relational analysis of IS_B1_A1_A2_A2

### Relational analysis result of IS_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0504881, upper bound: 1.0504897
time: 4.41 seconds

## BFS IS instance: IS_B1_A2_B1

### Backsubstitution after applying IS history:
0: -9.1671915, -5.8637075, -9.0790291, -5.8923688, -2.5117750, 2.4606671
1: -14.4821730, -11.0286884, -14.3744278, -11.0618801, -2.4192224, 2.3965282
2: 6.3319731, 9.3125286, 6.4282508, 9.2934217, -2.5107174, 2.4383340
3: -5.2217083, -2.4572256, -5.1993518, -2.5736549, -2.4885731, 2.5822821
4: -11.1475849, -7.9343743, -11.0901146, -7.9849434, -2.5160503, 2.4782548
5: -10.7217999, -7.9235353, -10.6980047, -8.0034914, -2.1055899, 2.1535296
6: -13.7167587, -9.5867825, -13.5517426, -9.6218510, -2.9907069, 2.8700686
7: -4.3658018, -1.8472103, -4.3116722, -1.9036618, -2.0649576, 2.0721278
8: -2.2851281, 0.2019258, -2.1009564, 0.1413708, -2.1670294, 2.1173372
9: -9.3682957, -6.3043346, -9.3548565, -6.3507552, -2.1013603, 2.1409202

Time for backsubstitution: 14.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 470
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 6184
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 470

## Relational analysis of IS_B1_A2_B1_B1

### Relational analysis result of IS_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0492265, upper bound: 1.0515785
time: 4.50 seconds

## Relational analysis of IS_B1_A2_B1_B2

### Relational analysis result of IS_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0499727, upper bound: 1.0532825
time: 4.28 seconds

## BFS IS instance: IS_B1_A2_B2

### Backsubstitution after applying IS history:
0: -9.1740484, -5.8600569, -9.0940247, -5.8776684, -2.5276866, 2.4804235
1: -14.4906054, -11.0263939, -14.3950043, -11.0544987, -2.4381075, 2.4187136
2: 6.3238983, 9.3150740, 6.4066410, 9.3040104, -2.5232534, 2.4668612
3: -5.2275538, -2.4522879, -5.2118006, -2.5546553, -2.5058174, 2.6040428
4: -11.1547060, -7.9222674, -11.1238613, -7.9606323, -2.5379038, 2.5128038
5: -10.7289629, -7.9221439, -10.7155752, -7.9956598, -2.1212382, 2.1655865
6: -13.7209845, -9.5588779, -13.6023264, -9.5642567, -2.9868102, 2.9294324
7: -4.3688564, -1.8304526, -4.3414202, -1.8685040, -2.0757074, 2.1185536
8: -2.3125799, 0.2082901, -2.1575346, 0.1959357, -2.2211251, 2.1125455
9: -9.3729391, -6.3019805, -9.3674040, -6.3416371, -2.1278887, 2.1551971

Time for backsubstitution: 14.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 470
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 6184
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4616

## Relational analysis of IS_B1_A2_B2_B1

### Relational analysis result of IS_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0504834, upper bound: 1.0513708
time: 4.57 seconds

## Relational analysis of IS_B1_A2_B2_B2

### Relational analysis result of IS_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0504834, upper bound: 1.0555150
time: 4.55 seconds

## BFS IS instance: IS_B2_A1_A1

### Backsubstitution after applying IS history:
0: -9.0682144, -5.8933640, -9.1612377, -5.8650560, -2.4486966, 2.5037675
1: -14.3577671, -11.0526676, -14.4735413, -11.0290585, -2.3801126, 2.4146233
2: 6.4348307, 9.2958460, 6.3355112, 9.3118343, -2.4302058, 2.5078411
3: -5.1976781, -2.5738511, -5.2199841, -2.4569163, -2.5768175, 2.4828341
4: -11.0707560, -7.9851246, -11.1354904, -7.9348865, -2.4553494, 2.5036683
5: -10.6979399, -8.0089207, -10.7204628, -7.9259663, -2.1504698, 2.1102514
6: -13.5517006, -9.6222210, -13.7166271, -9.5877762, -2.8681870, 2.9889083
7: -4.3114948, -1.9079556, -4.3654270, -1.8511888, -2.0733442, 2.0593929
8: -2.1014929, 0.1395392, -2.2863855, 0.1975389, -2.1274571, 2.1678843
9: -9.3540840, -6.3528371, -9.3673887, -6.3054132, -2.1410236, 2.0970809

Time for backsubstitution: 14.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4639
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 6184
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4639

## Relational analysis of IS_B2_A1_A1_A1

### Relational analysis result of IS_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0465625, upper bound: 1.0492252
time: 5.04 seconds

## Relational analysis of IS_B2_A1_A1_A2

### Relational analysis result of IS_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0465605, upper bound: 1.0542495
time: 4.30 seconds

## BFS IS instance: IS_B2_A1_A2

### Backsubstitution after applying IS history:
0: -9.0833054, -5.8784556, -9.1677418, -5.8634572, -2.4602518, 2.5157456
1: -14.3770828, -11.0409756, -14.4825535, -11.0284872, -2.3967972, 2.4283741
2: 6.4246030, 9.3002167, 6.3313046, 9.3126068, -2.4421315, 2.5156751
3: -5.2041883, -2.5693281, -5.2217550, -2.4560132, -2.5854073, 2.4932818
4: -11.0978346, -7.9650521, -11.1479263, -7.9343615, -2.4761882, 2.5376182
5: -10.7059889, -8.0019207, -10.7219143, -7.9229574, -2.1592903, 2.1173582
6: -13.5567150, -9.6171675, -13.7180443, -9.5867844, -2.8744750, 2.9938827
7: -4.3179297, -1.8976306, -4.3658013, -1.8470060, -2.0845366, 2.0682583
8: -2.1074576, 0.1493282, -2.2869310, 0.2019229, -2.1372561, 2.1763616
9: -9.3594255, -6.3488283, -9.3682976, -6.3037558, -2.1485214, 2.1021838

Time for backsubstitution: 14.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4639
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 6184
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4639

## Relational analysis of IS_B2_A1_A2_A1

### Relational analysis result of IS_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0482659, upper bound: 1.0499715
time: 4.78 seconds

## Relational analysis of IS_B2_A1_A2_A2

### Relational analysis result of IS_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0482659, upper bound: 1.0549904
time: 4.60 seconds

## BFS IS instance: IS_B2_A2_A1

### Backsubstitution after applying IS history:
0: -9.0935650, -5.8978939, -9.1740961, -5.8672457, -2.4727116, 2.5076351
1: -14.3612823, -11.0504370, -14.4790735, -11.0269394, -2.3866320, 2.4179285
2: 6.4338713, 9.3026962, 6.3328362, 9.3145494, -2.4402342, 2.5028176
3: -5.2075481, -2.5687659, -5.2262721, -2.4560387, -2.5916052, 2.4889874
4: -11.1156693, -7.9643521, -11.1511211, -7.9237003, -2.4950809, 2.5295725
5: -10.7082434, -7.9975696, -10.7257442, -7.9221277, -2.1592708, 2.1283236
6: -13.5910168, -9.6180868, -13.7203979, -9.5774231, -2.8741121, 2.9301734
7: -4.3343558, -1.8696046, -4.3667402, -1.8316259, -2.1075425, 2.0681057
8: -2.0994487, 0.1887412, -2.2941813, 0.2065740, -2.0645843, 2.1696258
9: -9.3603325, -6.3759217, -9.3721046, -6.3133860, -2.1350465, 2.0930753

Time for backsubstitution: 14.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 470
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 4639
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 6184

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 470

## Relational analysis of IS_B2_A2_A1_B1

### Relational analysis result of IS_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0506273, upper bound: 1.0537767
time: 4.75 seconds

## Relational analysis of IS_B2_A2_A1_B2

### Relational analysis result of IS_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0513693, upper bound: 1.0555130
time: 5.06 seconds

## BFS IS instance: IS_B2_A2_A2

### Backsubstitution after applying IS history:
0: -9.0962963, -5.8753548, -9.1746063, -5.8598042, -2.4829926, 2.5295453
1: -14.3965101, -11.0441761, -14.4909954, -11.0261927, -2.4079466, 2.4449642
2: 6.4049134, 9.3080006, 6.3232260, 9.3151531, -2.4520416, 2.5248227
3: -5.2135940, -2.5530446, -5.2276030, -2.4510734, -2.6276445, 2.5071199
4: -11.1295004, -7.9597445, -11.1550579, -7.9222579, -2.5145779, 2.5644889
5: -10.7184601, -7.9950123, -10.7290792, -7.9215646, -2.1661582, 2.1343794
6: -13.6051540, -9.5626574, -13.7222748, -9.5588779, -2.9243708, 2.9452839
7: -4.3419867, -1.8644570, -4.3688564, -1.8302432, -2.1173863, 2.0978243
8: -2.1591871, 0.2029357, -2.3143859, 0.2082901, -2.0761580, 2.2220111
9: -9.3684692, -6.3403959, -9.3729401, -6.3014011, -2.1593251, 2.1089735

Time for backsubstitution: 14.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 470
type: A, layer: 1, pos: 4639
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 6184

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 470

## Relational analysis of IS_B2_A2_A2_B1

### Relational analysis result of IS_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0547518, upper bound: 1.0537769
time: 4.85 seconds

## Relational analysis of IS_B2_A2_A2_B2

### Relational analysis result of IS_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0555134, upper bound: 1.0555141
time: 4.90 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 24.53 seconds
IS_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 24.53
Output dim: 2, lower bound: -1.0465652, upper bound: 1.0492301
IS_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 24.53
Output dim: 2, lower bound: -1.0482705, upper bound: 1.0499759
IS_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 24.53
Output dim: 2, lower bound: -1.0463571, upper bound: 1.0504870
IS_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 24.53
Output dim: 2, lower bound: -1.0504881, upper bound: 1.0504897
IS_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 24.53
Output dim: 2, lower bound: -1.0492265, upper bound: 1.0515785
IS_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 24.53
Output dim: 2, lower bound: -1.0499727, upper bound: 1.0532825
IS_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 24.53
Output dim: 2, lower bound: -1.0504834, upper bound: 1.0513708
IS_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 24.53
Output dim: 2, lower bound: -1.0504834, upper bound: 1.0555150
IS_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 24.53
Output dim: 2, lower bound: -1.0465625, upper bound: 1.0492252
IS_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 24.53
Output dim: 2, lower bound: -1.0465605, upper bound: 1.0542495
IS_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 24.53
Output dim: 2, lower bound: -1.0482659, upper bound: 1.0499715
IS_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 24.53
Output dim: 2, lower bound: -1.0482659, upper bound: 1.0549904
IS_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.53
Output dim: 2, lower bound: -1.0506273, upper bound: 1.0537767
IS_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.53
Output dim: 2, lower bound: -1.0513693, upper bound: 1.0555130
IS_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.53
Output dim: 2, lower bound: -1.0547518, upper bound: 1.0537769
IS_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.53
Output dim: 2, lower bound: -1.0555134, upper bound: 1.0555141

## BFS IS instance: IS_B1_A1_A1_A1

### Backsubstitution after applying IS history:
0: -9.0659456, -5.8956752, -9.0807076, -5.8829002, -2.4263182, 2.4251151
1: -14.3562717, -11.0629959, -14.3776798, -11.0574245, -2.3390846, 2.3536687
2: 6.4365492, 9.2918520, 6.4189849, 9.3006573, -2.4106870, 2.4189005
3: -5.1958823, -2.5754578, -5.2042246, -2.5604355, -2.4683800, 2.4623454
4: -11.0651045, -7.9860134, -11.1042957, -7.9732599, -2.4165592, 2.4554553
5: -10.6950626, -8.0095654, -10.7069569, -8.0000629, -2.0800285, 2.0828032
6: -13.5488701, -9.6238375, -13.5966787, -9.5931320, -2.8589582, 2.8753042
7: -4.3109312, -1.9120073, -4.3380275, -1.8894674, -2.0253620, 2.0379033
8: -2.0998287, 0.1325450, -2.1296234, 0.1851625, -2.0996442, 2.0744860
9: -9.3530121, -6.3540897, -9.3618374, -6.3456259, -2.0905523, 2.0898378

Time for backsubstitution: 14.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 6184
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 4616

## Relational analysis of IS_B1_A1_A1_A1_B1

### Relational analysis result of IS_B1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0465596, upper bound: 1.0450616
time: 4.55 seconds

## Relational analysis of IS_B1_A1_A1_A1_B2

### Relational analysis result of IS_B1_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0465596, upper bound: 1.0492216
time: 5.24 seconds

## BFS IS instance: IS_B1_A1_A1_A2

### Backsubstitution after applying IS history:
0: -9.0810356, -5.8807631, -9.0872192, -5.8813071, -2.4378853, 2.4463992
1: -14.3755856, -11.0513020, -14.3866749, -11.0568705, -2.3557887, 2.3752680
2: 6.4263258, 9.2962246, 6.4147797, 9.3014317, -2.4226193, 2.4289150
3: -5.2023950, -2.5709348, -5.2059941, -2.5595465, -2.4767618, 2.4727857
4: -11.0921841, -7.9659395, -11.1167154, -7.9727311, -2.4373865, 2.4893944
5: -10.7031097, -8.0025692, -10.7084265, -7.9970427, -2.0919628, 2.0899014
6: -13.5538807, -9.6187859, -13.5981274, -9.5921354, -2.8652482, 2.8822594
7: -4.3173671, -1.9016829, -4.3383980, -1.8852634, -2.0365171, 2.0470595
8: -2.1057940, 0.1423340, -2.1301830, 0.1895456, -2.1099129, 2.0828907
9: -9.3583565, -6.3500795, -9.3627586, -6.3439770, -2.0980501, 2.0949450

Time for backsubstitution: 14.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4616

## Relational analysis of IS_B1_A1_A1_A2_B1

### Relational analysis result of IS_B1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0482650, upper bound: 1.0458024
time: 4.36 seconds

## Relational analysis of IS_B1_A1_A1_A2_B2

### Relational analysis result of IS_B1_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0482650, upper bound: 1.0499674
time: 4.50 seconds

## BFS IS instance: IS_B1_A1_A2_A1

### Backsubstitution after applying IS history:
0: -9.0912952, -5.9002075, -9.0935431, -5.8851275, -2.4503069, 2.4314981
1: -14.3597765, -11.0607643, -14.3831949, -11.0552788, -2.3455510, 2.3702269
2: 6.4356031, 9.2987013, 6.4162474, 9.3033829, -2.4206247, 2.4270926
3: -5.2057519, -2.5703812, -5.2104454, -2.5596409, -2.4955282, 2.4685059
4: -11.1100311, -7.9652405, -11.1199141, -7.9620633, -2.4710813, 2.4812253
5: -10.7053547, -7.9982185, -10.7122459, -7.9962158, -2.0884542, 2.1008658
6: -13.5881834, -9.6196985, -13.6004810, -9.5827818, -2.8649139, 2.8189020
7: -4.3337994, -1.8736566, -4.3393497, -1.8698962, -2.0630984, 2.0558100
8: -2.0977821, 0.1817436, -2.1375365, 0.1942253, -2.0371599, 2.0801280
9: -9.3592720, -6.3771706, -9.3665829, -6.3536153, -2.0845628, 2.0858579

Time for backsubstitution: 14.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 470
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 6184

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 470

## Relational analysis of IS_B1_A1_A2_A1_B1

### Relational analysis result of IS_B1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0456045, upper bound: 1.0487515
time: 4.42 seconds

## Relational analysis of IS_B1_A1_A2_A1_B2

### Relational analysis result of IS_B1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0463556, upper bound: 1.0504873
time: 4.44 seconds

## BFS IS instance: IS_B1_A1_A2_A2

### Backsubstitution after applying IS history:
0: -9.0940237, -5.8776746, -9.0940313, -5.8776655, -2.4605961, 2.4532056
1: -14.3949976, -11.0545006, -14.3950109, -11.0544987, -2.3668904, 2.3889990
2: 6.4066491, 9.3040085, 6.4066324, 9.3040123, -2.4324379, 2.4422684
3: -5.2118006, -2.5546591, -5.2118063, -2.5546520, -2.5362153, 2.4866798
4: -11.1238585, -7.9606347, -11.1238689, -7.9606199, -2.4910717, 2.5158963
5: -10.7155714, -7.9956598, -10.7155790, -7.9956598, -2.0953970, 2.1069322
6: -13.6023254, -9.5642681, -13.6023302, -9.5642252, -2.9152188, 2.8340735
7: -4.3414192, -1.8685054, -4.3414240, -1.8684896, -2.0729218, 2.0893002
8: -2.1575167, 0.1959338, -2.1575670, 0.1959410, -2.0489311, 2.1323254
9: -9.3674049, -6.3416443, -9.3674068, -6.3416371, -2.1088476, 2.1018014

Time for backsubstitution: 14.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 470
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 6184

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 470

## Relational analysis of IS_B1_A1_A2_A2_B1

### Relational analysis result of IS_B1_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0497249, upper bound: 1.0487492
time: 4.40 seconds

## Relational analysis of IS_B1_A1_A2_A2_B2

### Relational analysis result of IS_B1_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0504866, upper bound: 1.0504864
time: 4.36 seconds

## BFS IS instance: IS_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -9.1606836, -5.8653097, -9.0659456, -5.8956752, -2.5005484, 2.4461269
1: -14.4731531, -11.0292568, -14.3562717, -11.0629959, -2.4039922, 2.3756247
2: 6.3361855, 9.3117523, 6.4365492, 9.2918520, -2.5031152, 2.4273992
3: -5.2199354, -2.4581292, -5.1958823, -2.5754578, -2.4815350, 2.5740914
4: -11.1351357, -7.9348950, -11.0651045, -7.9860134, -2.5010781, 2.4498463
5: -10.7203465, -7.9265442, -10.6950626, -8.0095654, -2.0970802, 2.1455135
6: -13.7153406, -9.5877781, -13.5488701, -9.6238375, -2.9863143, 2.8652925
7: -4.3654261, -1.8513963, -4.3109312, -1.9120073, -2.0554309, 2.0667725
8: -2.2845812, 0.1975389, -2.0998287, 0.1325450, -2.1584492, 2.1122568
9: -9.3673840, -6.3059921, -9.3530121, -6.3540897, -2.0956168, 2.1357906

Time for backsubstitution: 14.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 6184
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4616

## Relational analysis of IS_B1_A2_B1_B1_A1

### Relational analysis result of IS_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0450583, upper bound: 1.0515723
time: 4.74 seconds

## Relational analysis of IS_B1_A2_B1_B1_A2

### Relational analysis result of IS_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0492181, upper bound: 1.0515729
time: 4.09 seconds

## BFS IS instance: IS_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -9.1671829, -5.8637094, -9.0810356, -5.8807631, -2.5125294, 2.4576769
1: -14.4821644, -11.0286884, -14.3755856, -11.0513020, -2.4177437, 2.3923225
2: 6.3319783, 9.3125267, 6.4263258, 9.2962246, -2.5109444, 2.4393406
3: -5.2217054, -2.4572275, -5.2023950, -2.5709348, -2.4919739, 2.5826812
4: -11.1475735, -7.9343739, -11.0921841, -7.9659395, -2.5350275, 2.4706900
5: -10.7217979, -7.9235373, -10.7031097, -8.0025692, -2.1041923, 2.1543360
6: -13.7167549, -9.5867844, -13.5538807, -9.6187859, -2.9912896, 2.8715806
7: -4.3658018, -1.8472131, -4.3173671, -1.9016829, -2.0642996, 2.0779438
8: -2.2851272, 0.2019219, -2.1057940, 0.1423340, -2.1669178, 2.1225655
9: -9.3682966, -6.3043346, -9.3583565, -6.3500795, -2.1007204, 2.1432889

Time for backsubstitution: 14.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4616

## Relational analysis of IS_B1_A2_B1_B2_A1

### Relational analysis result of IS_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0457992, upper bound: 1.0532769
time: 4.31 seconds

## Relational analysis of IS_B1_A2_B1_B2_A2

### Relational analysis result of IS_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0499642, upper bound: 1.0532770
time: 4.47 seconds

## BFS IS instance: IS_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -9.1735411, -5.8674955, -9.0912952, -5.9002075, -2.5044136, 2.4701447
1: -14.4786854, -11.0271406, -14.3597765, -11.0607643, -2.4073167, 2.3821511
2: 6.3335066, 9.3144703, 6.4356031, 9.2987013, -2.4980907, 2.4374275
3: -5.2262230, -2.4572530, -5.2057519, -2.5703812, -2.4876790, 2.5888758
4: -11.1507673, -7.9237118, -11.1100311, -7.9652405, -2.5269804, 2.4895749
5: -10.7256298, -7.9227099, -10.7053547, -7.9982185, -2.1151743, 2.1543074
6: -13.7191124, -9.5774231, -13.5881834, -9.6196985, -2.9275808, 2.8712025
7: -4.3667402, -1.8318331, -4.3337994, -1.8736566, -2.0641465, 2.1055346
8: -2.2923775, 0.2065740, -2.0977821, 0.1817436, -2.1601877, 2.0498204
9: -9.3721008, -6.3139658, -9.3592720, -6.3771706, -2.0916071, 2.1298194

Time for backsubstitution: 14.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 470
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 6184

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 470

## Relational analysis of IS_B1_A2_B2_B1_A1

### Relational analysis result of IS_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0487445, upper bound: 1.0506272
time: 4.95 seconds

## Relational analysis of IS_B1_A2_B2_B1_A2

### Relational analysis result of IS_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0504818, upper bound: 1.0513693
time: 4.52 seconds

## BFS IS instance: IS_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -9.1740484, -5.8600569, -9.0940237, -5.8776746, -2.5263262, 2.4804230
1: -14.4906054, -11.0263939, -14.3949976, -11.0545006, -2.4343514, 2.4034605
2: 6.3238983, 9.3150740, 6.4066491, 9.3040085, -2.5200968, 2.4492335
3: -5.2275538, -2.4522879, -5.2118006, -2.5546591, -2.5058107, 2.6249208
4: -11.1547060, -7.9222674, -11.1238585, -7.9606347, -2.5618896, 2.5090919
5: -10.7289629, -7.9221439, -10.7155714, -7.9956598, -2.1212378, 2.1611958
6: -13.7209845, -9.5588779, -13.6023254, -9.5642681, -2.9426928, 2.9214621
7: -4.3688564, -1.8304526, -4.3414192, -1.8685054, -2.0938735, 2.1153641
8: -2.3125799, 0.2082901, -2.1575167, 0.1959338, -2.2125759, 2.0615840
9: -9.3729391, -6.3019805, -9.3674049, -6.3416443, -2.1075149, 2.1540968

Time for backsubstitution: 14.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 6184

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 470

## Relational analysis of IS_B1_A2_B2_B2_A1

### Relational analysis result of IS_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0487445, upper bound: 1.0547519
time: 5.26 seconds

## Relational analysis of IS_B1_A2_B2_B2_A2

### Relational analysis result of IS_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0504817, upper bound: 1.0555144
time: 4.67 seconds

## BFS IS instance: IS_B2_A1_A1_A1

### Backsubstitution after applying IS history:
0: -9.0659456, -5.8956752, -9.1606836, -5.8653097, -2.4461269, 2.5005479
1: -14.3562717, -11.0629959, -14.4731531, -11.0292568, -2.3756242, 2.4039922
2: 6.4365492, 9.2918520, 6.3361855, 9.3117523, -2.4273992, 2.5031157
3: -5.1958823, -2.5754578, -5.2199354, -2.4581292, -2.5740919, 2.4815357
4: -11.0651045, -7.9860134, -11.1351357, -7.9348950, -2.4498463, 2.5010781
5: -10.6950626, -8.0095654, -10.7203465, -7.9265442, -2.1455131, 2.0970802
6: -13.5488701, -9.6238375, -13.7153406, -9.5877781, -2.8652921, 2.9863143
7: -4.3109312, -1.9120073, -4.3654261, -1.8513963, -2.0667725, 2.0554309
8: -2.0998287, 0.1325450, -2.2845812, 0.1975389, -2.1122565, 2.1584489
9: -9.3530121, -6.3540897, -9.3673840, -6.3059921, -2.1357903, 2.0956163

Time for backsubstitution: 14.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 6184
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 4616

## Relational analysis of IS_B2_A1_A1_A1_B1

### Relational analysis result of IS_B2_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0465550, upper bound: 1.0450572
time: 4.93 seconds

## Relational analysis of IS_B2_A1_A1_A1_B2

### Relational analysis result of IS_B2_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0465550, upper bound: 1.0492170
time: 5.27 seconds

## BFS IS instance: IS_B2_A1_A1_A2

### Backsubstitution after applying IS history:
0: -9.1464558, -5.8778267, -9.1612377, -5.8650560, -2.5247517, 2.5217986
1: -14.4523001, -11.0346632, -14.4735413, -11.0290585, -2.4268351, 2.4412255
2: 6.3530784, 9.3030596, 6.3355112, 9.3118343, -2.4998879, 2.5082459
3: -5.2115622, -2.4720678, -5.2199841, -2.4569163, -2.5943117, 2.5880406
4: -11.0963211, -7.9476395, -11.1354904, -7.9348865, -2.4874439, 2.5263710
5: -10.7085466, -7.9354286, -10.7204628, -7.9259663, -2.1610956, 2.1639419
6: -13.6689215, -9.6184597, -13.7166271, -9.5877762, -2.9674850, 2.9936595
7: -4.3382711, -1.8737339, -4.3654270, -1.8511888, -2.0853705, 2.0985975
8: -2.2566004, 0.1449680, -2.2863855, 0.1975389, -2.2062616, 2.1738510
9: -9.3585987, -6.3139334, -9.3673887, -6.3054132, -2.1457744, 2.1450126

Time for backsubstitution: 14.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 6184
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 4616

## Relational analysis of IS_B2_A1_A1_A2_B1

### Relational analysis result of IS_B2_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0465550, upper bound: 1.0500760
time: 4.41 seconds

## Relational analysis of IS_B2_A1_A1_A2_B2

### Relational analysis result of IS_B2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0465550, upper bound: 1.0542410
time: 4.54 seconds

## BFS IS instance: IS_B2_A1_A2_A1

### Backsubstitution after applying IS history:
0: -9.0810356, -5.8807631, -9.1671829, -5.8637094, -2.4576769, 2.5125294
1: -14.3755856, -11.0513020, -14.4821644, -11.0286884, -2.3923225, 2.4177437
2: 6.4263258, 9.2962246, 6.3319783, 9.3125267, -2.4393406, 2.5109448
3: -5.2023950, -2.5709348, -5.2217054, -2.4572275, -2.5826812, 2.4919736
4: -11.0921841, -7.9659395, -11.1475735, -7.9343739, -2.4706898, 2.5350282
5: -10.7031097, -8.0025692, -10.7217979, -7.9235373, -2.1543360, 2.1041918
6: -13.5538807, -9.6187859, -13.7167549, -9.5867844, -2.8715801, 2.9912906
7: -4.3173671, -1.9016829, -4.3658018, -1.8472131, -2.0779438, 2.0642996
8: -2.1057940, 0.1423340, -2.2851272, 0.2019219, -2.1225653, 2.1669178
9: -9.3583565, -6.3500795, -9.3682966, -6.3043346, -2.1432891, 2.1007204

Time for backsubstitution: 14.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4616

## Relational analysis of IS_B2_A1_A2_A1_B1

### Relational analysis result of IS_B2_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0482603, upper bound: 1.0457981
time: 5.04 seconds

## Relational analysis of IS_B2_A1_A2_A1_B2

### Relational analysis result of IS_B2_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0482603, upper bound: 1.0499631
time: 5.01 seconds

## BFS IS instance: IS_B2_A1_A2_A2

### Backsubstitution after applying IS history:
0: -9.1615753, -5.8629818, -9.1677418, -5.8634572, -2.5362997, 2.5337477
1: -14.4717083, -11.0229206, -14.4825535, -11.0284872, -2.4434357, 2.4555230
2: 6.3427973, 9.3074379, 6.3313046, 9.3126068, -2.5119472, 2.5183148
3: -5.2180767, -2.4675124, -5.2217550, -2.4560132, -2.6028595, 2.5951128
4: -11.1235008, -7.9275818, -11.1479263, -7.9343615, -2.5083284, 2.5504673
5: -10.7165384, -7.9284391, -10.7219143, -7.9229574, -2.1707802, 2.1710863
6: -13.6739388, -9.6134119, -13.7180443, -9.5867844, -2.9733038, 2.9986320
7: -4.3447266, -1.8634270, -4.3658013, -1.8470060, -2.0920978, 2.1075683
8: -2.2626839, 0.1547961, -2.2869310, 0.2019229, -2.2139573, 2.1824324
9: -9.3639050, -6.3099093, -9.3682976, -6.3037558, -2.1532383, 2.1501153

Time for backsubstitution: 14.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 6184
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 4616

## Relational analysis of IS_B2_A1_A2_A2_B1

### Relational analysis result of IS_B2_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0482603, upper bound: 1.0508198
time: 4.50 seconds

## Relational analysis of IS_B2_A1_A2_A2_B2

### Relational analysis result of IS_B2_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0482603, upper bound: 1.0549819
time: 4.84 seconds

## BFS IS instance: IS_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -9.0870457, -5.8994656, -9.1610298, -5.8704481, -2.4631066, 2.4931917
1: -14.3523083, -11.0509911, -14.4608698, -11.0280895, -2.3733120, 2.3967052
2: 6.4380932, 9.3019295, 6.3413386, 9.3129978, -2.4330597, 2.4918895
3: -5.2057524, -2.5696509, -5.2226887, -2.4578519, -2.5841756, 2.4811189
4: -11.1032286, -7.9648776, -11.1261196, -7.9247532, -2.4771256, 2.5009480
5: -10.7067680, -8.0005894, -10.7228041, -7.9281702, -2.1506214, 2.1209888
6: -13.5895634, -9.6190796, -13.7175827, -9.5794296, -2.8697491, 2.9253511
7: -4.3339767, -1.8738434, -4.3659821, -1.8400831, -2.0979362, 2.0620613
8: -2.0988891, 0.1843505, -2.2930765, 0.1977949, -2.0556674, 2.1638963
9: -9.3594046, -6.3775721, -9.3702574, -6.3167210, -2.1292982, 2.0879054

Time for backsubstitution: 14.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4639
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 6184

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4639

## Relational analysis of IS_B2_A2_A1_B1_A1

### Relational analysis result of IS_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0455997, upper bound: 1.0487466
time: 5.40 seconds

## Relational analysis of IS_B2_A2_A1_B1_A2

### Relational analysis result of IS_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0455997, upper bound: 1.0537786
time: 4.33 seconds

## BFS IS instance: IS_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -9.0935564, -5.8978953, -9.1761913, -5.8556905, -2.4843378, 2.5047679
1: -14.3612728, -11.0504370, -14.4803743, -11.0163460, -2.3948374, 2.4116797
2: 6.4338770, 9.3026962, 6.3308067, 9.3173409, -2.4430656, 2.5040450
3: -5.2075462, -2.5687673, -5.2293429, -2.4532330, -2.5913219, 2.4894361
4: -11.1156559, -7.9643531, -11.1533775, -7.9047012, -2.4955177, 2.5220468
5: -10.7082415, -7.9975719, -10.7308159, -7.9211922, -2.1575403, 2.1310143
6: -13.5910149, -9.6180840, -13.7225943, -9.5743742, -2.8747168, 2.9311776
7: -4.3343563, -1.8696089, -4.3724861, -1.8295801, -2.1069760, 2.0688410
8: -2.0994475, 0.1887364, -2.2991657, 0.2076483, -2.0645447, 2.1715713
9: -9.3603325, -6.3759241, -9.3755608, -6.3126822, -2.1344004, 2.0953598

Time for backsubstitution: 14.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4639
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 6184

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 4639

## Relational analysis of IS_B2_A2_A1_B2_A1

### Relational analysis result of IS_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0463509, upper bound: 1.0504807
time: 5.39 seconds

## Relational analysis of IS_B2_A2_A1_B2_A2

### Relational analysis result of IS_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0463509, upper bound: 1.0555130
time: 4.85 seconds

## BFS IS instance: IS_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -9.0897779, -5.8769474, -9.1615410, -5.8630047, -2.4733925, 2.5151038
1: -14.3874426, -11.0447311, -14.4727726, -11.0273457, -2.3946228, 2.4236717
2: 6.4091635, 9.3072357, 6.3317513, 9.3135986, -2.4448376, 2.5138431
3: -5.2118020, -2.5539293, -5.2240191, -2.4528828, -2.6202030, 2.4992564
4: -11.1170654, -7.9602718, -11.1300535, -7.9233103, -2.4967275, 2.5357809
5: -10.7169847, -7.9980307, -10.7261391, -7.9276066, -2.1575022, 2.1270418
6: -13.6037064, -9.5636511, -13.7194557, -9.5608807, -2.9199867, 2.9404473
7: -4.3416204, -1.8686916, -4.3681087, -1.8387009, -2.1077566, 2.0917773
8: -2.1586273, 0.1985455, -2.3132806, 0.1995087, -2.0672445, 2.2162557
9: -9.3675442, -6.3420472, -9.3710966, -6.3047361, -2.1535850, 2.1037941

Time for backsubstitution: 14.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4639
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 6184

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4639

## Relational analysis of IS_B2_A2_A2_B1_A1

### Relational analysis result of IS_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0497199, upper bound: 1.0487439
time: 9.02 seconds

## Relational analysis of IS_B2_A2_A2_B1_A2

### Relational analysis result of IS_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0497199, upper bound: 1.0537769
time: 5.04 seconds

## BFS IS instance: IS_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -9.0962887, -5.8753586, -9.1766987, -5.8482552, -2.4946127, 2.5266776
1: -14.3965034, -11.0441751, -14.4923067, -11.0155973, -2.4161539, 2.4387300
2: 6.4049177, 9.3080015, 6.3211894, 9.3179436, -2.4548721, 2.5260491
3: -5.2135916, -2.5530453, -5.2306714, -2.4482839, -2.6273069, 2.5075700
4: -11.1294889, -7.9597459, -11.1573429, -7.9032574, -2.5150175, 2.5569263
5: -10.7184601, -7.9950151, -10.7341480, -7.9206295, -2.1644268, 2.1389198
6: -13.6051531, -9.5626564, -13.7244539, -9.5558319, -2.9249554, 2.9462743
7: -4.3419867, -1.8644608, -4.3746123, -1.8281931, -2.1168184, 2.0985081
8: -2.1591871, 0.2029300, -2.3193762, 0.2093797, -2.0761309, 2.2239652
9: -9.3684654, -6.3403959, -9.3763962, -6.3006992, -2.1586804, 2.1112251

Time for backsubstitution: 14.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4639
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 6184

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4639

## Relational analysis of IS_B2_A2_A2_B2_A1

### Relational analysis result of IS_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0504816, upper bound: 1.0504810
time: 4.78 seconds

## Relational analysis of IS_B2_A2_A2_B2_A2

### Relational analysis result of IS_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0504816, upper bound: 1.0555141
time: 5.06 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 24.75 seconds
IS_B1_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 24.75
Output dim: 2, lower bound: -1.0465596, upper bound: 1.0450616
IS_B1_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 24.75
Output dim: 2, lower bound: -1.0465596, upper bound: 1.0492216
IS_B1_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 24.75
Output dim: 2, lower bound: -1.0482650, upper bound: 1.0458024
IS_B1_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 24.75
Output dim: 2, lower bound: -1.0482650, upper bound: 1.0499674
IS_B1_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 24.75
Output dim: 2, lower bound: -1.0456045, upper bound: 1.0487515
IS_B1_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 24.75
Output dim: 2, lower bound: -1.0463556, upper bound: 1.0504873
IS_B1_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 24.75
Output dim: 2, lower bound: -1.0497249, upper bound: 1.0487492
IS_B1_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 24.75
Output dim: 2, lower bound: -1.0504866, upper bound: 1.0504864
IS_B1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.75
Output dim: 2, lower bound: -1.0450583, upper bound: 1.0515723
IS_B1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.75
Output dim: 2, lower bound: -1.0492181, upper bound: 1.0515729
IS_B1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.75
Output dim: 2, lower bound: -1.0457992, upper bound: 1.0532769
IS_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.75
Output dim: 2, lower bound: -1.0499642, upper bound: 1.0532770
IS_B1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.75
Output dim: 2, lower bound: -1.0487445, upper bound: 1.0506272
IS_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.75
Output dim: 2, lower bound: -1.0504818, upper bound: 1.0513693
IS_B1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.75
Output dim: 2, lower bound: -1.0487445, upper bound: 1.0547519
IS_B1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.75
Output dim: 2, lower bound: -1.0504817, upper bound: 1.0555144
IS_B2_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 24.75
Output dim: 2, lower bound: -1.0465550, upper bound: 1.0450572
IS_B2_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 24.75
Output dim: 2, lower bound: -1.0465550, upper bound: 1.0492170
IS_B2_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 24.75
Output dim: 2, lower bound: -1.0465550, upper bound: 1.0500760
IS_B2_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 24.75
Output dim: 2, lower bound: -1.0465550, upper bound: 1.0542410
IS_B2_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 24.75
Output dim: 2, lower bound: -1.0482603, upper bound: 1.0457981
IS_B2_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 24.75
Output dim: 2, lower bound: -1.0482603, upper bound: 1.0499631
IS_B2_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 24.75
Output dim: 2, lower bound: -1.0482603, upper bound: 1.0508198
IS_B2_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 24.75
Output dim: 2, lower bound: -1.0482603, upper bound: 1.0549819
IS_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.75
Output dim: 2, lower bound: -1.0455997, upper bound: 1.0487466
IS_B2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.75
Output dim: 2, lower bound: -1.0455997, upper bound: 1.0537786
IS_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.75
Output dim: 2, lower bound: -1.0463509, upper bound: 1.0504807
IS_B2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.75
Output dim: 2, lower bound: -1.0463509, upper bound: 1.0555130
IS_B2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.75
Output dim: 2, lower bound: -1.0497199, upper bound: 1.0487439
IS_B2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.75
Output dim: 2, lower bound: -1.0497199, upper bound: 1.0537769
IS_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.75
Output dim: 2, lower bound: -1.0504816, upper bound: 1.0504810
IS_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.75
Output dim: 2, lower bound: -1.0504816, upper bound: 1.0555141

## BFS IS instance: IS_B1_A1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -9.0654812, -5.9031982, -9.0779953, -5.9054828, -2.4032116, 2.4148412
1: -14.3444099, -11.0637608, -14.3425856, -11.0636692, -2.3204274, 2.3165283
2: 6.4461470, 9.2912264, 6.4478946, 9.2953577, -2.3955326, 2.3894243
3: -5.1945724, -2.5806143, -5.1982274, -2.5761733, -2.4501610, 2.4474790
4: -11.0611143, -7.9874687, -11.0903816, -7.9778934, -2.4056082, 2.4320211
5: -10.6917868, -8.0101242, -10.6968107, -8.0026264, -2.0740185, 2.0715590
6: -13.5469666, -9.6423693, -13.5824709, -9.6485224, -2.7997599, 2.8327599
7: -4.3087296, -1.9134117, -4.3302050, -1.8946228, -2.0138550, 2.0247288
8: -2.0798080, 0.1306539, -2.0698781, 0.1707544, -2.0497665, 2.0116992
9: -9.3522148, -6.3660903, -9.3537140, -6.3811545, -2.0543003, 2.0644412

Time for backsubstitution: 14.65 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=2.4546236991882324
rel_dist={2: [-1.0555928908856513, 1.0555926528901223]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4639
type: B, layer: 1, pos: 4639
type: A, layer: 1, pos: 6184
type: B, layer: 1, pos: 6184
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4639

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9012838, upper bound: 0.8963065
time: 4.82 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9012838, upper bound: 0.9012825
time: 4.82 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 9.83 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 9.83
Output dim: 2, lower bound: -0.9012838, upper bound: 0.8963065
IS_A2, status: Status.UNKNOWN, split count: 1, time: 9.83
Output dim: 2, lower bound: -0.9012838, upper bound: 0.9012825

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -9.0940342, -5.8776608, -9.0962868, -5.8753662, -2.3459616, 2.3458042
1: -14.3950214, -11.0544968, -14.3965225, -11.0442467, -2.2779431, 2.2695584
2: 6.4066234, 9.3040161, 6.4049034, 9.3079796, -2.3824224, 2.3801394
3: -5.2118101, -2.5546470, -5.2135897, -2.5530472, -2.4344902, 2.4350829
4: -11.1238737, -7.9606109, -11.1294737, -7.9597273, -2.3984222, 2.4028361
5: -10.7155838, -7.9956589, -10.7184505, -7.9950175, -2.0269241, 2.0304146
6: -13.6023312, -9.5642033, -13.6051369, -9.5626068, -2.8135662, 2.8146062
7: -4.3414259, -1.8684752, -4.3419886, -1.8644593, -2.0285454, 2.0250764
8: -2.1576006, 0.1959448, -2.1592581, 0.2028818, -2.1054735, 2.1001499
9: -9.3674126, -6.3416343, -9.3684692, -6.3403964, -2.0197740, 2.0197432

Time for backsubstitution: 14.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4639
type: A, layer: 1, pos: 6184
type: B, layer: 1, pos: 6184
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4639

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8963070, upper bound: 0.8963065
time: 4.99 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8963070, upper bound: 0.8963065
time: 4.74 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -9.1724749, -5.8609247, -9.0963039, -5.8753443, -2.4194851, 2.3645482
1: -14.4892521, -11.0271034, -14.3965321, -11.0441847, -2.3265967, 2.3050714
2: 6.3262300, 9.3148012, 6.4048910, 9.3079987, -2.4637804, 2.3962054
3: -5.2273884, -2.4565163, -5.2136021, -2.5530350, -2.4534712, 2.5370221
4: -11.1534786, -7.9222965, -11.1295109, -7.9597244, -2.4422054, 2.4353518
5: -10.7285652, -7.9241676, -10.7184677, -7.9950113, -2.0517998, 2.0939736
6: -13.7165394, -9.5588541, -13.6051617, -9.5625896, -2.9204540, 2.8209481
7: -4.3688574, -1.8311636, -4.3419924, -1.8644295, -2.0451117, 2.0708299
8: -2.3063180, 0.2082953, -2.1592693, 0.2029395, -2.1918759, 2.1230602
9: -9.3729343, -6.3040028, -9.3684750, -6.3403883, -2.0255313, 2.0656614

Time for backsubstitution: 14.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6184
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6184

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9004664, upper bound: 0.8993578
time: 4.98 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9012809, upper bound: 0.9012794
time: 4.95 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 24.74 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 24.74
Output dim: 2, lower bound: -0.8963070, upper bound: 0.8963065
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 24.74
Output dim: 2, lower bound: -0.8963070, upper bound: 0.8963065
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 24.74
Output dim: 2, lower bound: -0.9004664, upper bound: 0.8993578
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 24.74
Output dim: 2, lower bound: -0.9012809, upper bound: 0.9012794

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -9.1643562, -5.8652692, -9.0812922, -5.8900628, -2.3937731, 2.3448429
1: -14.4792738, -11.0298262, -14.3759260, -11.0515661, -2.3053908, 2.2776499
2: 6.3358097, 9.3117743, 6.4265299, 9.2974091, -2.4397373, 2.3683767
3: -5.2204642, -2.4623947, -5.2011476, -2.5720506, -2.4173579, 2.5074267
4: -11.1450253, -7.9366555, -11.0957565, -7.9840603, -2.4064837, 2.3756151
5: -10.7200871, -7.9258180, -10.7008791, -8.0028458, -2.0347991, 2.0738978
6: -13.7115402, -9.5919495, -13.5545712, -9.6202354, -2.8535786, 2.7337623
7: -4.3652234, -1.8510361, -4.3122330, -1.8996128, -2.0031967, 2.0118847
8: -2.2737465, 0.2007232, -2.1026168, 0.1483588, -2.0953908, 2.0589943
9: -9.3674364, -6.3067961, -9.3559217, -6.3495054, -2.0043044, 2.0449190

Time for backsubstitution: 14.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 470
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 6184

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 470

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8996591, upper bound: 0.8978406
time: 4.86 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9004650, upper bound: 0.8993563
time: 5.26 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -9.1724663, -5.8609285, -9.0962925, -5.8753510, -2.4103122, 2.3644977
1: -14.4892416, -11.0271063, -14.3965149, -11.0441875, -2.3255329, 2.3000073
2: 6.3262415, 9.3147984, 6.4049087, 9.3079967, -2.4534783, 2.3967204
3: -5.2273846, -2.4565210, -5.2135921, -2.5530446, -2.4337358, 2.5288565
4: -11.1534729, -7.9223118, -11.1294947, -7.9597445, -2.4276505, 2.4104178
5: -10.7285604, -7.9241705, -10.7184601, -7.9950142, -2.0517921, 2.0857959
6: -13.7165375, -9.5588837, -13.6051531, -9.5626431, -2.8468266, 2.7979722
7: -4.3688550, -1.8311809, -4.3419862, -1.8644601, -2.0121312, 2.0539579
8: -2.3062768, 0.2082872, -2.1592047, 0.2029281, -2.1534967, 2.0505106
9: -9.3729305, -6.3040056, -9.3684683, -6.3403926, -2.0312395, 2.0591431

Time for backsubstitution: 14.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 4639
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 6184

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 4616

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9012744, upper bound: 0.8979957
time: 4.65 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9012743, upper bound: 0.9012728
time: 5.04 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 24.47 seconds
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 24.47
Output dim: 2, lower bound: -0.8996591, upper bound: 0.8978406
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 24.47
Output dim: 2, lower bound: -0.9004650, upper bound: 0.8993563
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 24.47
Output dim: 2, lower bound: -0.9012744, upper bound: 0.8979957
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 24.47
Output dim: 2, lower bound: -0.9012743, upper bound: 0.9012728

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -9.1565094, -5.8672042, -9.0682106, -5.8933663, -2.3814087, 2.3299828
1: -14.4684162, -11.0305099, -14.3577671, -11.0526810, -2.2884622, 2.2563524
2: 6.3408661, 9.3108368, 6.4348335, 9.2958393, -2.4312449, 2.3572168
3: -5.2183371, -2.4634848, -5.1976762, -2.5738542, -2.4097304, 2.4988451
4: -11.1300449, -7.9372864, -11.0707483, -7.9851279, -2.3886881, 2.3470945
5: -10.7183342, -7.9294405, -10.6979342, -8.0089207, -2.0258799, 2.0652943
6: -13.7098198, -9.5931463, -13.5516968, -9.6222258, -2.8488374, 2.7287474
7: -4.3647699, -1.8560659, -4.3114929, -1.9079602, -1.9935837, 2.0055757
8: -2.2730875, 0.1954470, -2.1014891, 0.1395326, -2.0867000, 2.0530434
9: -9.3663349, -6.3087897, -9.3540821, -6.3528380, -1.9982543, 2.0393658

Time for backsubstitution: 14.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 4639
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 6184

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4616

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8963937, upper bound: 0.8978362
time: 4.61 seconds

## Relational analysis of IS_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8996522, upper bound: 0.8978361
time: 5.08 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -9.1643457, -5.8652687, -9.0833035, -5.8784571, -2.3945270, 2.3412952
1: -14.4792624, -11.0298252, -14.3770838, -11.0409889, -2.3039098, 2.2724028
2: 6.3358135, 9.3117733, 6.4246011, 9.3002110, -2.4399219, 2.3693838
3: -5.2204618, -2.4623964, -5.2041855, -2.5693307, -2.4205990, 2.5078239
4: -11.1450100, -7.9366570, -11.0978260, -7.9650521, -2.4253664, 2.3668640
5: -10.7200842, -7.9258213, -10.7059851, -8.0019217, -2.0330310, 2.0747013
6: -13.7115383, -9.5919485, -13.5567112, -9.6171684, -2.8541608, 2.7351770
7: -4.3652225, -1.8510405, -4.3179293, -1.8976346, -2.0021935, 2.0176997
8: -2.2737460, 0.2007184, -2.1074567, 0.1493216, -2.0951304, 2.0629082
9: -9.3674335, -6.3067970, -9.3594246, -6.3488293, -2.0033860, 2.0472879

Time for backsubstitution: 14.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 4639
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4616

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8971980, upper bound: 0.8993520
time: 5.32 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9004582, upper bound: 0.8993518
time: 5.28 seconds

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: -9.1718378, -5.8701553, -9.0935631, -5.8978977, -2.3869066, 2.3524241
1: -14.4744844, -11.0280352, -14.3612795, -11.0504494, -2.2926884, 2.2632289
2: 6.3381462, 9.3140497, 6.4338741, 9.3026876, -2.4266930, 2.3671207
3: -5.2257318, -2.4626684, -5.2075458, -2.5687671, -2.4153066, 2.5127871
4: -11.1485825, -7.9240971, -11.1156616, -7.9643526, -2.4152622, 2.3871412
5: -10.7244320, -7.9248686, -10.7082396, -7.9975705, -2.0448914, 2.0743861
6: -13.7142382, -9.5818624, -13.5910120, -9.6180878, -2.7871265, 2.7368689
7: -4.3662262, -1.8328989, -4.3343558, -1.8696102, -1.9999342, 2.0405469
8: -2.2812605, 0.2061567, -2.0994461, 0.1887345, -2.0894952, 1.9872022
9: -9.3718910, -6.3188524, -9.3603334, -6.3759222, -1.9946818, 2.0300846

Time for backsubstitution: 14.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 4639
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 6184

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 470

## Relational analysis of IS_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8997301, upper bound: 0.8971883
time: 4.35 seconds

## Relational analysis of IS_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9012730, upper bound: 0.8979944
time: 5.23 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: -9.1724663, -5.8609285, -9.0962944, -5.8753586, -2.4088821, 2.3644967
1: -14.4892416, -11.0271063, -14.3965111, -11.0441885, -2.3217773, 2.2839923
2: 6.3262415, 9.3147984, 6.4049158, 9.3079958, -2.4503236, 2.3782129
3: -5.2273846, -2.4565210, -5.2135921, -2.5530472, -2.4337287, 2.5487199
4: -11.1534729, -7.9223118, -11.1294937, -7.9597445, -2.4503222, 2.4067073
5: -10.7285604, -7.9241705, -10.7184563, -7.9950132, -2.0517921, 2.0811834
6: -13.7165375, -9.5588837, -13.6051531, -9.5626564, -2.8004980, 2.7900019
7: -4.3688550, -1.8311809, -4.3419867, -1.8644611, -2.0291824, 2.0507698
8: -2.3062768, 0.2082872, -2.1591864, 0.2029271, -2.1449471, 1.9966164
9: -9.3729305, -6.3040056, -9.3684654, -6.3403955, -2.0088511, 2.0580440

Time for backsubstitution: 14.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 4639
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 6184

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 470

## Relational analysis of IS_A2_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8997300, upper bound: 0.9004464
time: 4.42 seconds

## Relational analysis of IS_A2_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9012729, upper bound: 0.9012715
time: 4.97 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 24.26 seconds
IS_A2_B1_B1_A1, status: Status.VERIFIED, split count: 4, time: 24.26
Output dim: 2, lower bound: -0.8963937, upper bound: 0.8978362
IS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 24.26
Output dim: 2, lower bound: -0.8996522, upper bound: 0.8978361
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 24.26
Output dim: 2, lower bound: -0.8971980, upper bound: 0.8993520
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 24.26
Output dim: 2, lower bound: -0.9004582, upper bound: 0.8993518
IS_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 24.26
Output dim: 2, lower bound: -0.8997301, upper bound: 0.8971883
IS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 24.26
Output dim: 2, lower bound: -0.9012730, upper bound: 0.8979944
IS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 24.26
Output dim: 2, lower bound: -0.8997300, upper bound: 0.9004464
IS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 24.26
Output dim: 2, lower bound: -0.9012729, upper bound: 0.9012715

## BFS IS instance: IS_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -9.1565104, -5.8672075, -9.0682106, -5.8933663, -2.3802299, 2.3285551
1: -14.4684105, -11.0305099, -14.3577671, -11.0526810, -2.2724371, 2.2563524
2: 6.3408728, 9.3108397, 6.4348335, 9.2958393, -2.4127207, 2.3572164
3: -5.2183356, -2.4634874, -5.1976762, -2.5738542, -2.4341021, 2.4988422
4: -11.1300421, -7.9372883, -11.0707483, -7.9851279, -2.3855858, 2.3662889
5: -10.7183304, -7.9294395, -10.6979342, -8.0089207, -2.0212808, 2.0644526
6: -13.7098160, -9.5931616, -13.5516968, -9.6222258, -2.8408980, 2.6824965
7: -4.3647690, -1.8560659, -4.3114929, -1.9079602, -1.9903946, 2.0263915
8: -2.2730694, 0.1954474, -2.1014891, 0.1395326, -2.0325322, 2.0458751
9: -9.3663387, -6.3087945, -9.3540821, -6.3528380, -1.9971590, 2.0171208

Time for backsubstitution: 14.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 6184

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4639

## Relational analysis of IS_A2_B1_B1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8946751, upper bound: 0.8978362
time: 4.79 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8946751, upper bound: 0.8978368
time: 5.24 seconds

## BFS IS instance: IS_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -9.1616192, -5.8878317, -9.0827312, -5.8877687, -2.3776617, 2.3179722
1: -14.4438391, -11.0359554, -14.3623486, -11.0419407, -2.2665858, 2.2509017
2: 6.3647385, 9.3065434, 6.4365377, 9.2994347, -2.4101753, 2.3519077
3: -5.2145648, -2.4780681, -5.2025676, -2.5756917, -2.4051304, 2.4895608
4: -11.1311378, -7.9412985, -11.0928316, -7.9668579, -2.4019413, 2.3544145
5: -10.7099438, -7.9283915, -10.7019320, -8.0026093, -2.0216827, 2.0653338
6: -13.6973515, -9.6473026, -13.5543671, -9.6401157, -2.7934055, 2.6754975
7: -4.3572426, -1.8561161, -4.3151855, -1.8993757, -1.9884448, 2.0054917
8: -2.2135415, 0.1862659, -2.0826395, 0.1469498, -2.0315967, 1.9988151
9: -9.3592968, -6.3423510, -9.3584385, -6.3636951, -1.9742608, 2.0107045

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 6184

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4639

## Relational analysis of IS_A2_B1_B2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8922208, upper bound: 0.8993520
time: 5.23 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8922228, upper bound: 0.8993532
time: 5.03 seconds

## BFS IS instance: IS_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -9.1643457, -5.8652720, -9.0833035, -5.8784571, -2.3933492, 2.3398681
1: -14.4792566, -11.0298252, -14.3770838, -11.0409889, -2.2878861, 2.2724018
2: 6.3358207, 9.3117714, 6.4246011, 9.3002110, -2.4213991, 2.3693838
3: -5.2204590, -2.4623985, -5.2041855, -2.5693307, -2.4449654, 2.5078208
4: -11.1450071, -7.9366603, -11.0978260, -7.9650521, -2.4216571, 2.3860822
5: -10.7200794, -7.9258213, -10.7059851, -8.0019217, -2.0284319, 2.0738592
6: -13.7115364, -9.5919647, -13.5567112, -9.6171684, -2.8462205, 2.6889272
7: -4.3652210, -1.8510405, -4.3179293, -1.8976346, -1.9990048, 2.0384641
8: -2.2737279, 0.2007184, -2.1074567, 0.1493216, -2.0409775, 2.0543332
9: -9.3674335, -6.3068027, -9.3594246, -6.3488293, -2.0022898, 2.0250320

Time for backsubstitution: 14.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 6184

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4639

## Relational analysis of IS_A2_B1_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8954810, upper bound: 0.8993515
time: 5.14 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8954810, upper bound: 0.8993526
time: 5.15 seconds

## BFS IS instance: IS_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -9.1587543, -5.8733540, -9.0857162, -5.8997841, -2.3721590, 2.3415031
1: -14.4562798, -11.0291862, -14.3504839, -11.0511150, -2.2710438, 2.2480068
2: 6.3466454, 9.3124943, 6.4389491, 9.3017673, -2.4155359, 2.3589678
3: -5.2221479, -2.4644787, -5.2053871, -2.5698330, -2.4070215, 2.5048003
4: -11.1235809, -7.9251509, -11.1006956, -7.9649858, -2.3865261, 2.3666353
5: -10.7214890, -7.9309144, -10.7064638, -8.0012045, -2.0369081, 2.0653248
6: -13.7113953, -9.5838671, -13.5892687, -9.6192837, -2.7820640, 2.7321634
7: -4.3654671, -1.8413565, -4.3338990, -1.8747075, -1.9930310, 2.0308611
8: -2.2801545, 0.1973782, -2.0987730, 0.1834531, -2.0830011, 1.9781818
9: -9.3700390, -6.3221836, -9.3592091, -6.3779068, -1.9890852, 2.0240250

Time for backsubstitution: 14.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4639
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 6184

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4639

## Relational analysis of IS_A2_B2_B1_A1_B1

### Relational analysis result of IS_A2_B2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8947534, upper bound: 0.8971883
time: 4.32 seconds

## Relational analysis of IS_A2_B2_B1_A1_B2

### Relational analysis result of IS_A2_B2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8947534, upper bound: 0.8971891
time: 4.92 seconds

## BFS IS instance: IS_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -9.1739311, -5.8585939, -9.0935555, -5.8978987, -2.3834758, 2.3626661
1: -14.4757824, -11.0174427, -14.3612690, -11.0504503, -2.2853909, 2.2714295
2: 6.3361177, 9.3168411, 6.4338813, 9.3026896, -2.4279194, 2.3699088
3: -5.2288055, -2.4598548, -5.2075424, -2.5687695, -2.4157572, 2.5123305
4: -11.1508303, -7.9050999, -11.1156454, -7.9643526, -2.4065485, 2.3875782
5: -10.7295065, -7.9239340, -10.7082376, -7.9975748, -2.0468664, 2.0722866
6: -13.7163963, -9.5788088, -13.5910082, -9.6180868, -2.7880263, 2.7374697
7: -4.3719702, -1.8308505, -4.3343554, -1.8696132, -2.0006618, 2.0396366
8: -2.2862380, 0.2072277, -2.0994461, 0.1887283, -2.0914450, 1.9870100
9: -9.3753481, -6.3181486, -9.3603315, -6.3759270, -1.9969692, 2.0291615

Time for backsubstitution: 14.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4639
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 6184

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4639

## Relational analysis of IS_A2_B2_B1_A2_B1

### Relational analysis result of IS_A2_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8962961, upper bound: 0.8979944
time: 4.83 seconds

## Relational analysis of IS_A2_B2_B1_A2_B2

### Relational analysis result of IS_A2_B2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8962961, upper bound: 0.8979951
time: 5.00 seconds

## BFS IS instance: IS_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -9.1593838, -5.8641257, -9.0884466, -5.8772736, -2.3941364, 2.3535805
1: -14.4710159, -11.0282612, -14.3855991, -11.0448580, -2.3000541, 2.2687659
2: 6.3347688, 9.3132429, 6.4100285, 9.3070707, -2.4391127, 2.3700252
3: -5.2238011, -2.4583278, -5.2114377, -2.5541110, -2.4254465, 2.5407205
4: -11.1284666, -7.9233642, -11.1145267, -7.9603786, -2.4215074, 2.3863077
5: -10.7256184, -7.9302111, -10.7166796, -7.9986458, -2.0438032, 2.0721126
6: -13.7136936, -9.5608854, -13.6034069, -9.5638571, -2.7954149, 2.7852750
7: -4.3681078, -1.8396392, -4.3415437, -1.8695561, -2.0222812, 2.0410643
8: -2.3051708, 0.1995072, -2.1585128, 0.1976471, -2.1384144, 1.9876025
9: -9.3710880, -6.3073373, -9.3673553, -6.3423853, -2.0032473, 2.0519915

Time for backsubstitution: 14.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 6184

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 4639

## Relational analysis of IS_A2_B2_B2_A1_B1

### Relational analysis result of IS_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8947534, upper bound: 0.9004464
time: 4.72 seconds

## Relational analysis of IS_A2_B2_B2_A1_B2

### Relational analysis result of IS_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8947534, upper bound: 0.9004472
time: 4.64 seconds

## BFS IS instance: IS_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -9.1745605, -5.8493776, -9.0962849, -5.8753595, -2.4054489, 2.3761172
1: -14.4905519, -11.0165148, -14.3964977, -11.0441895, -2.3144951, 2.2921977
2: 6.3242040, 9.3175888, 6.4049196, 9.3079948, -2.4515467, 2.3810015
3: -5.2304540, -2.4537280, -5.2135892, -2.5530472, -2.4341798, 2.5482116
4: -11.1557550, -7.9033141, -11.1294785, -7.9597487, -2.4415751, 2.4071441
5: -10.7336321, -7.9232354, -10.7184553, -7.9950180, -2.0555220, 2.0790820
6: -13.7186804, -9.5558357, -13.6051493, -9.5626583, -2.8013783, 2.7905850
7: -4.3746109, -1.8291309, -4.3419857, -1.8644657, -2.0298681, 2.0498598
8: -2.3112626, 0.2093754, -2.1591852, 0.2029223, -2.1469021, 1.9964435
9: -9.3763847, -6.3033028, -9.3684654, -6.3403974, -2.0111022, 2.0571198

Time for backsubstitution: 14.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 6184

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4639

## Relational analysis of IS_A2_B2_B2_A2_B1

### Relational analysis result of IS_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8962960, upper bound: 0.9012714
time: 5.33 seconds

## Relational analysis of IS_A2_B2_B2_A2_B2

### Relational analysis result of IS_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8962960, upper bound: 0.9012746
time: 4.34 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 24.58 seconds
IS_A2_B1_B1_A2_B1, status: Status.VERIFIED, split count: 5, time: 24.58
Output dim: 2, lower bound: -0.8946751, upper bound: 0.8978362
IS_A2_B1_B1_A2_B2, status: Status.VERIFIED, split count: 5, time: 24.58
Output dim: 2, lower bound: -0.8946751, upper bound: 0.8978368
IS_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 24.58
Output dim: 2, lower bound: -0.8922208, upper bound: 0.8993520
IS_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 24.58
Output dim: 2, lower bound: -0.8922228, upper bound: 0.8993532
IS_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 24.58
Output dim: 2, lower bound: -0.8954810, upper bound: 0.8993515
IS_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 24.58
Output dim: 2, lower bound: -0.8954810, upper bound: 0.8993526
IS_A2_B2_B1_A1_B1, status: Status.VERIFIED, split count: 5, time: 24.58
Output dim: 2, lower bound: -0.8947534, upper bound: 0.8971883
IS_A2_B2_B1_A1_B2, status: Status.VERIFIED, split count: 5, time: 24.58
Output dim: 2, lower bound: -0.8947534, upper bound: 0.8971891
IS_A2_B2_B1_A2_B1, status: Status.VERIFIED, split count: 5, time: 24.58
Output dim: 2, lower bound: -0.8962961, upper bound: 0.8979944
IS_A2_B2_B1_A2_B2, status: Status.VERIFIED, split count: 5, time: 24.58
Output dim: 2, lower bound: -0.8962961, upper bound: 0.8979951
IS_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 24.58
Output dim: 2, lower bound: -0.8947534, upper bound: 0.9004464
IS_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 24.58
Output dim: 2, lower bound: -0.8947534, upper bound: 0.9004472
IS_A2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 24.58
Output dim: 2, lower bound: -0.8962960, upper bound: 0.9012714
IS_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 24.58
Output dim: 2, lower bound: -0.8962960, upper bound: 0.9012746

## BFS IS instance: IS_A2_B1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -9.1568651, -5.8892469, -9.0804615, -5.8900776, -2.3704424, 2.3140197
1: -14.4415569, -11.0370970, -14.3608551, -11.0522509, -2.2539368, 2.2475538
2: 6.3688817, 9.3060856, 6.4382563, 9.2954464, -2.4021130, 2.3490086
3: -5.2142887, -2.4869020, -5.2007742, -2.5772972, -2.4032221, 2.4803867
4: -11.1288414, -7.9413939, -11.0871916, -7.9677448, -2.3972178, 2.3484299
5: -10.7092438, -7.9332457, -10.6990566, -8.0032568, -2.0089645, 2.0563731
6: -13.6886997, -9.6473656, -13.5515356, -9.6417303, -2.7827988, 2.6729465
7: -4.3571472, -1.8583692, -4.3146267, -1.9034252, -1.9844317, 1.9976006
8: -2.2032139, 0.1861968, -2.0809793, 0.1399641, -2.0149112, 1.9948997
9: -9.3591967, -6.3456941, -9.3573723, -6.3649473, -1.9727139, 2.0023906

Time for backsubstitution: 14.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 6184

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4616

## Relational analysis of IS_A2_B1_B2_A1_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8922208, upper bound: 0.8960917
time: 5.05 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8922228, upper bound: 0.8993533
time: 4.84 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -9.1637421, -5.8867068, -9.1609869, -5.8722439, -2.3979511, 2.3933544
1: -14.4455643, -11.0350399, -14.4568481, -11.0238400, -2.2955470, 2.2888708
2: 6.3617229, 9.3069077, 6.3547168, 9.3066940, -2.4160547, 2.4225554
3: -5.2147884, -2.4726226, -5.2164898, -2.4738419, -2.5065575, 2.5111260
4: -11.1327362, -7.9412446, -11.1185303, -7.9293838, -2.4174891, 2.3864465
5: -10.7104568, -7.9257874, -10.7124844, -7.9291382, -2.0759959, 2.0792351
6: -13.7030163, -9.6473036, -13.6715717, -9.6363487, -2.8030176, 2.7751923
7: -4.3572435, -1.8551776, -4.3419313, -1.8651417, -2.0277405, 2.0140476
8: -2.2216041, 0.1862659, -2.2376475, 0.1524243, -2.0438752, 2.0776215
9: -9.3593063, -6.3397527, -9.3629074, -6.3247838, -2.0221114, 2.0184250

Time for backsubstitution: 14.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 6184

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4616

## Relational analysis of IS_A2_B1_B2_A1_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8922208, upper bound: 0.8960925
time: 4.91 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2_B2

### Relational analysis result of IS_A2_B1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8922208, upper bound: 0.8993528
time: 5.06 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -9.1595945, -5.8666863, -9.0810356, -5.8807631, -2.3861303, 2.3359141
1: -14.4769468, -11.0309563, -14.3755856, -11.0513020, -2.2752690, 2.2690616
2: 6.3399658, 9.3113241, 6.4263258, 9.2962246, -2.4133415, 2.3664851
3: -5.2201905, -2.4712298, -5.2023950, -2.5709348, -2.4430742, 2.4986567
4: -11.1427212, -7.9367523, -11.0921841, -7.9659395, -2.4169507, 2.3801069
5: -10.7193823, -7.9306736, -10.7031097, -8.0025692, -2.0157294, 2.0648980
6: -13.7028904, -9.5920258, -13.5538807, -9.6187859, -2.8356223, 2.6863751
7: -4.3651171, -1.8532895, -4.3173671, -1.9016829, -1.9949799, 2.0305142
8: -2.2632995, 0.2006464, -2.1057940, 0.1423340, -2.0243077, 2.0504122
9: -9.3673306, -6.3101511, -9.3583565, -6.3500795, -2.0007401, 2.0167246

Time for backsubstitution: 14.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 6184

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4616

## Relational analysis of IS_A2_B1_B2_A2_B1_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8954805, upper bound: 0.8960895
time: 4.97 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8954811, upper bound: 0.8960896
time: 5.08 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -9.1664791, -5.8641472, -9.1615753, -5.8629818, -2.4137707, 2.4153399
1: -14.4810047, -11.0289135, -14.4717083, -11.0229206, -2.3168206, 2.3175056
2: 6.3328032, 9.3121281, 6.3427973, 9.3074379, -2.4272385, 2.4400549
3: -5.2206779, -2.4569545, -5.2180767, -2.4675124, -2.5417271, 2.5293381
4: -11.1465893, -7.9366064, -11.1235008, -7.9275818, -2.4372106, 2.4214742
5: -10.7205963, -7.9232154, -10.7165384, -7.9284391, -2.0827641, 2.0877967
6: -13.7172508, -9.5919647, -13.6739388, -9.6134119, -2.8557816, 2.7884693
7: -4.3652210, -1.8501062, -4.3447266, -1.8634270, -2.0382857, 2.0430741
8: -2.2818241, 0.2007189, -2.2626839, 0.1547961, -2.0532618, 2.1333280
9: -9.3674421, -6.3041997, -9.3639050, -6.3099093, -2.0501323, 2.0327187

Time for backsubstitution: 14.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 6184

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 470

## Relational analysis of IS_A2_B1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8939651, upper bound: 0.8985467
time: 5.19 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8939651, upper bound: 0.8989162
time: 5.21 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -9.1546326, -5.8655372, -9.0861797, -5.8795862, -2.3869085, 2.3496342
1: -14.4686947, -11.0293961, -14.3840933, -11.0551691, -2.2874031, 2.2654157
2: 6.3389082, 9.3127975, 6.4117579, 9.3030853, -2.4310451, 2.3671041
3: -5.2235308, -2.4671605, -5.2096457, -2.5557218, -2.4235525, 2.5315599
4: -11.1261835, -7.9234624, -11.1088934, -7.9612665, -2.4168324, 2.3803267
5: -10.7249126, -7.9350648, -10.7137938, -7.9992938, -2.0311260, 2.0631442
6: -13.7050467, -9.5609446, -13.6005821, -9.5654669, -2.7848148, 2.7826967
7: -4.3680019, -1.8418870, -4.3409786, -1.8736019, -2.0182719, 2.0374489
8: -2.2947116, 0.1994333, -2.1568434, 0.1906528, -2.1217160, 1.9762063
9: -9.3709869, -6.3106861, -9.3662910, -6.3436346, -2.0017042, 2.0436678

Time for backsubstitution: 14.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 6184

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 470

## Relational analysis of IS_A2_B2_B2_A1_B1_B1

### Relational analysis result of IS_A2_B2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8947534, upper bound: 0.8997328
time: 4.68 seconds

## Relational analysis of IS_A2_B2_B2_A1_B1_B2

### Relational analysis result of IS_A2_B2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8947534, upper bound: 0.9004464
time: 4.97 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -9.1615410, -5.8630047, -9.1667709, -5.8617172, -2.4145470, 2.4213376
1: -14.4727736, -11.0273466, -14.4800577, -11.0268831, -2.3289723, 2.3164105
2: 6.3317533, 9.3135967, 6.3283563, 9.3142204, -2.4487638, 2.4408541
3: -5.2240195, -2.4528835, -5.2254419, -2.4521608, -2.5259275, 2.5622549
4: -11.1300516, -7.9233136, -11.1400566, -7.9228992, -2.4427280, 2.4271789
5: -10.7261381, -7.9276071, -10.7273159, -7.9251866, -2.0938706, 2.0850258
6: -13.7194557, -9.5608845, -13.7205639, -9.5601177, -2.8049650, 2.8792381
7: -4.3681078, -1.8387041, -4.3684034, -1.8353291, -2.0615888, 2.0451827
8: -2.3132739, 0.1995072, -2.3136735, 0.2030025, -2.1506677, 2.0663817
9: -9.3710976, -6.3047371, -9.3718395, -6.3034019, -2.0510464, 2.0597539

Time for backsubstitution: 14.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 6184

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 470

## Relational analysis of IS_A2_B2_B2_A1_B2_B1

### Relational analysis result of IS_A2_B2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8947534, upper bound: 0.8997335
time: 5.03 seconds

## Relational analysis of IS_A2_B2_B2_A1_B2_B2

### Relational analysis result of IS_A2_B2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8947534, upper bound: 0.9004472
time: 4.98 seconds

## BFS IS instance: IS_A2_B2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -9.1698055, -5.8507900, -9.0940151, -5.8776755, -2.3982220, 2.3721690
1: -14.4882355, -11.0176477, -14.3949890, -11.0545006, -2.3019094, 2.2888412
2: 6.3283434, 9.3171501, 6.4066534, 9.3040104, -2.4434853, 2.3780808
3: -5.2301893, -2.4625635, -5.2117977, -2.5546584, -2.4322853, 2.5390468
4: -11.1534681, -7.9034081, -11.1238451, -7.9606338, -2.4368935, 2.4011664
5: -10.7329330, -7.9280882, -10.7155704, -7.9956617, -2.0436726, 2.0701127
6: -13.7100401, -9.5558968, -13.6023207, -9.5642700, -2.7907772, 2.7880077
7: -4.3745060, -1.8313769, -4.3414183, -1.8685102, -2.0258627, 2.0462275
8: -2.3007903, 0.2093043, -2.1575155, 0.1959295, -2.1302028, 1.9849930
9: -9.3762836, -6.3066578, -9.3674030, -6.3416443, -2.0095592, 2.0488005

Time for backsubstitution: 14.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 6184

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 470

## Relational analysis of IS_A2_B2_B2_A2_B1_B1

### Relational analysis result of IS_A2_B2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8954713, upper bound: 0.8997285
time: 4.63 seconds

## Relational analysis of IS_A2_B2_B2_A2_B1_B2

### Relational analysis result of IS_A2_B2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8954714, upper bound: 0.9008349
time: 5.00 seconds

## BFS IS instance: IS_A2_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -9.1766987, -5.8482556, -9.1745930, -5.8598137, -2.4258413, 2.4344702
1: -14.4923067, -11.0155973, -14.4909735, -11.0261965, -2.3433228, 2.3323395
2: 6.3211899, 9.3179426, 6.3232450, 9.3151503, -2.4614143, 2.4519258
3: -5.2306709, -2.4482856, -5.2275944, -2.4510822, -2.5348978, 2.5697346
4: -11.1573400, -7.9032593, -11.1550331, -7.9222689, -2.4628072, 2.4480414
5: -10.7341480, -7.9206281, -10.7290707, -7.9215698, -2.1033430, 2.0922422
6: -13.7244549, -9.5558348, -13.7222662, -9.5589218, -2.8109264, 2.8845773
7: -4.3746119, -1.8281956, -4.3688517, -1.8302646, -2.0692062, 2.0539966
8: -2.3193703, 0.2093778, -2.3143330, 0.2082777, -2.1591711, 2.0752451
9: -9.3763943, -6.3007002, -9.3729362, -6.3014078, -2.0589032, 2.0648823

Time for backsubstitution: 14.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 6184

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 470

## Relational analysis of IS_A2_B2_B2_A2_B2_B1

### Relational analysis result of IS_A2_B2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8954713, upper bound: 0.8997292
time: 4.75 seconds

## Relational analysis of IS_A2_B2_B2_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8954714, upper bound: 0.8997293
time: 4.97 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 24.64 seconds
IS_A2_B1_B2_A1_B1_B1, status: Status.VERIFIED, split count: 6, time: 24.64
Output dim: 2, lower bound: -0.8922208, upper bound: 0.8960917
IS_A2_B1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 24.64
Output dim: 2, lower bound: -0.8922228, upper bound: 0.8993533
IS_A2_B1_B2_A1_B2_B1, status: Status.VERIFIED, split count: 6, time: 24.64
Output dim: 2, lower bound: -0.8922208, upper bound: 0.8960925
IS_A2_B1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 24.64
Output dim: 2, lower bound: -0.8922208, upper bound: 0.8993528
IS_A2_B1_B2_A2_B1_B1, status: Status.VERIFIED, split count: 6, time: 24.64
Output dim: 2, lower bound: -0.8954805, upper bound: 0.8960895
IS_A2_B1_B2_A2_B1_B2, status: Status.VERIFIED, split count: 6, time: 24.64
Output dim: 2, lower bound: -0.8954811, upper bound: 0.8960896
IS_A2_B1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 24.64
Output dim: 2, lower bound: -0.8939651, upper bound: 0.8985467
IS_A2_B1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 24.64
Output dim: 2, lower bound: -0.8939651, upper bound: 0.8989162
IS_A2_B2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 24.64
Output dim: 2, lower bound: -0.8947534, upper bound: 0.8997328
IS_A2_B2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 24.64
Output dim: 2, lower bound: -0.8947534, upper bound: 0.9004464
IS_A2_B2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 24.64
Output dim: 2, lower bound: -0.8947534, upper bound: 0.8997335
IS_A2_B2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 24.64
Output dim: 2, lower bound: -0.8947534, upper bound: 0.9004472
IS_A2_B2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 24.64
Output dim: 2, lower bound: -0.8954713, upper bound: 0.8997285
IS_A2_B2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 24.64
Output dim: 2, lower bound: -0.8954714, upper bound: 0.9008349
IS_A2_B2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 24.64
Output dim: 2, lower bound: -0.8954713, upper bound: 0.8997292
IS_A2_B2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 24.64
Output dim: 2, lower bound: -0.8954714, upper bound: 0.8997293

## BFS IS instance: IS_A2_B1_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -9.1568651, -5.8892469, -9.0810356, -5.8807678, -2.3751745, 2.3146648
1: -14.4415569, -11.0370970, -14.3755789, -11.0513020, -2.2513118, 2.2616403
2: 6.3688817, 9.3060856, 6.4263296, 9.2962284, -2.3997974, 2.3610001
3: -5.2142887, -2.4869020, -5.2023945, -2.5709372, -2.4102106, 2.4772816
4: -11.1288414, -7.9413939, -11.0921822, -7.9659386, -2.3949118, 2.3522599
5: -10.7092438, -7.9332457, -10.7031078, -8.0025673, -2.0096307, 2.0581679
6: -13.6886997, -9.6473656, -13.5538797, -9.6187935, -2.7850919, 2.6731505
7: -4.3571472, -1.8583692, -4.3173680, -1.9016821, -1.9826617, 1.9980364
8: -2.2032139, 0.1861968, -2.1057813, 0.1423321, -2.0087614, 1.9975467
9: -9.3591967, -6.3456941, -9.3583555, -6.3500838, -1.9912901, 2.0026851

Time for backsubstitution: 14.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 470
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 6184

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 470

## Relational analysis of IS_A2_B1_B2_A1_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8907066, upper bound: 0.8985460
time: 5.16 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8907066, upper bound: 0.8978358
time: 5.00 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -9.1637421, -5.8867068, -9.1615753, -5.8629875, -2.4025402, 2.3928075
1: -14.4455643, -11.0350399, -14.4717035, -11.0229206, -2.2928925, 2.2934024
2: 6.3617229, 9.3069077, 6.3428006, 9.3074369, -2.4168906, 2.4290133
3: -5.2147884, -2.4726226, -5.2180762, -2.4675136, -2.5087132, 2.5079904
4: -11.1327362, -7.9412446, -11.1235008, -7.9275837, -2.4151893, 2.3908119
5: -10.7104568, -7.9257874, -10.7165384, -7.9284420, -2.0761590, 2.0809927
6: -13.7030163, -9.6473036, -13.6739426, -9.6134205, -2.8053408, 2.7697067
7: -4.3572435, -1.8551776, -4.3447247, -1.8634288, -2.0259743, 2.0141501
8: -2.2216041, 0.1862659, -2.2626705, 0.1547952, -2.0377455, 2.0801697
9: -9.3593063, -6.3397527, -9.3639011, -6.3099117, -2.0265980, 2.0187440

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 6184

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 470

## Relational analysis of IS_A2_B1_B2_A1_B2_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8907066, upper bound: 0.8985469
time: 5.32 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2_B2_A2

### Relational analysis result of IS_A2_B1_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8907066, upper bound: 0.8989163
time: 5.48 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -9.1534128, -5.8673992, -9.1615753, -5.8629818, -2.4008284, 2.4092894
1: -14.4629154, -11.0300646, -14.4717083, -11.0229206, -2.2981906, 2.3129892
2: 6.3412247, 9.3105621, 6.3427973, 9.3074379, -2.4168329, 2.4376960
3: -5.2171440, -2.4587784, -5.2180767, -2.4675124, -2.5354290, 2.5250053
4: -11.1216221, -7.9376636, -11.1235008, -7.9275818, -2.4093008, 2.4230795
5: -10.7176628, -7.9292583, -10.7165384, -7.9284391, -2.0787258, 2.0813341
6: -13.7144337, -9.5939655, -13.6739388, -9.6134119, -2.8521538, 2.7859535
7: -4.3644667, -1.8584931, -4.3447266, -1.8634270, -2.0373573, 2.0340719
8: -2.2807205, 0.1919503, -2.2626839, 0.1547961, -2.0507321, 2.1250887
9: -9.3656025, -6.3075333, -9.3639050, -6.3099093, -2.0478282, 2.0285139

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 6184

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4616

## Relational analysis of IS_A2_B1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8939647, upper bound: 0.8952887
time: 5.14 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8939652, upper bound: 0.8952886
time: 4.99 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9.1685486, -5.8525763, -9.1615753, -5.8629818, -2.4103212, 2.4161105
1: -14.4822769, -11.0183220, -14.4717083, -11.0229206, -2.3116684, 2.3190064
2: 6.3308096, 9.3149242, 6.3427973, 9.3074379, -2.4288216, 2.4433451
3: -5.2237344, -2.4541700, -5.2180767, -2.4675124, -2.5440607, 2.5308108
4: -11.1488018, -7.9176068, -11.1235008, -7.9275818, -2.4241676, 2.4259114
5: -10.7256794, -7.9222789, -10.7165384, -7.9284391, -2.0842137, 2.0858064
6: -13.7194538, -9.5889053, -13.6739388, -9.6134119, -2.8566761, 2.7890544
7: -4.3709688, -1.8480808, -4.3447266, -1.8634270, -2.0390615, 2.0421381
8: -2.2868087, 0.2017622, -2.2626839, 0.1547961, -2.0551939, 2.1331253
9: -9.3709164, -6.3035135, -9.3639050, -6.3099093, -2.0506644, 2.0299850

Time for backsubstitution: 14.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 6184

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 4616

## Relational analysis of IS_A2_B1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8939647, upper bound: 0.8945785
time: 4.99 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8939652, upper bound: 0.8956543
time: 5.14 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -9.1546326, -5.8655372, -9.0809307, -5.8808756, -2.3856916, 2.3444080
1: -14.4686947, -11.0293961, -14.3768091, -11.0556250, -2.2862353, 2.2578335
2: 6.3389082, 9.3127975, 6.4151654, 9.3024597, -2.4301724, 2.3632164
3: -5.2235308, -2.4671605, -5.2082186, -2.5564423, -2.4218922, 2.5295515
4: -11.1261835, -7.9234624, -11.0988970, -7.9616923, -2.4163857, 2.3717842
5: -10.7249126, -7.9350648, -10.7126064, -8.0017233, -2.0285301, 2.0615625
6: -13.7050467, -9.5609446, -13.5994434, -9.5662727, -2.7838497, 2.7814317
7: -4.3680019, -1.8418870, -4.3406849, -1.8770077, -2.0151968, 2.0371094
8: -2.2947116, 0.1994333, -2.1563883, 0.1871548, -2.1189828, 1.9758010
9: -9.3709869, -6.3106861, -9.3655376, -6.3449697, -2.0000153, 2.0424404

Time for backsubstitution: 14.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 6184

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 850

## Relational analysis of IS_A2_B2_B2_A1_B1_B1_A1

### Relational analysis result of IS_A2_B2_B2_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8947355, upper bound: 0.8937590
time: 5.15 seconds

## Relational analysis of IS_A2_B2_B2_A1_B1_B1_A2

### Relational analysis result of IS_A2_B2_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8947355, upper bound: 0.8997149
time: 4.96 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -9.1546326, -5.8655372, -9.0960484, -5.8661022, -2.3894844, 2.3587818
1: -14.4686947, -11.0293961, -14.3962517, -11.0439100, -2.2887750, 2.2765117
2: 6.3389082, 9.3127975, 6.4046564, 9.3068066, -2.4325843, 2.3740487
3: -5.2235308, -2.4671605, -5.2148976, -2.5518744, -2.4267726, 2.5355844
4: -11.1261835, -7.9234624, -11.1260834, -7.9416261, -2.4234924, 2.3877997
5: -10.7249126, -7.9350648, -10.7206659, -7.9947419, -2.0356150, 2.0665388
6: -13.7050467, -9.5609446, -13.6044369, -9.5611963, -2.7868328, 2.7859187
7: -4.3680019, -1.8418870, -4.3471699, -1.8664768, -2.0232992, 2.0387537
8: -2.2947116, 0.1994333, -2.1623714, 0.1969810, -2.1254954, 1.9820027
9: -9.3709869, -6.3106861, -9.3708735, -6.3409615, -2.0040317, 2.0478377

Time for backsubstitution: 14.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 6184

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 850

## Relational analysis of IS_A2_B2_B2_A1_B1_B2_A1

### Relational analysis result of IS_A2_B2_B2_A1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8947355, upper bound: 0.8944727
time: 4.99 seconds

## Relational analysis of IS_A2_B2_B2_A1_B1_B2_A2

### Relational analysis result of IS_A2_B2_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8947355, upper bound: 0.9004285
time: 4.93 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -9.1615410, -5.8630047, -9.1615343, -5.8630133, -2.4133439, 2.4176941
1: -14.4727736, -11.0273466, -14.4727583, -11.0273476, -2.3278031, 2.3089118
2: 6.3317533, 9.3135967, 6.3317671, 9.3135948, -2.4478216, 2.4369226
3: -5.2240195, -2.4528835, -5.2240143, -2.4528890, -2.5245571, 2.5602427
4: -11.1300516, -7.9233136, -11.1300440, -7.9233232, -2.4423013, 2.4186409
5: -10.7261381, -7.9276071, -10.7261314, -7.9276047, -2.0918980, 2.0833654
6: -13.7194557, -9.5608845, -13.7194519, -9.5609217, -2.8040028, 2.8779588
7: -4.3681078, -1.8387041, -4.3681054, -1.8387178, -2.0585194, 2.0448384
8: -2.3132739, 0.1995072, -2.3132281, 0.1995020, -2.1479073, 2.0659595
9: -9.3710976, -6.3047371, -9.3710918, -6.3047414, -2.0493565, 2.0585294

Time for backsubstitution: 14.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 6184

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 850

## Relational analysis of IS_A2_B2_B2_A1_B2_B1_A1

### Relational analysis result of IS_A2_B2_B2_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8947355, upper bound: 0.8937597
time: 4.89 seconds

## Relational analysis of IS_A2_B2_B2_A1_B2_B1_A2

### Relational analysis result of IS_A2_B2_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8947355, upper bound: 0.8997159
time: 4.62 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 24.42 seconds
IS_A2_B1_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 24.42
Output dim: 2, lower bound: -0.8907066, upper bound: 0.8985460
IS_A2_B1_B2_A1_B1_B2_A2, status: Status.VERIFIED, split count: 7, time: 24.42
Output dim: 2, lower bound: -0.8907066, upper bound: 0.8978358
IS_A2_B1_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 24.42
Output dim: 2, lower bound: -0.8907066, upper bound: 0.8985469
IS_A2_B1_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 24.42
Output dim: 2, lower bound: -0.8907066, upper bound: 0.8989163
IS_A2_B1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 7, time: 24.42
Output dim: 2, lower bound: -0.8939647, upper bound: 0.8952887
IS_A2_B1_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 7, time: 24.42
Output dim: 2, lower bound: -0.8939652, upper bound: 0.8952886
IS_A2_B1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 7, time: 24.42
Output dim: 2, lower bound: -0.8939647, upper bound: 0.8945785
IS_A2_B1_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 7, time: 24.42
Output dim: 2, lower bound: -0.8939652, upper bound: 0.8956543
IS_A2_B2_B2_A1_B1_B1_A1, status: Status.VERIFIED, split count: 7, time: 24.42
Output dim: 2, lower bound: -0.8947355, upper bound: 0.8937590
IS_A2_B2_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 24.42
Output dim: 2, lower bound: -0.8947355, upper bound: 0.8997149
IS_A2_B2_B2_A1_B1_B2_A1, status: Status.VERIFIED, split count: 7, time: 24.42
Output dim: 2, lower bound: -0.8947355, upper bound: 0.8944727
IS_A2_B2_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 24.42
Output dim: 2, lower bound: -0.8947355, upper bound: 0.9004285
IS_A2_B2_B2_A1_B2_B1_A1, status: Status.VERIFIED, split count: 7, time: 24.42
Output dim: 2, lower bound: -0.8947355, upper bound: 0.8937597
IS_A2_B2_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 24.42
Output dim: 2, lower bound: -0.8947355, upper bound: 0.8997159
IS_A2_B2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 24.42
Output dim: 2, lower bound: -0.8947534, upper bound: 0.9004472
IS_A2_B2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 24.42
Output dim: 2, lower bound: -0.8954713, upper bound: 0.8997285
IS_A2_B2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 24.42
Output dim: 2, lower bound: -0.8954714, upper bound: 0.9008349
IS_A2_B2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 24.42
Output dim: 2, lower bound: -0.8954713, upper bound: 0.8997292
IS_A2_B2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 24.42
Output dim: 2, lower bound: -0.8954714, upper bound: 0.8997293
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=2.384212017059326
rel_dist={2: [-0.9012986819155433, 0.9012981480281343]}

## Binary Search with IS_dual Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2406.35 seconds
