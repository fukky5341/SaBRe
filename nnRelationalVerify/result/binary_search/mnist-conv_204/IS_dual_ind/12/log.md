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
execution time: IAR + LP analysis = 15.36 + 32.45 = 47.80 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3552.20 seconds, max iter: 100)

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
Binary search time: 199.93 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Individual Split (IS_dual_ind) starts
Time budget: 3352.27 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4639
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4639

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4790352, upper bound: 1.4751890
time: 4.46 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4790338, upper bound: 1.4790332
time: 4.14 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 8.78 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 8.78
Output dim: 2, lower bound: -1.4790352, upper bound: 1.4751890
IS_A2, status: Status.UNKNOWN, split count: 1, time: 8.78
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

Time for backsubstitution: 14.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4639

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4751891, upper bound: 1.4751890
time: 4.27 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4751891, upper bound: 1.4751890
time: 4.47 seconds

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

Time for backsubstitution: 14.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4639

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4751891, upper bound: 1.4790338
time: 4.46 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4751891, upper bound: 1.4790336
time: 4.40 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 23.73 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 23.73
Output dim: 2, lower bound: -1.4751891, upper bound: 1.4751890
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 23.73
Output dim: 2, lower bound: -1.4751891, upper bound: 1.4751890
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 23.73
Output dim: 2, lower bound: -1.4751891, upper bound: 1.4790338
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 23.73
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

Time for backsubstitution: 14.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4616

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4679656, upper bound: 1.4751746
time: 4.26 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4751769, upper bound: 1.4751746
time: 4.37 seconds

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

Time for backsubstitution: 14.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4616

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4679656, upper bound: 1.4751744
time: 4.19 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4751769, upper bound: 1.4751745
time: 4.63 seconds

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

Time for backsubstitution: 14.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4616

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4679632, upper bound: 1.4790187
time: 4.05 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4751744, upper bound: 1.4790187
time: 4.01 seconds

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

Time for backsubstitution: 14.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4616

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4679632, upper bound: 1.4790192
time: 4.43 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4751745, upper bound: 1.4790192
time: 4.69 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 24.02 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 24.02
Output dim: 2, lower bound: -1.4679656, upper bound: 1.4751746
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 24.02
Output dim: 2, lower bound: -1.4751769, upper bound: 1.4751746
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 24.02
Output dim: 2, lower bound: -1.4679656, upper bound: 1.4751744
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 24.02
Output dim: 2, lower bound: -1.4751769, upper bound: 1.4751745
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 24.02
Output dim: 2, lower bound: -1.4679632, upper bound: 1.4790187
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 24.02
Output dim: 2, lower bound: -1.4751744, upper bound: 1.4790187
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 24.02
Output dim: 2, lower bound: -1.4679632, upper bound: 1.4790192
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 24.02
Output dim: 2, lower bound: -1.4751745, upper bound: 1.4790192

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -9.0913048, -5.9002004, -9.0937557, -5.8819513, -2.8012848, 2.7855816
1: -14.3597946, -11.0607595, -14.3882380, -11.0549440, -2.7070804, 2.7302189
2: 6.4355822, 9.2987061, 6.4121389, 9.3036537, -2.6309290, 2.6490865
3: -5.2057600, -2.5703738, -5.2110305, -2.5575159, -2.6482441, 2.6406567
4: -11.1100454, -7.9652205, -11.1216125, -7.9614391, -2.8019295, 2.8166327
5: -10.7053633, -7.9982147, -10.7136688, -7.9959769, -2.3384819, 2.3450150
6: -13.5881920, -9.6196489, -13.6012745, -9.5748491, -3.3131371, 3.2869377
7: -4.3338032, -1.8736289, -4.3402424, -1.8692797, -2.2675977, 2.2694628
8: -2.0978289, 0.1817565, -2.1461048, 0.1949663, -2.2927952, 2.3278613
9: -9.3592806, -6.3771639, -9.3669405, -6.3485079, -2.3897066, 2.3727043

Time for backsubstitution: 14.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4616

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4679691, upper bound: 1.4679691
time: 4.14 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4679691, upper bound: 1.4751806
time: 4.39 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -9.0940342, -5.8776674, -9.0940342, -5.8776608, -2.8085217, 2.8073654
1: -14.3950167, -11.0544987, -14.3950214, -11.0544968, -2.7307143, 2.7436886
2: 6.4066296, 9.3040142, 6.4066234, 9.3040161, -2.6451092, 2.6600718
3: -5.2118092, -2.5546496, -5.2118101, -2.5546470, -2.6571622, 2.6571605
4: -11.1238718, -7.9606109, -11.1238737, -7.9606109, -2.8218894, 2.8526182
5: -10.7155809, -7.9956589, -10.7155838, -7.9956589, -2.3458481, 2.3495746
6: -13.6023312, -9.5642166, -13.6023312, -9.5642033, -3.3452673, 3.3078933
7: -4.3414249, -1.8684764, -4.3414259, -1.8684752, -2.2770896, 2.3052363
8: -2.1575828, 0.1959438, -2.1576006, 0.1959448, -2.3344016, 2.3535445
9: -9.3674107, -6.3416414, -9.3674126, -6.3416343, -2.4073677, 2.3942313

Time for backsubstitution: 14.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4616

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4751805, upper bound: 1.4679689
time: 4.24 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4751805, upper bound: 1.4751805
time: 4.23 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -9.0913048, -5.9002004, -9.1743183, -5.8640709, -2.8213544, 2.8688602
1: -14.3597946, -11.0607595, -14.4841595, -11.0266190, -2.7439704, 2.7767313
2: 6.4355822, 9.2987061, 6.3287311, 9.3148108, -2.6478472, 2.7282763
3: -5.2057600, -2.5703738, -5.2268438, -2.4539242, -2.7518358, 2.6564701
4: -11.1100454, -7.9652205, -11.1528139, -7.9230747, -2.8377934, 2.8627958
5: -10.7053633, -7.9982147, -10.7271671, -7.9218874, -2.4082313, 2.3594532
6: -13.5881920, -9.6196489, -13.7212009, -9.5694952, -3.3194232, 3.4012578
7: -4.3338032, -1.8736289, -4.3676510, -1.8310206, -2.3093901, 2.2904010
8: -2.0978289, 0.1817565, -2.3028233, 0.2073154, -2.3051443, 2.4207406
9: -9.3592806, -6.3771639, -9.3724670, -6.3082762, -2.4356351, 2.3784776

Time for backsubstitution: 14.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4616

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4718065, upper bound: 1.4679630
time: 4.05 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4718065, upper bound: 1.4751746
time: 4.20 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -9.0940342, -5.8776674, -9.1746120, -5.8598013, -2.8285875, 2.8907394
1: -14.3950167, -11.0544987, -14.4910040, -11.0261898, -2.7675867, 2.7985122
2: 6.4066296, 9.3040142, 6.3232183, 9.3151550, -2.6620226, 2.7460980
3: -5.2118092, -2.5546496, -5.2276077, -2.4510708, -2.7607384, 2.6729581
4: -11.1238718, -7.9606109, -11.1550636, -7.9222455, -2.8571839, 2.8990464
5: -10.7155809, -7.9956589, -10.7290831, -7.9215641, -2.4155436, 2.3640099
6: -13.6023312, -9.5642166, -13.7222738, -9.5588531, -3.3515949, 3.4221663
7: -4.3414249, -1.8684764, -4.3688583, -1.8302293, -2.3188858, 2.3223169
8: -2.1575828, 0.1959438, -2.3144202, 0.2082949, -2.3470044, 2.4644864
9: -9.3674107, -6.3416414, -9.3729439, -6.3013988, -2.4532919, 2.3999639

Time for backsubstitution: 14.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4616

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4790201, upper bound: 1.4679631
time: 4.32 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4790201, upper bound: 1.4751746
time: 4.17 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -9.1718369, -5.8823404, -9.0937557, -5.8819513, -2.8794680, 2.8056507
1: -14.4553471, -11.0323219, -14.3882380, -11.0549440, -2.7656374, 2.7671885
2: 6.3521647, 9.3099308, 6.4121389, 9.3036537, -2.7199407, 2.6660323
3: -5.2216492, -2.4667296, -5.2110305, -2.5575159, -2.6641333, 2.7443008
4: -11.1413088, -7.9268584, -11.1216125, -7.9614391, -2.8481150, 2.8525333
5: -10.7188530, -7.9241290, -10.7136688, -7.9959769, -2.3529158, 2.4119110
6: -13.7081127, -9.6142759, -13.6012745, -9.5748491, -3.4098444, 3.2932882
7: -4.3611021, -1.8353025, -4.3402424, -1.8692797, -2.2883487, 2.3112807
8: -2.2541590, 0.1941023, -2.1461048, 0.1949663, -2.4108105, 2.3402071
9: -9.3647861, -6.3369546, -9.3669405, -6.3485079, -2.3954535, 2.4185545

Time for backsubstitution: 14.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4616

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4679632, upper bound: 1.4718066
time: 3.99 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4679632, upper bound: 1.4790201
time: 4.19 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -9.1746111, -5.8598051, -9.0940342, -5.8776608, -2.8907261, 2.8274307
1: -14.4910002, -11.0261936, -14.3950214, -11.0544968, -2.7893000, 2.7805610
2: 6.3232241, 9.3151550, 6.4066234, 9.3040161, -2.7342653, 2.6769848
3: -5.2276049, -2.4510739, -5.2118101, -2.5546470, -2.6729579, 2.7607362
4: -11.1550627, -7.9222469, -11.1238737, -7.9606109, -2.8679967, 2.8853748
5: -10.7290783, -7.9215631, -10.7155838, -7.9956589, -2.3602829, 2.4184370
6: -13.7222748, -9.5588684, -13.6023312, -9.5642033, -3.4517217, 3.3142219
7: -4.3688560, -1.8302293, -4.3414259, -1.8684752, -2.2976203, 2.3470323
8: -2.3144000, 0.2082934, -2.1576006, 0.1959448, -2.4293199, 2.3658941
9: -9.3729439, -6.3014030, -9.3674126, -6.3416343, -2.4131546, 2.4400589

Time for backsubstitution: 14.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4616

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4751746, upper bound: 1.4718066
time: 4.12 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4751746, upper bound: 1.4790202
time: 4.11 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -9.1718369, -5.8823404, -9.1743183, -5.8640709, -2.8999825, 2.8895297
1: -14.4553471, -11.0323219, -14.4841595, -11.0266190, -2.8018379, 2.8142498
2: 6.3521647, 9.3099308, 6.3287311, 9.3148108, -2.7230701, 2.7413731
3: -5.2216492, -2.4667296, -5.2268438, -2.4539242, -2.7677250, 2.7601142
4: -11.1413088, -7.9268584, -11.1528139, -7.9230747, -2.8763542, 2.8910403
5: -10.7188530, -7.9241290, -10.7271671, -7.9218874, -2.4230251, 2.4278309
6: -13.7081127, -9.6142759, -13.7212009, -9.5694952, -3.4161315, 3.4075217
7: -4.3611021, -1.8353025, -4.3676510, -1.8310206, -2.3305497, 2.3324800
8: -2.2541590, 0.1941023, -2.3028233, 0.2073154, -2.4248686, 2.4347997
9: -9.3647861, -6.3369546, -9.3724670, -6.3082762, -2.4452720, 2.4282184

Time for backsubstitution: 14.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4616

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4679632, upper bound: 1.4718052
time: 3.97 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4679632, upper bound: 1.4790187
time: 4.13 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9.1746111, -5.8598051, -9.1746120, -5.8598013, -2.9113884, 2.9114017
1: -14.4910002, -11.0261936, -14.4910040, -11.0261898, -2.8256669, 2.8359406
2: 6.3232241, 9.3151550, 6.3232183, 9.3151550, -2.7373915, 2.7523584
3: -5.2276049, -2.4510739, -5.2276077, -2.4510708, -2.7765341, 2.7765338
4: -11.1550627, -7.9222469, -11.1550636, -7.9222455, -2.8962665, 2.9272864
5: -10.7290783, -7.9215631, -10.7290831, -7.9215641, -2.4303842, 2.4341106
6: -13.7222748, -9.5588684, -13.7222738, -9.5588531, -3.4579654, 3.4284115
7: -4.3688560, -1.8302293, -4.3688583, -1.8302293, -2.3402081, 2.3656816
8: -2.3144000, 0.2082934, -2.3144202, 0.2082949, -2.4433670, 2.4785302
9: -9.3729439, -6.3014030, -9.3729439, -6.3013988, -2.4629693, 2.4496832

Time for backsubstitution: 14.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4616

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4751746, upper bound: 1.4718058
time: 4.34 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4751746, upper bound: 1.4790187
time: 4.01 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 23.24 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.24
Output dim: 2, lower bound: -1.4679691, upper bound: 1.4679691
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.24
Output dim: 2, lower bound: -1.4679691, upper bound: 1.4751806
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.24
Output dim: 2, lower bound: -1.4751805, upper bound: 1.4679689
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.24
Output dim: 2, lower bound: -1.4751805, upper bound: 1.4751805
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.24
Output dim: 2, lower bound: -1.4718065, upper bound: 1.4679630
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.24
Output dim: 2, lower bound: -1.4718065, upper bound: 1.4751746
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.24
Output dim: 2, lower bound: -1.4790201, upper bound: 1.4679631
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.24
Output dim: 2, lower bound: -1.4790201, upper bound: 1.4751746
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.24
Output dim: 2, lower bound: -1.4679632, upper bound: 1.4718066
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.24
Output dim: 2, lower bound: -1.4679632, upper bound: 1.4790201
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.24
Output dim: 2, lower bound: -1.4751746, upper bound: 1.4718066
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.24
Output dim: 2, lower bound: -1.4751746, upper bound: 1.4790202
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.24
Output dim: 2, lower bound: -1.4679632, upper bound: 1.4718052
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.24
Output dim: 2, lower bound: -1.4679632, upper bound: 1.4790187
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.24
Output dim: 2, lower bound: -1.4751746, upper bound: 1.4718058
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.24
Output dim: 2, lower bound: -1.4751746, upper bound: 1.4790187

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -9.0913048, -5.9002004, -9.0913048, -5.9002004, -2.7829504, 2.7829504
1: -14.3597946, -11.0607595, -14.3597946, -11.0607595, -2.7010221, 2.7010231
2: 6.4355822, 9.2987061, 6.4355822, 9.2987061, -2.6258225, 2.6258221
3: -5.2057600, -2.5703738, -5.2057600, -2.5703738, -2.6353862, 2.6353862
4: -11.1100454, -7.9652205, -11.1100454, -7.9652205, -2.7977085, 2.7977087
5: -10.7053633, -7.9982147, -10.7053633, -7.9982147, -2.3362637, 2.3362632
6: -13.5881920, -9.6196489, -13.5881920, -9.6196489, -3.2733903, 3.2733903
7: -4.3338032, -1.8736289, -4.3338032, -1.8736289, -2.2595396, 2.2595391
8: -2.0978289, 0.1817565, -2.0978289, 0.1817565, -2.2795854, 2.2795854
9: -9.3592806, -6.3771639, -9.3592806, -6.3771639, -2.3634281, 2.3634286

Time for backsubstitution: 14.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6184

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4644240, upper bound: 1.4677554
time: 4.29 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4679622, upper bound: 1.4679642
time: 4.28 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -9.0913048, -5.9002004, -9.0940342, -5.8776674, -2.8055644, 2.7859030
1: -14.3597946, -11.0607595, -14.3950167, -11.0544987, -2.7075872, 2.7371202
2: 6.4355822, 9.2987061, 6.4066296, 9.3040142, -2.6313200, 2.6545701
3: -5.2057600, -2.5703738, -5.2118092, -2.5546496, -2.6511104, 2.6414354
4: -11.1100454, -7.9652205, -11.1238718, -7.9606109, -2.8025846, 2.8170047
5: -10.7053633, -7.9982147, -10.7155809, -7.9956589, -2.3387856, 2.3470483
6: -13.5881920, -9.6196489, -13.6023312, -9.5642166, -3.3141994, 3.2880836
7: -4.3338032, -1.8736289, -4.3414249, -1.8684764, -2.2653418, 2.2681022
8: -2.0978289, 0.1817565, -2.1575828, 0.1959438, -2.2937727, 2.3393393
9: -9.3592806, -6.3771639, -9.3674107, -6.3416414, -2.3985271, 2.3722646

Time for backsubstitution: 14.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6184

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4644239, upper bound: 1.4749643
time: 4.53 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4679622, upper bound: 1.4751736
time: 4.06 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -9.0940342, -5.8776674, -9.0913048, -5.9002004, -2.7859030, 2.8055644
1: -14.3950167, -11.0544987, -14.3597946, -11.0607595, -2.7371206, 2.7075877
2: 6.4066296, 9.3040142, 6.4355822, 9.2987061, -2.6545701, 2.6313200
3: -5.2118092, -2.5546496, -5.2057600, -2.5703738, -2.6414354, 2.6511104
4: -11.1238718, -7.9606109, -11.1100454, -7.9652205, -2.8170052, 2.8025842
5: -10.7155809, -7.9956589, -10.7053633, -7.9982147, -2.3470478, 2.3387861
6: -13.6023312, -9.5642166, -13.5881920, -9.6196489, -3.2880826, 3.3141990
7: -4.3414249, -1.8684764, -4.3338032, -1.8736289, -2.2681026, 2.2653420
8: -2.1575828, 0.1959438, -2.0978289, 0.1817565, -2.3393393, 2.2937727
9: -9.3674107, -6.3416414, -9.3592806, -6.3771639, -2.3722649, 2.3985271

Time for backsubstitution: 14.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6184

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4716027, upper bound: 1.4677540
time: 4.39 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4751730, upper bound: 1.4679623
time: 4.30 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -9.0940342, -5.8776674, -9.0940342, -5.8776674, -2.8073645, 2.8073640
1: -14.3950167, -11.0544987, -14.3950167, -11.0544987, -2.7307124, 2.7307134
2: 6.4066296, 9.3040142, 6.4066296, 9.3040142, -2.6451077, 2.6451082
3: -5.2118092, -2.5546496, -5.2118092, -2.5546496, -2.6571596, 2.6571596
4: -11.1238718, -7.9606109, -11.1238718, -7.9606109, -2.8526087, 2.8526092
5: -10.7155809, -7.9956589, -10.7155809, -7.9956589, -2.3458467, 2.3458471
6: -13.6023312, -9.5642166, -13.6023312, -9.5642166, -3.3078909, 3.3078899
7: -4.3414249, -1.8684764, -4.3414249, -1.8684764, -2.3052330, 2.3052330
8: -2.1575828, 0.1959438, -2.1575828, 0.1959438, -2.3343997, 2.3343997
9: -9.3674107, -6.3416414, -9.3674107, -6.3416414, -2.3942299, 2.3942292

Time for backsubstitution: 14.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6184

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4716031, upper bound: 1.4677540
time: 4.13 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4751735, upper bound: 1.4679622
time: 4.18 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -9.0913048, -5.9002004, -9.1718369, -5.8823404, -2.8030195, 2.8662400
1: -14.3597946, -11.0607595, -14.4553471, -11.0323219, -2.7379923, 2.7594838
2: 6.4355822, 9.2987061, 6.3521647, 9.3099308, -2.6427684, 2.7148018
3: -5.2057600, -2.5703738, -5.2216492, -2.4667296, -2.7390304, 2.6512754
4: -11.1100454, -7.9652205, -11.1413088, -7.9268584, -2.8335729, 2.8438935
5: -10.7053633, -7.9982147, -10.7188530, -7.9241290, -2.4059949, 2.3506980
6: -13.5881920, -9.6196489, -13.7081127, -9.6142759, -3.2797399, 3.3878751
7: -4.3338032, -1.8736289, -4.3611021, -1.8353025, -2.3013573, 2.2803271
8: -2.0978289, 0.1817565, -2.2541590, 0.1941023, -2.2919312, 2.3972580
9: -9.3592806, -6.3771639, -9.3647861, -6.3369546, -2.4092789, 2.3691750

Time for backsubstitution: 14.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6184

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4682614, upper bound: 1.4677498
time: 4.35 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4717996, upper bound: 1.4679584
time: 4.21 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -9.0913048, -5.9002004, -9.1746111, -5.8598051, -2.8256302, 2.8680048
1: -14.3597946, -11.0607595, -14.4910002, -11.0261936, -2.7444615, 2.7787099
2: 6.4355822, 9.2987061, 6.3232241, 9.3151550, -2.6482329, 2.7297034
3: -5.2057600, -2.5703738, -5.2276049, -2.4510739, -2.7546861, 2.6572311
4: -11.1100454, -7.9652205, -11.1550627, -7.9222469, -2.8359947, 2.8631120
5: -10.7053633, -7.9982147, -10.7290783, -7.9215631, -2.4076986, 2.3614841
6: -13.5881920, -9.6196489, -13.7222748, -9.5588684, -3.3204989, 3.3944147
7: -4.3338032, -1.8736289, -4.3688560, -1.8302293, -2.3071375, 2.2886739
8: -2.0978289, 0.1817565, -2.3144000, 0.2082934, -2.3061223, 2.4219344
9: -9.3592806, -6.3771639, -9.3729439, -6.3014030, -2.4444489, 2.3780510

Time for backsubstitution: 14.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6184

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4682614, upper bound: 1.4749584
time: 4.25 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4717996, upper bound: 1.4751678
time: 4.39 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -9.0940342, -5.8776674, -9.1718369, -5.8823404, -2.8059731, 2.8816078
1: -14.3950167, -11.0544987, -14.4553471, -11.0323219, -2.7740898, 2.7624042
2: 6.4066296, 9.3040142, 6.3521647, 9.3099308, -2.6715169, 2.7171736
3: -5.2118092, -2.5546496, -5.2216492, -2.4667296, -2.7450795, 2.6669996
4: -11.1238718, -7.9606109, -11.1413088, -7.9268584, -2.8522725, 2.8487687
5: -10.7155809, -7.9956589, -10.7188530, -7.9241290, -2.4127660, 2.3532209
6: -13.6023312, -9.5642166, -13.7081127, -9.6142759, -3.2944341, 3.4109073
7: -4.3414249, -1.8684764, -4.3611021, -1.8353025, -2.3099203, 2.2855422
8: -2.1575828, 0.1959438, -2.2541590, 0.1941023, -2.3516850, 2.4033296
9: -9.3674107, -6.3416414, -9.3647861, -6.3369546, -2.4181147, 2.4042735

Time for backsubstitution: 14.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6184

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4754457, upper bound: 1.4677481
time: 4.03 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4790126, upper bound: 1.4679564
time: 4.45 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -9.0940342, -5.8776674, -9.1746111, -5.8598051, -2.8274293, 2.8897626
1: -14.3950167, -11.0544987, -14.4910002, -11.0261936, -2.7675862, 2.7877314
2: 6.4066296, 9.3040142, 6.3232241, 9.3151550, -2.6620216, 2.7336364
3: -5.2118092, -2.5546496, -5.2276049, -2.4510739, -2.7607353, 2.6729553
4: -11.1238718, -7.9606109, -11.1550627, -7.9222469, -2.8853703, 2.8990374
5: -10.7155809, -7.9956589, -10.7290783, -7.9215631, -2.4153337, 2.3602829
6: -13.6023312, -9.5642166, -13.7222748, -9.5588684, -3.3142190, 3.4205542
7: -4.3414249, -1.8684764, -4.3688560, -1.8302293, -2.3470287, 2.3223152
8: -2.1575828, 0.1959438, -2.3144000, 0.2082934, -2.3470030, 2.4282846
9: -9.3674107, -6.3416414, -9.3729439, -6.3014030, -2.4400568, 2.3999605

Time for backsubstitution: 14.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6184

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4754460, upper bound: 1.4677481
time: 4.32 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4790130, upper bound: 1.4679564
time: 4.28 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -9.1718369, -5.8823404, -9.0913048, -5.9002004, -2.8662405, 2.8030195
1: -14.4553471, -11.0323219, -14.3597946, -11.0607595, -2.7594833, 2.7379923
2: 6.3521647, 9.3099308, 6.4355822, 9.2987061, -2.7148018, 2.6427689
3: -5.2216492, -2.4667296, -5.2057600, -2.5703738, -2.6512754, 2.7390304
4: -11.1413088, -7.9268584, -11.1100454, -7.9652205, -2.8438931, 2.8335726
5: -10.7188530, -7.9241290, -10.7053633, -7.9982147, -2.3506985, 2.4059947
6: -13.7081127, -9.6142759, -13.5881920, -9.6196489, -3.3878746, 3.2797403
7: -4.3611021, -1.8353025, -4.3338032, -1.8736289, -2.2803273, 2.3013570
8: -2.2541590, 0.1941023, -2.0978289, 0.1817565, -2.3972583, 2.2919312
9: -9.3647861, -6.3369546, -9.3592806, -6.3771639, -2.3691750, 2.4092789

Time for backsubstitution: 14.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6184

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4644181, upper bound: 1.4715933
time: 4.50 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4679563, upper bound: 1.4718017
time: 4.80 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -9.1718369, -5.8823404, -9.0940342, -5.8776674, -2.8816080, 2.8059726
1: -14.4553471, -11.0323219, -14.3950167, -11.0544987, -2.7624040, 2.7740898
2: 6.3521647, 9.3099308, 6.4066296, 9.3040142, -2.7171741, 2.6715169
3: -5.2216492, -2.4667296, -5.2118092, -2.5546496, -2.6669996, 2.7450795
4: -11.1413088, -7.9268584, -11.1238718, -7.9606109, -2.8487682, 2.8522725
5: -10.7188530, -7.9241290, -10.7155809, -7.9956589, -2.3532205, 2.4127660
6: -13.7081127, -9.6142759, -13.6023312, -9.5642166, -3.4109077, 3.2944341
7: -4.3611021, -1.8353025, -4.3414249, -1.8684764, -2.2855425, 2.3099201
8: -2.2541590, 0.1941023, -2.1575828, 0.1959438, -2.4033298, 2.3516850
9: -9.3647861, -6.3369546, -9.3674107, -6.3416414, -2.4042730, 2.4181149

Time for backsubstitution: 14.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6184

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4644180, upper bound: 1.4788038
time: 4.40 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4679563, upper bound: 1.4790132
time: 5.02 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -9.1746111, -5.8598051, -9.0913048, -5.9002004, -2.8680048, 2.8256297
1: -14.4910002, -11.0261936, -14.3597946, -11.0607595, -2.7787099, 2.7444606
2: 6.3232241, 9.3151550, 6.4355822, 9.2987061, -2.7297039, 2.6482334
3: -5.2276049, -2.4510739, -5.2057600, -2.5703738, -2.6572311, 2.7546861
4: -11.1550627, -7.9222469, -11.1100454, -7.9652205, -2.8631115, 2.8359950
5: -10.7290783, -7.9215631, -10.7053633, -7.9982147, -2.3614845, 2.4076986
6: -13.7222748, -9.5588684, -13.5881920, -9.6196489, -3.3944149, 3.3204999
7: -4.3688560, -1.8302293, -4.3338032, -1.8736289, -2.2886739, 2.3071377
8: -2.3144000, 0.2082934, -2.0978289, 0.1817565, -2.4219341, 2.3061223
9: -9.3729439, -6.3014030, -9.3592806, -6.3771639, -2.3780508, 2.4444494

Time for backsubstitution: 14.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6184

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4715966, upper bound: 1.4715916
time: 4.11 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4751671, upper bound: 1.4717998
time: 4.42 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -9.1746111, -5.8598051, -9.0940342, -5.8776674, -2.8897629, 2.8274293
1: -14.4910002, -11.0261936, -14.3950167, -11.0544987, -2.7877312, 2.7675858
2: 6.3232241, 9.3151550, 6.4066296, 9.3040142, -2.7336364, 2.6620221
3: -5.2276049, -2.4510739, -5.2118092, -2.5546496, -2.6729553, 2.7607353
4: -11.1550627, -7.9222469, -11.1238718, -7.9606109, -2.8990374, 2.8853698
5: -10.7290783, -7.9215631, -10.7155809, -7.9956589, -2.3602824, 2.4153342
6: -13.7222748, -9.5588684, -13.6023312, -9.5642166, -3.4205542, 3.3142190
7: -4.3688560, -1.8302293, -4.3414249, -1.8684764, -2.3223152, 2.3470285
8: -2.3144000, 0.2082934, -2.1575828, 0.1959438, -2.4282842, 2.3470030
9: -9.3729439, -6.3014030, -9.3674107, -6.3416414, -2.3999605, 2.4400570

Time for backsubstitution: 14.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6184

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4715970, upper bound: 1.4715916
time: 4.17 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4751675, upper bound: 1.4717998
time: 4.14 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -9.1718369, -5.8823404, -9.1718369, -5.8823404, -2.8869090, 2.8869095
1: -14.4553471, -11.0323219, -14.4553471, -11.0323219, -2.7958593, 2.7958589
2: 6.3521647, 9.3099308, 6.3521647, 9.3099308, -2.7179937, 2.7179933
3: -5.2216492, -2.4667296, -5.2216492, -2.4667296, -2.7549195, 2.7549195
4: -11.1413088, -7.9268584, -11.1413088, -7.9268584, -2.8721385, 2.8721383
5: -10.7188530, -7.9241290, -10.7188530, -7.9241290, -2.4207869, 2.4207869
6: -13.7081127, -9.6142759, -13.7081127, -9.6142759, -3.3941383, 3.3941391
7: -4.3611021, -1.8353025, -4.3611021, -1.8353025, -2.3224578, 2.3224576
8: -2.2541590, 0.1941023, -2.2541590, 0.1941023, -2.4113173, 2.4113173
9: -9.3647861, -6.3369546, -9.3647861, -6.3369546, -2.4189157, 2.4189160

Time for backsubstitution: 14.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6184

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4644181, upper bound: 1.4715926
time: 4.43 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4679563, upper bound: 1.4718010
time: 4.14 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -9.1718369, -5.8823404, -9.1746111, -5.8598051, -2.9020567, 2.8886743
1: -14.4553471, -11.0323219, -14.4910002, -11.0261936, -2.7998328, 2.8162284
2: 6.3521647, 9.3099308, 6.3232241, 9.3151550, -2.7234578, 2.7465143
3: -5.2216492, -2.4667296, -5.2276049, -2.4510739, -2.7705753, 2.7608752
4: -11.1413088, -7.9268584, -11.1550627, -7.9222469, -2.8770370, 2.8913572
5: -10.7188530, -7.9241290, -10.7290783, -7.9215631, -2.4233379, 2.4286685
6: -13.7081127, -9.6142759, -13.7222748, -9.5588684, -3.4172091, 3.4006784
7: -4.3611021, -1.8353025, -4.3688560, -1.8302293, -2.3283129, 2.3311672
8: -2.2541590, 0.1941023, -2.3144000, 0.2082934, -2.4173732, 2.4359937
9: -9.3647861, -6.3369546, -9.3729439, -6.3014030, -2.4527426, 2.4277928

Time for backsubstitution: 14.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6184

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4644180, upper bound: 1.4788025
time: 4.39 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4679563, upper bound: 1.4790125
time: 4.60 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -9.1746111, -5.8598051, -9.1718369, -5.8823404, -2.8886743, 2.9020562
1: -14.4910002, -11.0261936, -14.4553471, -11.0323219, -2.8162284, 2.7998328
2: 6.3232241, 9.3151550, 6.3521647, 9.3099308, -2.7465143, 2.7234588
3: -5.2276049, -2.4510739, -5.2216492, -2.4667296, -2.7608752, 2.7705753
4: -11.1550627, -7.9222469, -11.1413088, -7.9268584, -2.8913569, 2.8770373
5: -10.7290783, -7.9215631, -10.7188530, -7.9241290, -2.4286685, 2.4233375
6: -13.7222748, -9.5588684, -13.7081127, -9.6142759, -3.4006786, 3.4172082
7: -4.3688560, -1.8302293, -4.3611021, -1.8353025, -2.3311667, 2.3283122
8: -2.3144000, 0.2082934, -2.2541590, 0.1941023, -2.4359937, 2.4173732
9: -9.3729439, -6.3014030, -9.3647861, -6.3369546, -2.4277925, 2.4527426

Time for backsubstitution: 14.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6184

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4715966, upper bound: 1.4715901
time: 4.38 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4751671, upper bound: 1.4717991
time: 4.34 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -9.1746111, -5.8598051, -9.1746111, -5.8598051, -2.9104261, 2.9104254
1: -14.4910002, -11.0261936, -14.4910002, -11.0261936, -2.8251595, 2.8251595
2: 6.3232241, 9.3151550, 6.3232241, 9.3151550, -2.7373896, 2.7373896
3: -5.2276049, -2.4510739, -5.2276049, -2.4510739, -2.7765310, 2.7765310
4: -11.1550627, -7.9222469, -11.1550627, -7.9222469, -2.9272766, 2.9272766
5: -10.7290783, -7.9215631, -10.7290783, -7.9215631, -2.4303846, 2.4303842
6: -13.7222748, -9.5588684, -13.7222748, -9.5588684, -3.4267993, 3.4267991
7: -4.3688560, -1.8302293, -4.3688560, -1.8302293, -2.3656797, 2.3656797
8: -2.3144000, 0.2082934, -2.3144000, 0.2082934, -2.4423285, 2.4423285
9: -9.3729439, -6.3014030, -9.3729439, -6.3014030, -2.4496808, 2.4496808

Time for backsubstitution: 14.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6184

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4715970, upper bound: 1.4715902
time: 4.06 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4751675, upper bound: 1.4717984
time: 4.33 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 23.31 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.31
Output dim: 2, lower bound: -1.4644240, upper bound: 1.4677554
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.31
Output dim: 2, lower bound: -1.4679622, upper bound: 1.4679642
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.31
Output dim: 2, lower bound: -1.4644239, upper bound: 1.4749643
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.31
Output dim: 2, lower bound: -1.4679622, upper bound: 1.4751736
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.31
Output dim: 2, lower bound: -1.4716027, upper bound: 1.4677540
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.31
Output dim: 2, lower bound: -1.4751730, upper bound: 1.4679623
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.31
Output dim: 2, lower bound: -1.4716031, upper bound: 1.4677540
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.31
Output dim: 2, lower bound: -1.4751735, upper bound: 1.4679622
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.31
Output dim: 2, lower bound: -1.4682614, upper bound: 1.4677498
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.31
Output dim: 2, lower bound: -1.4717996, upper bound: 1.4679584
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.31
Output dim: 2, lower bound: -1.4682614, upper bound: 1.4749584
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.31
Output dim: 2, lower bound: -1.4717996, upper bound: 1.4751678
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.31
Output dim: 2, lower bound: -1.4754457, upper bound: 1.4677481
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.31
Output dim: 2, lower bound: -1.4790126, upper bound: 1.4679564
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.31
Output dim: 2, lower bound: -1.4754460, upper bound: 1.4677481
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.31
Output dim: 2, lower bound: -1.4790130, upper bound: 1.4679564
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.31
Output dim: 2, lower bound: -1.4644181, upper bound: 1.4715933
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.31
Output dim: 2, lower bound: -1.4679563, upper bound: 1.4718017
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.31
Output dim: 2, lower bound: -1.4644180, upper bound: 1.4788038
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.31
Output dim: 2, lower bound: -1.4679563, upper bound: 1.4790132
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.31
Output dim: 2, lower bound: -1.4715966, upper bound: 1.4715916
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.31
Output dim: 2, lower bound: -1.4751671, upper bound: 1.4717998
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.31
Output dim: 2, lower bound: -1.4715970, upper bound: 1.4715916
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.31
Output dim: 2, lower bound: -1.4751675, upper bound: 1.4717998
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.31
Output dim: 2, lower bound: -1.4644181, upper bound: 1.4715926
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.31
Output dim: 2, lower bound: -1.4679563, upper bound: 1.4718010
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.31
Output dim: 2, lower bound: -1.4644180, upper bound: 1.4788025
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.31
Output dim: 2, lower bound: -1.4679563, upper bound: 1.4790125
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.31
Output dim: 2, lower bound: -1.4715966, upper bound: 1.4715901
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.31
Output dim: 2, lower bound: -1.4751671, upper bound: 1.4717991
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.31
Output dim: 2, lower bound: -1.4715970, upper bound: 1.4715902
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.31
Output dim: 2, lower bound: -1.4751675, upper bound: 1.4717984

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -9.0763741, -5.9151163, -9.0873947, -5.9023066, -2.7659988, 2.7594414
1: -14.3388357, -11.0680933, -14.3551044, -11.0621185, -2.6750751, 2.6848469
2: 6.4572568, 9.2881012, 6.4402657, 9.2972298, -2.5996060, 2.6057882
3: -5.1934748, -2.5897779, -5.2024417, -2.5731673, -2.6203074, 2.6126637
4: -11.0762119, -7.9895926, -11.1059017, -7.9721966, -2.7397623, 2.7676876
5: -10.6879616, -8.0060501, -10.7012825, -7.9990125, -2.3174214, 2.3238220
6: -13.5373802, -9.6772423, -13.5857506, -9.6356678, -3.2041025, 3.2096109
7: -4.3037243, -1.9087962, -4.3319731, -1.8832761, -2.2076302, 2.2203293
8: -2.0411971, 0.1265268, -2.0820923, 0.1779847, -2.2191818, 2.2086191
9: -9.3468189, -6.3863430, -9.3566113, -6.3785028, -2.3444786, 2.3452280

Time for backsubstitution: 14.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6184

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4644240, upper bound: 1.4644238
time: 4.29 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4644240, upper bound: 1.4677558
time: 4.30 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -9.0912952, -5.9002075, -9.0913029, -5.9002018, -2.7865129, 2.7778401
1: -14.3597765, -11.0607643, -14.3597908, -11.0607605, -2.6976199, 2.7036686
2: 6.4356031, 9.2987013, 6.4355903, 9.2987061, -2.6293488, 2.6198664
3: -5.2057519, -2.5703812, -5.2057586, -2.5703738, -2.6353781, 2.6353774
4: -11.1100311, -7.9652405, -11.1100426, -7.9652243, -2.7876186, 2.7927823
5: -10.7053547, -7.9982185, -10.7053642, -7.9982171, -2.3301706, 2.3362584
6: -13.5881834, -9.6196985, -13.5881910, -9.6196585, -3.2529411, 3.2161384
7: -4.3337994, -1.8736566, -4.3338022, -1.8736360, -2.2504930, 2.2461104
8: -2.0977821, 0.1817436, -2.0978203, 0.1817536, -2.2455769, 2.2774076
9: -9.3592720, -6.3771706, -9.3592777, -6.3771667, -2.3588877, 2.3712449

Time for backsubstitution: 14.74 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=2.6658577919006348
rel_dist={2: [-1.4790870249746977, 1.4790871137342352]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4639
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4639

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0555285, upper bound: 1.0504969
time: 4.63 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0555282, upper bound: 1.0555271
time: 4.49 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 9.30 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 9.30
Output dim: 2, lower bound: -1.0555285, upper bound: 1.0504969
IS_A2, status: Status.UNKNOWN, split count: 1, time: 9.30
Output dim: 2, lower bound: -1.0555282, upper bound: 1.0555271

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -9.0940342, -5.8776608, -9.0963173, -5.8753343, -2.4622397, 2.4620786
1: -14.3950214, -11.0544968, -14.3965416, -11.0441065, -2.3970313, 2.3885317
2: 6.4066234, 9.3040161, 6.4048781, 9.3080320, -2.4528890, 2.4505739
3: -5.2118101, -2.5546470, -5.2136140, -2.5530245, -2.5060449, 2.5066454
4: -11.1238737, -7.9606109, -11.1295519, -7.9597182, -2.5053072, 2.5097816
5: -10.7155838, -7.9956589, -10.7184896, -7.9950051, -2.1078148, 2.1113539
6: -13.6023312, -9.5642033, -13.6051750, -9.5625849, -2.9468727, 2.9479275
7: -4.3414259, -1.8684752, -4.3419952, -1.8644050, -2.0924296, 2.0889139
8: -2.1576006, 0.1959448, -2.1592803, 0.2029767, -2.1755223, 2.1701262
9: -9.3674126, -6.3416343, -9.3684855, -6.3403826, -2.1173277, 2.1172960

Time for backsubstitution: 14.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4639

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0504955, upper bound: 1.0504976
time: 4.26 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0504956, upper bound: 1.0504953
time: 4.86 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -9.1746120, -5.8598013, -9.0963058, -5.8753428, -2.5400829, 2.4821334
1: -14.4910040, -11.0261898, -14.3965349, -11.0441704, -2.4494376, 2.4278955
2: 6.3232183, 9.3151550, 6.4048896, 9.3080044, -2.5382977, 2.4683700
3: -5.2276077, -2.4510708, -5.2136040, -2.5530334, -2.5252209, 2.6134405
4: -11.1550636, -7.9222455, -11.1295176, -7.9597225, -2.5525808, 2.5432236
5: -10.7290831, -7.9215641, -10.7184725, -7.9950113, -2.1343861, 2.1782751
6: -13.7222738, -9.5588531, -13.6051636, -9.5625916, -3.0592546, 2.9546270
7: -4.3688583, -1.8302293, -4.3419929, -1.8644251, -2.1100249, 2.1366191
8: -2.3144202, 0.2082949, -2.1592720, 0.2029467, -2.2689390, 2.1953993
9: -9.3729439, -6.3013988, -9.3684769, -6.3403883, -2.1231012, 2.1663473

Time for backsubstitution: 14.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4616

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0555199, upper bound: 1.0513745
time: 5.01 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0555199, upper bound: 1.0555187
time: 4.55 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 24.43 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 24.43
Output dim: 2, lower bound: -1.0504955, upper bound: 1.0504976
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 24.43
Output dim: 2, lower bound: -1.0504956, upper bound: 1.0504953
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 24.43
Output dim: 2, lower bound: -1.0555199, upper bound: 1.0513745
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 24.43
Output dim: 2, lower bound: -1.0555199, upper bound: 1.0555187

## BFS IS instance: IS_A1_B1

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

Time for backsubstitution: 14.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4616

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0463580, upper bound: 1.0504859
time: 4.36 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0504887, upper bound: 1.0504869
time: 4.42 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -9.0940342, -5.8776608, -9.1740541, -5.8600497, -2.4795599, 2.5368609
1: -14.3950214, -11.0544968, -14.4906120, -11.0263929, -2.4234085, 2.4388266
2: 6.4066234, 9.3040161, 6.3238883, 9.3150768, -2.4655623, 2.5335693
3: -5.2118101, -2.5546470, -5.2275581, -2.4522831, -2.6107149, 2.5239143
4: -11.1238737, -7.9606109, -11.1547117, -7.9222574, -2.5377383, 2.5499892
5: -10.7155838, -7.9956589, -10.7289677, -7.9221430, -2.1733131, 2.1212459
6: -13.6023312, -9.5642033, -13.7209873, -9.5588531, -2.9517360, 3.0566635
7: -4.3414259, -1.8684752, -4.3688583, -1.8304369, -2.1299744, 2.1060758
8: -2.1576006, 0.1959448, -2.3126140, 0.2082953, -2.1801443, 2.2595029
9: -9.3674126, -6.3416343, -9.3729429, -6.3019805, -2.1611161, 2.1216407

Time for backsubstitution: 14.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4616

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0463580, upper bound: 1.0504866
time: 4.74 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0504887, upper bound: 1.0504868
time: 5.07 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -9.1741018, -5.8672438, -9.0935764, -5.8978868, -2.5168095, 2.4717221
1: -14.4790821, -11.0269394, -14.3612976, -11.0504322, -2.4188452, 2.3909302
2: 6.3328276, 9.3145542, 6.4338560, 9.3026991, -2.5132794, 2.4389367
3: -5.2262764, -2.4560356, -5.2075586, -2.5687597, -2.5067577, 2.5984783
4: -11.1511269, -7.9236879, -11.1156845, -7.9643316, -2.5416799, 2.5200181
5: -10.7257490, -7.9221272, -10.7082529, -7.9975681, -2.1283293, 2.1670008
6: -13.7204027, -9.5774002, -13.5910206, -9.6180363, -3.0000443, 2.9123411
7: -4.3667436, -1.8316127, -4.3343620, -1.8695784, -2.0985255, 2.1235895
8: -2.2942121, 0.2065802, -2.0994973, 0.1887527, -2.2080035, 2.1326938
9: -9.3721085, -6.3133826, -9.3603411, -6.3759170, -2.0868382, 2.1410303

Time for backsubstitution: 14.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6184

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0532803, upper bound: 1.0508181
time: 4.79 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0555159, upper bound: 1.0513706
time: 4.49 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -9.1746120, -5.8598013, -9.0963058, -5.8753481, -2.5387197, 2.4821324
1: -14.4910040, -11.0261898, -14.3965302, -11.0441704, -2.4456830, 2.4126425
2: 6.3232183, 9.3151550, 6.4048958, 9.3080044, -2.5351419, 2.4507627
3: -5.2276077, -2.4510708, -5.2136021, -2.5530362, -2.5252156, 2.6343181
4: -11.1550636, -7.9222455, -11.1295137, -7.9597244, -2.5766301, 2.5395133
5: -10.7290831, -7.9215641, -10.7184677, -7.9950109, -2.1343861, 2.1738839
6: -13.7222738, -9.5588531, -13.6051607, -9.5626030, -3.0151367, 2.9546251
7: -4.3688583, -1.8302293, -4.3419919, -1.8644258, -2.1281915, 2.1338120
8: -2.3144202, 0.2082949, -2.1592536, 0.2029438, -2.2603889, 2.1445816
9: -9.3729439, -6.3013988, -9.3684769, -6.3403912, -2.1027689, 2.1652474

Time for backsubstitution: 14.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4616

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0513756, upper bound: 1.0555211
time: 4.91 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0513756, upper bound: 1.0555184
time: 4.86 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 24.65 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 24.65
Output dim: 2, lower bound: -1.0463580, upper bound: 1.0504859
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 24.65
Output dim: 2, lower bound: -1.0504887, upper bound: 1.0504869
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 24.65
Output dim: 2, lower bound: -1.0463580, upper bound: 1.0504866
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 24.65
Output dim: 2, lower bound: -1.0504887, upper bound: 1.0504868
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 24.65
Output dim: 2, lower bound: -1.0532803, upper bound: 1.0508181
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 24.65
Output dim: 2, lower bound: -1.0555159, upper bound: 1.0513706
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 24.65
Output dim: 2, lower bound: -1.0513756, upper bound: 1.0555211
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 24.65
Output dim: 2, lower bound: -1.0513756, upper bound: 1.0555184

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -9.0913048, -5.9002004, -9.0935478, -5.8851228, -2.4493732, 2.4366107
1: -14.3597946, -11.0607595, -14.3832035, -11.0552778, -2.3498487, 2.3682432
2: 6.4355822, 9.2987061, 6.4162388, 9.3033848, -2.4194031, 2.4337788
3: -5.2057600, -2.5703738, -5.2104511, -2.5596383, -2.4896088, 2.4862790
4: -11.1100454, -7.9652205, -11.1199188, -7.9620519, -2.4811792, 2.4934275
5: -10.7053633, -7.9982147, -10.7122488, -7.9962139, -2.0956187, 2.1008725
6: -13.5881920, -9.6196489, -13.6004839, -9.5827560, -2.9031420, 2.8862185
7: -4.3338032, -1.8736289, -4.3393517, -1.8698833, -2.0754123, 2.0768819
8: -2.0978289, 0.1817565, -2.1375675, 0.1942315, -2.1048303, 2.1185062
9: -9.3592806, -6.3771639, -9.3665857, -6.3536129, -2.0905466, 2.0796103

Time for backsubstitution: 14.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6184

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0458053, upper bound: 1.0482652
time: 4.67 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0463572, upper bound: 1.0504869
time: 4.43 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -9.0940342, -5.8776674, -9.0940342, -5.8776608, -2.4597907, 2.4584308
1: -14.3950167, -11.0544987, -14.3950214, -11.0544968, -2.3715854, 2.3868380
2: 6.4066296, 9.3040142, 6.4066234, 9.3040161, -2.4312334, 2.4488378
3: -5.2118092, -2.5546496, -5.2118101, -2.5546470, -2.5301666, 2.5047774
4: -11.1238718, -7.9606109, -11.1238737, -7.9606109, -2.5012736, 2.5281308
5: -10.7155809, -7.9956589, -10.7155838, -7.9956589, -2.1025620, 2.1069388
6: -13.6023312, -9.5642166, -13.6023312, -9.5642033, -2.9454060, 2.9014277
7: -4.3414249, -1.8684764, -4.3414259, -1.8684752, -2.0855975, 2.1104240
8: -2.1575828, 0.1959438, -2.1576006, 0.1959448, -2.1167307, 2.1675406
9: -9.3674107, -6.3416414, -9.3674126, -6.3416343, -2.1147680, 2.0955849

Time for backsubstitution: 14.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4616

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0504923, upper bound: 1.0463608
time: 4.90 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0504923, upper bound: 1.0504920
time: 4.92 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -9.0913048, -5.9002004, -9.1735458, -5.8674936, -2.4691496, 2.5135894
1: -14.3597946, -11.0607595, -14.4786930, -11.0271425, -2.3864498, 2.4082360
2: 6.4355822, 9.2987061, 6.3334961, 9.3144760, -2.4361334, 2.5085464
3: -5.2057600, -2.5703738, -5.2262278, -2.4572489, -2.5957527, 2.5054524
4: -11.1100454, -7.9652205, -11.1507730, -7.9237022, -2.5145121, 2.5390871
5: -10.7053633, -7.9982147, -10.7256336, -7.9227095, -2.1620398, 2.1151814
6: -13.5881920, -9.6196489, -13.7191124, -9.5774002, -2.9094310, 2.9974513
7: -4.3338032, -1.8736289, -4.3667421, -1.8318204, -2.1169815, 2.0945683
8: -2.0978289, 0.1817565, -2.2924075, 0.2065792, -2.1174450, 2.1985652
9: -9.3592806, -6.3771639, -9.3721046, -6.3139644, -2.1358027, 2.0853698

Time for backsubstitution: 14.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6184

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0508188, upper bound: 1.0482615
time: 4.42 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0513713, upper bound: 1.0504819
time: 4.68 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -9.0940342, -5.8776674, -9.1740541, -5.8600497, -2.4795580, 2.5354991
1: -14.3950167, -11.0544987, -14.4906120, -11.0263929, -2.4081554, 2.4350717
2: 6.4066296, 9.3040142, 6.3238883, 9.3150768, -2.4479556, 2.5304132
3: -5.2118092, -2.5546496, -5.2275581, -2.4522831, -2.6315928, 2.5239098
4: -11.1238718, -7.9606109, -11.1547117, -7.9222574, -2.5340276, 2.5740302
5: -10.7155809, -7.9956589, -10.7289677, -7.9221430, -2.1689224, 2.1212449
6: -13.6023312, -9.5642166, -13.7209873, -9.5588531, -2.9517341, 3.0125451
7: -4.3414249, -1.8684764, -4.3688583, -1.8304369, -2.1271658, 2.1242414
8: -2.1575828, 0.1959438, -2.3126140, 0.2082953, -2.1293325, 2.2509534
9: -9.3674107, -6.3416414, -9.3729429, -6.3019805, -2.1600170, 2.1013095

Time for backsubstitution: 14.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4616

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0555195, upper bound: 1.0463567
time: 4.39 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0555194, upper bound: 1.0504885
time: 4.67 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -9.1590481, -5.8820419, -9.0867825, -5.9015751, -2.4979038, 2.4467125
1: -14.4585056, -11.0342512, -14.3530626, -11.0527945, -2.3926759, 2.3709869
2: 6.3543797, 9.3040409, 6.4419966, 9.3001194, -2.4858384, 2.4160972
3: -5.2137542, -2.4753685, -5.2018037, -2.5736690, -2.4786921, 2.5633569
4: -11.1173964, -7.9480329, -11.1084528, -7.9764771, -2.4822721, 2.4857764
5: -10.7081785, -7.9299436, -10.7011662, -7.9989576, -2.1086931, 2.1501131
6: -13.6698122, -9.6349993, -13.5867538, -9.6459122, -2.9084120, 2.8463306
7: -4.3367863, -1.8668076, -4.3311472, -1.8863584, -2.0406089, 2.0821629
8: -2.2375064, 0.1518970, -2.0721002, 0.1821375, -2.1448326, 2.0428300
9: -9.3596153, -6.3225980, -9.3557110, -6.3782539, -2.0666656, 2.1206293

Time for backsubstitution: 14.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4639

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0482620, upper bound: 1.0508172
time: 4.67 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0482620, upper bound: 1.0508181
time: 4.99 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -9.1740913, -5.8672495, -9.0935707, -5.8978910, -2.5125141, 2.4664927
1: -14.4790640, -11.0269413, -14.3612890, -11.0504341, -2.4135456, 2.3925662
2: 6.3328457, 9.3145485, 6.4338655, 9.3026991, -2.5088243, 2.4319415
3: -5.2262673, -2.4560428, -5.2075539, -2.5687618, -2.5123892, 2.5757251
4: -11.1511154, -7.9237118, -11.1156759, -7.9643421, -2.5314550, 2.4979687
5: -10.7257414, -7.9221301, -10.7082481, -7.9975681, -2.1211643, 2.1624446
6: -13.7203941, -9.5774517, -13.5910196, -9.6180582, -2.9619493, 2.8423901
7: -4.3667374, -1.8316414, -4.3343582, -1.8695905, -2.0721421, 2.1021690
8: -2.2941494, 0.2065697, -2.0994730, 0.1887474, -2.1369991, 2.0969810
9: -9.3721018, -6.3133860, -9.3603382, -6.3759217, -2.0808945, 2.1438937

Time for backsubstitution: 14.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4639

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0504833, upper bound: 1.0513706
time: 4.53 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0504833, upper bound: 1.0513711
time: 5.02 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -9.1718369, -5.8823404, -9.0963058, -5.8753481, -2.5282292, 2.4595175
1: -14.4553471, -11.0323219, -14.3965302, -11.0441704, -2.4095750, 2.4214225
2: 6.3521647, 9.3099308, 6.4048958, 9.3080044, -2.5062170, 2.4628992
3: -5.2216492, -2.4667296, -5.2136021, -2.5530362, -2.5167203, 2.5916429
4: -11.1413088, -7.9268584, -11.1295137, -7.9597244, -2.5302229, 2.5346024
5: -10.7188530, -7.9241290, -10.7184677, -7.9950109, -2.1236134, 2.1709218
6: -13.7081127, -9.6142759, -13.6051607, -9.5626030, -3.0032206, 2.8974628
7: -4.3611021, -1.8353025, -4.3419919, -1.8644258, -2.0947590, 2.1247711
8: -2.2541590, 0.1941023, -2.1592536, 0.2029438, -2.1992326, 2.1395886
9: -9.3647861, -6.3369546, -9.3684769, -6.3403912, -2.1131253, 2.1300707

Time for backsubstitution: 14.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4639

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0463565, upper bound: 1.0555173
time: 4.37 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0463565, upper bound: 1.0555187
time: 4.86 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9.1746111, -5.8598051, -9.0963058, -5.8753481, -2.5377431, 2.4807715
1: -14.4910002, -11.0261936, -14.3965302, -11.0441704, -2.4326091, 2.4126430
2: 6.3232241, 9.3151550, 6.4048958, 9.3080044, -2.5200291, 2.4507613
3: -5.2276049, -2.4510739, -5.2136021, -2.5530362, -2.5506005, 2.6343153
4: -11.1550627, -7.9222469, -11.1295137, -7.9597244, -2.5766206, 2.5638072
5: -10.7290783, -7.9215631, -10.7184677, -7.9950109, -2.1300097, 2.1736741
6: -13.7222748, -9.5588684, -13.6051607, -9.5626030, -3.0135241, 2.9106455
7: -4.3688560, -1.8302293, -4.3419919, -1.8644258, -2.1281900, 2.1586242
8: -2.3144000, 0.2082934, -2.1592536, 0.2029438, -2.2163081, 2.1445806
9: -9.3729439, -6.3014030, -9.3684769, -6.3403912, -2.1027656, 2.1459637

Time for backsubstitution: 14.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4639

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0463586, upper bound: 1.0513744
time: 5.36 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0463565, upper bound: 1.0513752
time: 4.84 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 25.06 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 25.06
Output dim: 2, lower bound: -1.0458053, upper bound: 1.0482652
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.06
Output dim: 2, lower bound: -1.0463572, upper bound: 1.0504869
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.06
Output dim: 2, lower bound: -1.0504923, upper bound: 1.0463608
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.06
Output dim: 2, lower bound: -1.0504923, upper bound: 1.0504920
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 25.06
Output dim: 2, lower bound: -1.0508188, upper bound: 1.0482615
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.06
Output dim: 2, lower bound: -1.0513713, upper bound: 1.0504819
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.06
Output dim: 2, lower bound: -1.0555195, upper bound: 1.0463567
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.06
Output dim: 2, lower bound: -1.0555194, upper bound: 1.0504885
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 25.06
Output dim: 2, lower bound: -1.0482620, upper bound: 1.0508172
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.06
Output dim: 2, lower bound: -1.0482620, upper bound: 1.0508181
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.06
Output dim: 2, lower bound: -1.0504833, upper bound: 1.0513706
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.06
Output dim: 2, lower bound: -1.0504833, upper bound: 1.0513711
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 25.06
Output dim: 2, lower bound: -1.0463565, upper bound: 1.0555173
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.06
Output dim: 2, lower bound: -1.0463565, upper bound: 1.0555187
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.06
Output dim: 2, lower bound: -1.0463586, upper bound: 1.0513744
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.06
Output dim: 2, lower bound: -1.0463565, upper bound: 1.0513752

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -9.0845156, -5.9038849, -9.0785618, -5.8998909, -2.4244409, 2.4177556
1: -14.3515596, -11.0631189, -14.3625460, -11.0626478, -2.3299818, 2.3412910
2: 6.4437203, 9.2961235, 6.4378710, 9.2927933, -2.3965845, 2.4064307
3: -5.2000055, -2.5752833, -5.1980424, -2.5788074, -2.4545083, 2.4580772
4: -11.1028137, -7.9773650, -11.0861216, -7.9864025, -2.4469976, 2.4340172
5: -10.6982822, -7.9996023, -10.6947250, -8.0040503, -2.0800481, 2.0812454
6: -13.5839243, -9.6475229, -13.5498447, -9.6403818, -2.8371081, 2.8045282
7: -4.3305898, -1.8904124, -4.3094783, -1.9050618, -2.0342913, 2.0192003
8: -2.0704360, 0.1751423, -2.0809338, 0.1394787, -2.0202498, 2.0554063
9: -9.3546467, -6.3795052, -9.3540611, -6.3627553, -2.0701914, 2.0594301

Time for backsubstitution: 14.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 470

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0440983, upper bound: 1.0475228
time: 4.89 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0458037, upper bound: 1.0482637
time: 4.40 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -9.0912991, -5.9002037, -9.0935364, -5.8851314, -2.4441891, 2.4373288
1: -14.3597860, -11.0607595, -14.3831863, -11.0552845, -2.3514867, 2.3636823
2: 6.4355946, 9.2987051, 6.4162579, 9.3033810, -2.4124689, 2.4349623
3: -5.2057562, -2.5703764, -5.2104416, -2.5596461, -2.4716301, 2.4919167
4: -11.1100397, -7.9652281, -11.1199055, -7.9620748, -2.4688435, 2.4832723
5: -10.7053604, -7.9982147, -10.7122383, -7.9962187, -2.0956106, 2.0937085
6: -13.5881863, -9.6196718, -13.6004772, -9.5828056, -2.8331909, 2.8638568
7: -4.3338013, -1.8736434, -4.3393469, -1.8699131, -2.0540752, 2.0646479
8: -2.0978076, 0.1817503, -2.1375055, 0.1942201, -2.0780396, 2.0474706
9: -9.3592758, -6.3771677, -9.3665771, -6.3536158, -2.0967441, 2.0736768

Time for backsubstitution: 14.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 470

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0446297, upper bound: 1.0497247
time: 4.70 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0463556, upper bound: 1.0504880
time: 4.20 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -9.0940342, -5.8776674, -9.0913048, -5.9002004, -2.4371710, 2.4568329
1: -14.3950167, -11.0544987, -14.3597946, -11.0607595, -2.3802691, 2.3507366
2: 6.4066296, 9.3040142, 6.4355822, 9.2987061, -2.4433360, 2.4200864
3: -5.2118092, -2.5546496, -5.2057600, -2.5703738, -2.4846158, 2.4961574
4: -11.1238718, -7.9606109, -11.1100454, -7.9652205, -2.4963884, 2.4819679
5: -10.7155809, -7.9956589, -10.7053633, -7.9982147, -2.1044121, 2.0961504
6: -13.6023312, -9.5642166, -13.5881920, -9.6196489, -2.8882222, 2.9049926
7: -4.3414249, -1.8684764, -4.3338032, -1.8736289, -2.0766106, 2.0738506
8: -2.1575828, 0.1959438, -2.0978289, 0.1817565, -2.1206627, 2.1066930
9: -9.3674107, -6.3416414, -9.3592806, -6.3771639, -2.0796642, 2.1059275

Time for backsubstitution: 14.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6184

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0482661, upper bound: 1.0458039
time: 4.46 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0504877, upper bound: 1.0463576
time: 5.12 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -9.0940342, -5.8776674, -9.0940342, -5.8776674, -2.4584303, 2.4584293
1: -14.3950167, -11.0544987, -14.3950167, -11.0544987, -2.3715854, 2.3715849
2: 6.4066296, 9.3040142, 6.4066296, 9.3040142, -2.4312320, 2.4312320
3: -5.2118092, -2.5546496, -5.2118092, -2.5546496, -2.5301609, 2.5301614
4: -11.1238718, -7.9606109, -11.1238718, -7.9606109, -2.5281210, 2.5281217
5: -10.7155809, -7.9956589, -10.7155809, -7.9956589, -2.1025615, 2.1025615
6: -13.6023312, -9.5642166, -13.6023312, -9.5642166, -2.9014254, 2.9014254
7: -4.3414249, -1.8684764, -4.3414249, -1.8684764, -2.1104202, 2.1104207
8: -2.1575828, 0.1959438, -2.1575828, 0.1959438, -2.1167283, 2.1167288
9: -9.3674107, -6.3416414, -9.3674107, -6.3416414, -2.0955830, 2.0955827

Time for backsubstitution: 14.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6184

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0482666, upper bound: 1.0458039
time: 4.69 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0504882, upper bound: 1.0463585
time: 4.53 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -9.0845156, -5.9038849, -9.1584949, -5.8822927, -2.4441438, 2.4946871
1: -14.3515596, -11.0631189, -14.4581156, -11.0344543, -2.3665156, 2.3820496
2: 6.4437203, 9.2961235, 6.3550534, 9.3039608, -2.4132986, 2.4811106
3: -5.2000055, -2.5752833, -5.2137051, -2.4765823, -2.5606279, 2.4773786
4: -11.1028137, -7.9773650, -11.1170397, -7.9480438, -2.4802823, 2.4796793
5: -10.6982822, -7.9996023, -10.7080650, -7.9305239, -2.1451540, 2.0955091
6: -13.5839243, -9.6475229, -13.6685219, -9.6349993, -2.8434191, 2.9058175
7: -4.3305898, -1.8904124, -4.3367872, -1.8670166, -2.0756545, 2.0366473
8: -2.0704360, 0.1751423, -2.2357039, 0.1518965, -2.0328498, 2.1353970
9: -9.3546467, -6.3795052, -9.3596115, -6.3231764, -2.1154008, 2.0652020

Time for backsubstitution: 14.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 470

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0491124, upper bound: 1.0475182
time: 4.66 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0508172, upper bound: 1.0482600
time: 4.33 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -9.0912991, -5.9002037, -9.1735363, -5.8675013, -2.4639201, 2.5092983
1: -14.3597860, -11.0607595, -14.4786758, -11.0271406, -2.3880863, 2.4029355
2: 6.4355946, 9.2987051, 6.3335152, 9.3144684, -2.4291449, 2.5040898
3: -5.2057562, -2.5703764, -5.2262201, -2.4572577, -2.5729980, 2.5110888
4: -11.1100397, -7.9652281, -11.1507597, -7.9237223, -2.4924598, 2.5288615
5: -10.7053604, -7.9982147, -10.7256279, -7.9227123, -2.1574831, 2.1080165
6: -13.5881863, -9.6196718, -13.7191105, -9.5774517, -2.8394794, 2.9593549
7: -4.3338013, -1.8736434, -4.3667374, -1.8318514, -2.0955629, 2.0681837
8: -2.0978076, 0.1817503, -2.2923470, 0.2065692, -2.0907969, 2.1275644
9: -9.3592758, -6.3771677, -9.3720951, -6.3139682, -2.1413913, 2.0794249

Time for backsubstitution: 14.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 470

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0496569, upper bound: 1.0497196
time: 4.83 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0513697, upper bound: 1.0504803
time: 4.39 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -9.0940342, -5.8776674, -9.1712799, -5.8825922, -2.4569440, 2.5250072
1: -14.3950167, -11.0544987, -14.4549646, -11.0325251, -2.4169335, 2.3989506
2: 6.4066296, 9.3040142, 6.3528357, 9.3098507, -2.4600921, 2.5014858
3: -5.2118092, -2.5546496, -5.2215977, -2.4679422, -2.5889187, 2.5154126
4: -11.1238718, -7.9606109, -11.1409531, -7.9268703, -2.5291204, 2.5276270
5: -10.7155809, -7.9956589, -10.7187376, -7.9247074, -2.1659594, 2.1104565
6: -13.6023312, -9.5642166, -13.7068233, -9.6142750, -2.8945732, 3.0006223
7: -4.3414249, -1.8684764, -4.3611031, -1.8355104, -2.1181998, 2.0908089
8: -2.1575828, 0.1959438, -2.2523608, 0.1941018, -2.1334105, 2.1897995
9: -9.3674107, -6.3416414, -9.3647823, -6.3375349, -2.1248412, 2.1116652

Time for backsubstitution: 14.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6184

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0532794, upper bound: 1.0458002
time: 4.72 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0555149, upper bound: 1.0463527
time: 4.55 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -9.0940342, -5.8776674, -9.1740532, -5.8600578, -2.4781971, 2.5345230
1: -14.3950167, -11.0544987, -14.4906101, -11.0263939, -2.4081545, 2.4219980
2: 6.4066296, 9.3040142, 6.3238955, 9.3150759, -2.4479551, 2.5152993
3: -5.2118092, -2.5546496, -5.2275562, -2.4522867, -2.6315899, 2.5492945
4: -11.1238718, -7.9606109, -11.1547070, -7.9222593, -2.5583267, 2.5740209
5: -10.7155809, -7.9956589, -10.7289648, -7.9221458, -2.1687117, 2.1168680
6: -13.6023312, -9.5642166, -13.7209835, -9.5588684, -2.9077535, 3.0109320
7: -4.3414249, -1.8684764, -4.3688574, -1.8304386, -2.1519880, 2.1242399
8: -2.1575828, 0.1959438, -2.3125944, 0.2082944, -2.1293311, 2.2068789
9: -9.3674107, -6.3416414, -9.3729429, -6.3019838, -2.1407394, 2.1013074

Time for backsubstitution: 14.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6184

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0532799, upper bound: 1.0458002
time: 4.47 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0555154, upper bound: 1.0463526
time: 4.59 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -9.1584949, -5.8822927, -9.0845156, -5.9038849, -2.4946871, 2.4441442
1: -14.4581156, -11.0344543, -14.3515596, -11.0631189, -2.3820496, 2.3665152
2: 6.3550534, 9.3039608, 6.4437203, 9.2961235, -2.4811106, 2.4132986
3: -5.2137051, -2.4765823, -5.2000055, -2.5752833, -2.4773779, 2.5606279
4: -11.1170397, -7.9480438, -11.1028137, -7.9773650, -2.4796791, 2.4802821
5: -10.7080650, -7.9305239, -10.6982822, -7.9996023, -2.0955086, 2.1451540
6: -13.6685219, -9.6349993, -13.5839243, -9.6475229, -2.9058180, 2.8434191
7: -4.3367872, -1.8670166, -4.3305898, -1.8904124, -2.0366473, 2.0756545
8: -2.2357039, 0.1518965, -2.0704360, 0.1751423, -2.1353970, 2.0328496
9: -9.3596115, -6.3231764, -9.3546467, -6.3795052, -2.0652018, 2.1154008

Time for backsubstitution: 14.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4616

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0440838, upper bound: 1.0508181
time: 4.85 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0440838, upper bound: 1.0508184
time: 4.70 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -9.1590481, -5.8820419, -9.1650105, -5.8860154, -2.5159035, 2.5174127
1: -14.4585056, -11.0342512, -14.4471321, -11.0346117, -2.4199538, 2.4173093
2: 6.3543797, 9.3040409, 6.3602319, 9.3073835, -2.4957414, 2.4859719
3: -5.2137542, -2.4753685, -5.2158551, -2.4716823, -2.5844889, 2.5808899
4: -11.1173964, -7.9480329, -11.1341000, -7.9389977, -2.5049467, 2.5180209
5: -10.7081785, -7.9299436, -10.7117596, -7.9255257, -2.1612196, 2.1612058
6: -13.6698122, -9.6349993, -13.7038231, -9.6421471, -2.9131689, 2.9402113
7: -4.3367863, -1.8668076, -4.3578591, -1.8520737, -2.0799255, 2.0948608
8: -2.2375064, 0.1518970, -2.2266903, 0.1875167, -2.1508627, 2.1194289
9: -9.3596153, -6.3225980, -9.3601589, -6.3393078, -2.1145682, 2.1253366

Time for backsubstitution: 14.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4616

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0440838, upper bound: 1.0508186
time: 5.06 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0440838, upper bound: 1.0508190
time: 4.89 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -9.1735363, -5.8675013, -9.0912991, -5.9002037, -2.5092983, 2.4639201
1: -14.4786758, -11.0271406, -14.3597860, -11.0607595, -2.4029355, 2.3880868
2: 6.3335152, 9.3144684, 6.4355946, 9.2987051, -2.5040898, 2.4291449
3: -5.2262201, -2.4572577, -5.2057562, -2.5703764, -2.5110893, 2.5729976
4: -11.1507597, -7.9237223, -11.1100397, -7.9652281, -2.5288610, 2.4924598
5: -10.7256279, -7.9227123, -10.7053604, -7.9982147, -2.1080165, 2.1574826
6: -13.7191105, -9.5774517, -13.5881863, -9.6196718, -2.9593544, 2.8394799
7: -4.3667374, -1.8318514, -4.3338013, -1.8736434, -2.0681834, 2.0955629
8: -2.2923470, 0.2065692, -2.0978076, 0.1817503, -2.1275644, 2.0907969
9: -9.3720951, -6.3139682, -9.3592758, -6.3771677, -2.0794249, 2.1413913

Time for backsubstitution: 14.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4616

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0463543, upper bound: 1.0513695
time: 4.49 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0463543, upper bound: 1.0513703
time: 4.90 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -9.1740913, -5.8672495, -9.1718321, -5.8823452, -2.5305557, 2.5331516
1: -14.4790640, -11.0269413, -14.4553413, -11.0323238, -2.4407883, 2.4382865
2: 6.3328457, 9.3145485, 6.3521738, 9.3099279, -2.5216079, 2.5016446
3: -5.2262673, -2.4560428, -5.2216439, -2.4667332, -2.6053667, 2.5932527
4: -11.1511154, -7.9237118, -11.1413002, -7.9268680, -2.5481668, 2.5388615
5: -10.7257414, -7.9221301, -10.7188473, -7.9241276, -2.1732125, 2.1739955
6: -13.7203941, -9.5774517, -13.7081099, -9.6142988, -2.9666967, 2.9362307
7: -4.3667374, -1.8316414, -4.3611012, -1.8353144, -2.1115389, 2.1055892
8: -2.2941494, 0.2065697, -2.2541366, 0.1940961, -2.1430869, 2.1735840
9: -9.3721018, -6.3133860, -9.3647804, -6.3369575, -2.1287966, 2.1485357

Time for backsubstitution: 14.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4616

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0463543, upper bound: 1.0513711
time: 5.01 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0463543, upper bound: 1.0513715
time: 4.82 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -9.1712799, -5.8825922, -9.0940342, -5.8776674, -2.5250072, 2.4569435
1: -14.4549646, -11.0325251, -14.3950167, -11.0544987, -2.3989506, 2.4169335
2: 6.3528357, 9.3098507, 6.4066296, 9.3040142, -2.5014863, 2.4600921
3: -5.2215977, -2.4679422, -5.2118092, -2.5546496, -2.5154123, 2.5889187
4: -11.1409531, -7.9268703, -11.1238718, -7.9606109, -2.5276270, 2.5291207
5: -10.7187376, -7.9247074, -10.7155809, -7.9956589, -2.1104569, 2.1659594
6: -13.7068233, -9.6142750, -13.6023312, -9.5642166, -3.0006218, 2.8945723
7: -4.3611031, -1.8355104, -4.3414249, -1.8684764, -2.0908089, 2.1182008
8: -2.2523608, 0.1941018, -2.1575828, 0.1959438, -2.1897993, 2.1334102
9: -9.3647823, -6.3375349, -9.3674107, -6.3416414, -2.1116652, 2.1248415

Time for backsubstitution: 14.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6184

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0440821, upper bound: 1.0549836
time: 4.56 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0463524, upper bound: 1.0555139
time: 4.51 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -9.1718369, -5.8823404, -9.1746111, -5.8598051, -2.5459385, 2.5341654
1: -14.4553471, -11.0323219, -14.4910002, -11.0261936, -2.4368000, 2.4496052
2: 6.3521647, 9.3099308, 6.3232241, 9.3151550, -2.5095363, 2.5284758
3: -5.2216492, -2.4667296, -5.2276049, -2.4510739, -2.6177049, 2.6090231
4: -11.1413088, -7.9268584, -11.1550627, -7.9222469, -2.5529289, 2.5672495
5: -10.7188530, -7.9241290, -10.7290783, -7.9215631, -2.1773081, 2.1824069
6: -13.7081127, -9.6142759, -13.7222748, -9.5588684, -3.0080142, 2.9987621
7: -4.3611021, -1.8353025, -4.3688560, -1.8302293, -2.1341734, 2.1372952
8: -2.2541590, 0.1941023, -2.3144000, 0.2082934, -2.2052569, 2.2161171
9: -9.3647861, -6.3369546, -9.3729439, -6.3014030, -2.1504040, 2.1348152

Time for backsubstitution: 14.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6184

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0440821, upper bound: 1.0549836
time: 4.85 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0463524, upper bound: 1.0555151
time: 5.05 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -9.1740532, -5.8600578, -9.0940342, -5.8776674, -2.5345230, 2.4781957
1: -14.4906101, -11.0263939, -14.3950167, -11.0544987, -2.4219980, 2.4081545
2: 6.3238955, 9.3150759, 6.4066296, 9.3040142, -2.5153003, 2.4479547
3: -5.2275562, -2.4522867, -5.2118092, -2.5546496, -2.5492945, 2.6315908
4: -11.1547070, -7.9222593, -11.1238718, -7.9606109, -2.5740213, 2.5583272
5: -10.7289648, -7.9221458, -10.7155809, -7.9956589, -2.1168685, 2.1687107
6: -13.7209835, -9.5588684, -13.6023312, -9.5642166, -3.0109320, 2.9077539
7: -4.3688574, -1.8304386, -4.3414249, -1.8684764, -2.1242399, 2.1519876
8: -2.3125944, 0.2082944, -2.1575828, 0.1959438, -2.2068791, 2.1293311
9: -9.3729429, -6.3019838, -9.3674107, -6.3416414, -2.1013074, 2.1407392

Time for backsubstitution: 14.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6184

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0482619, upper bound: 1.0508173
time: 4.83 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0504832, upper bound: 1.0513707
time: 4.49 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -9.1746111, -5.8598051, -9.1746111, -5.8598051, -2.5557127, 2.5557122
1: -14.4910002, -11.0261936, -14.4910002, -11.0261936, -2.4590096, 2.4590092
2: 6.3232241, 9.3151550, 6.3232241, 9.3151550, -2.5208254, 2.5208254
3: -5.2276049, -2.4510739, -5.2276049, -2.4510739, -2.6516948, 2.6516953
4: -11.1550627, -7.9222469, -11.1550627, -7.9222469, -2.5992975, 2.5992975
5: -10.7290783, -7.9215631, -10.7290783, -7.9215631, -2.1837053, 2.1837053
6: -13.7222748, -9.5588684, -13.7222748, -9.5588684, -3.0182538, 3.0182538
7: -4.3688560, -1.8302293, -4.3688560, -1.8302293, -2.1676044, 2.1676044
8: -2.3144000, 0.2082934, -2.3144000, 0.2082934, -2.2223330, 2.2223327
9: -9.3729439, -6.3014030, -9.3729439, -6.3014030, -2.1506572, 2.1506572

Time for backsubstitution: 14.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6184

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0482619, upper bound: 1.0508181
time: 5.54 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0504832, upper bound: 1.0513712
time: 5.08 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 25.52 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.52
Output dim: 2, lower bound: -1.0440983, upper bound: 1.0475228
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.52
Output dim: 2, lower bound: -1.0458037, upper bound: 1.0482637
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.52
Output dim: 2, lower bound: -1.0446297, upper bound: 1.0497247
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.52
Output dim: 2, lower bound: -1.0463556, upper bound: 1.0504880
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.52
Output dim: 2, lower bound: -1.0482661, upper bound: 1.0458039
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.52
Output dim: 2, lower bound: -1.0504877, upper bound: 1.0463576
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.52
Output dim: 2, lower bound: -1.0482666, upper bound: 1.0458039
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.52
Output dim: 2, lower bound: -1.0504882, upper bound: 1.0463585
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.52
Output dim: 2, lower bound: -1.0491124, upper bound: 1.0475182
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.52
Output dim: 2, lower bound: -1.0508172, upper bound: 1.0482600
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.52
Output dim: 2, lower bound: -1.0496569, upper bound: 1.0497196
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.52
Output dim: 2, lower bound: -1.0513697, upper bound: 1.0504803
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.52
Output dim: 2, lower bound: -1.0532794, upper bound: 1.0458002
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.52
Output dim: 2, lower bound: -1.0555149, upper bound: 1.0463527
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.52
Output dim: 2, lower bound: -1.0532799, upper bound: 1.0458002
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.52
Output dim: 2, lower bound: -1.0555154, upper bound: 1.0463526
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.52
Output dim: 2, lower bound: -1.0440838, upper bound: 1.0508181
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.52
Output dim: 2, lower bound: -1.0440838, upper bound: 1.0508184
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.52
Output dim: 2, lower bound: -1.0440838, upper bound: 1.0508186
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.52
Output dim: 2, lower bound: -1.0440838, upper bound: 1.0508190
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.52
Output dim: 2, lower bound: -1.0463543, upper bound: 1.0513695
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.52
Output dim: 2, lower bound: -1.0463543, upper bound: 1.0513703
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.52
Output dim: 2, lower bound: -1.0463543, upper bound: 1.0513711
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.52
Output dim: 2, lower bound: -1.0463543, upper bound: 1.0513715
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.52
Output dim: 2, lower bound: -1.0440821, upper bound: 1.0549836
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.52
Output dim: 2, lower bound: -1.0463524, upper bound: 1.0555139
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.52
Output dim: 2, lower bound: -1.0440821, upper bound: 1.0549836
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.52
Output dim: 2, lower bound: -1.0463524, upper bound: 1.0555151
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.52
Output dim: 2, lower bound: -1.0482619, upper bound: 1.0508173
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.52
Output dim: 2, lower bound: -1.0504832, upper bound: 1.0513707
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.52
Output dim: 2, lower bound: -1.0482619, upper bound: 1.0508181
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.52
Output dim: 2, lower bound: -1.0504832, upper bound: 1.0513712

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -9.0714216, -5.9071040, -9.0720501, -5.9015236, -2.4098902, 2.4081721
1: -14.3335571, -11.0642328, -14.3535080, -11.0631981, -2.3090868, 2.3278351
2: 6.4520898, 9.2945728, 6.4420004, 9.2920179, -2.3858175, 2.3992176
3: -5.1964569, -2.5770805, -5.1963048, -2.5797031, -2.4466348, 2.4509935
4: -11.0778656, -7.9784298, -11.0736589, -7.9869289, -2.4185538, 2.4190581
5: -10.6953278, -8.0056715, -10.6932640, -8.0070753, -2.0727730, 2.0727940
6: -13.5810413, -9.6495323, -13.5483980, -9.6413670, -2.8323607, 2.8003583
7: -4.3298206, -1.8988621, -4.3091068, -1.9092230, -2.0288792, 2.0097518
8: -2.0693088, 0.1663647, -2.0803761, 0.1350660, -2.0152144, 2.0467706
9: -9.3527641, -6.3828239, -9.3531466, -6.3644094, -2.0650425, 2.0537009

Time for backsubstitution: 14.77 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=2.4546236991882324
rel_dist={2: [-1.0555928908856513, 1.0555926528901223]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4639
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4639

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9012838, upper bound: 0.8963065
time: 4.81 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9012838, upper bound: 0.9012825
time: 4.84 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 9.82 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 9.82
Output dim: 2, lower bound: -0.9012838, upper bound: 0.8963065
IS_A2, status: Status.UNKNOWN, split count: 1, time: 9.82
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

Time for backsubstitution: 14.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4639

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8963070, upper bound: 0.8963065
time: 5.01 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8963070, upper bound: 0.8963065
time: 4.73 seconds

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

Time for backsubstitution: 14.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4616

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9012773, upper bound: 0.8979989
time: 5.16 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9012773, upper bound: 0.9012759
time: 5.34 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 25.37 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 25.37
Output dim: 2, lower bound: -0.8963070, upper bound: 0.8963065
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 25.37
Output dim: 2, lower bound: -0.8963070, upper bound: 0.8963065
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 25.37
Output dim: 2, lower bound: -0.9012773, upper bound: 0.8979989
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 25.37
Output dim: 2, lower bound: -0.9012773, upper bound: 0.9012759

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -9.1718445, -5.8701496, -9.0935726, -5.8978910, -2.3960805, 2.3523564
1: -14.4744930, -11.0280342, -14.3612957, -11.0504475, -2.2939949, 2.2678976
2: 6.3381329, 9.3140526, 6.4338584, 9.3026943, -2.4371700, 2.3666124
3: -5.2257361, -2.4626620, -5.2075539, -2.5687599, -2.4347172, 2.5211966
4: -11.1485910, -7.9240847, -11.1156788, -7.9643326, -2.4298511, 2.4120793
5: -10.7244368, -7.9248667, -10.7082500, -7.9975686, -2.0448995, 2.0825710
6: -13.7142429, -9.5818348, -13.5910206, -9.6180372, -2.8607798, 2.7750978
7: -4.3662295, -1.8328803, -4.3343601, -1.8695825, -2.0329909, 2.0574117
8: -2.2812963, 0.2061644, -2.0994945, 0.1887479, -2.1278744, 2.0599072
9: -9.3718948, -6.3188515, -9.3603401, -6.3759193, -1.9889879, 2.0366659

Time for backsubstitution: 14.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6184

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8993569, upper bound: 0.8971982
time: 4.78 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9012743, upper bound: 0.8979960
time: 4.87 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -9.1724749, -5.8609247, -9.0963058, -5.8753510, -2.4180565, 2.3645482
1: -14.4892521, -11.0271034, -14.3965292, -11.0441856, -2.3228421, 2.2890592
2: 6.3262300, 9.3148012, 6.4048963, 9.3079987, -2.4606237, 2.3777184
3: -5.2273884, -2.4565163, -5.2136011, -2.5530376, -2.4534645, 2.5568860
4: -11.1534786, -7.9222965, -11.1295052, -7.9597263, -2.4649272, 2.4316413
5: -10.7285652, -7.9241676, -10.7184658, -7.9950104, -2.0518003, 2.0893626
6: -13.7165394, -9.5588541, -13.6051569, -9.5626049, -2.8741264, 2.8209462
7: -4.3688574, -1.8311636, -4.3419909, -1.8644320, -2.0621638, 2.0680227
8: -2.3063180, 0.2082953, -2.1592526, 0.2029386, -2.1833258, 2.0696266
9: -9.3729343, -6.3040028, -9.3684750, -6.3403931, -2.0031843, 2.0645616

Time for backsubstitution: 14.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4616

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8980025, upper bound: 0.9012753
time: 4.53 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8980006, upper bound: 0.9012755
time: 4.87 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 24.26 seconds
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 24.26
Output dim: 2, lower bound: -0.8993569, upper bound: 0.8971982
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 24.26
Output dim: 2, lower bound: -0.9012743, upper bound: 0.8979960
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 24.26
Output dim: 2, lower bound: -0.8980025, upper bound: 0.9012753
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 24.26
Output dim: 2, lower bound: -0.8980006, upper bound: 0.9012755

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -9.1567936, -5.8849554, -9.0855312, -5.9022703, -2.3763781, 2.3266983
1: -14.4538975, -11.0353355, -14.3514977, -11.0532379, -2.2671928, 2.2463202
2: 6.3596964, 9.3035345, 6.4435015, 9.2996311, -2.4091473, 2.3425388
3: -5.2132263, -2.4820347, -5.2007461, -2.5746045, -2.4055672, 2.4853528
4: -11.1148415, -7.9484301, -11.1070948, -7.9787283, -2.3701739, 2.3760133
5: -10.7068863, -7.9326854, -10.6998672, -7.9992166, -2.0249987, 2.0644565
6: -13.6636515, -9.6394243, -13.5859528, -9.6510677, -2.7647252, 2.7081399
7: -4.3362436, -1.8680747, -4.3305345, -1.8894637, -1.9740334, 2.0151625
8: -2.2245963, 0.1514382, -2.0670278, 0.1808815, -2.0634303, 1.9652286
9: -9.3594093, -6.3280640, -9.3548565, -6.3786945, -1.9682708, 2.0153003

Time for backsubstitution: 14.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4639

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8943798, upper bound: 0.8971982
time: 5.40 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8943779, upper bound: 0.8971986
time: 4.92 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -9.1718349, -5.8701544, -9.0935669, -5.8978949, -2.3908892, 2.3471379
1: -14.4744759, -11.0280352, -14.3612862, -11.0504475, -2.2883835, 2.2691960
2: 6.3381519, 9.3140478, 6.4338689, 9.3026924, -2.4319239, 2.3596177
3: -5.2257290, -2.4626722, -5.2075500, -2.5687652, -2.4388628, 2.4971793
4: -11.1485786, -7.9241071, -11.1156664, -7.9643459, -2.4196391, 2.3875594
5: -10.7244282, -7.9248700, -10.7082434, -7.9975696, -2.0373774, 2.0780101
6: -13.7142353, -9.5818834, -13.5910158, -9.6180668, -2.8226709, 2.7013593
7: -4.3662248, -1.8329101, -4.3343582, -1.8695992, -2.0066066, 2.0333872
8: -2.2812362, 0.2061529, -2.0994658, 0.1887403, -2.0530777, 2.0234220
9: -9.3718872, -6.3188543, -9.3603334, -6.3759208, -1.9824457, 2.0396686

Time for backsubstitution: 14.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4639

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8962975, upper bound: 0.8979974
time: 4.79 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8962975, upper bound: 0.8979967
time: 4.99 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -9.1697063, -5.8834715, -9.0963058, -5.8753510, -2.4071121, 2.3419328
1: -14.4536228, -11.0332441, -14.3965292, -11.0441856, -2.2866850, 2.2985897
2: 6.3551769, 9.3095703, 6.4048963, 9.3079987, -2.4317083, 2.3907347
3: -5.2214246, -2.4721742, -5.2136011, -2.5530376, -2.4449635, 2.5152266
4: -11.1397104, -7.9269094, -11.1295052, -7.9597263, -2.4198284, 2.4267356
5: -10.7183380, -7.9267321, -10.7184658, -7.9950104, -2.0410256, 2.0863390
6: -13.7024250, -9.6142759, -13.6051569, -9.5626049, -2.8619804, 2.7637835
7: -4.3611021, -1.8362405, -4.3419909, -1.8644320, -2.0298448, 2.0589852
8: -2.2460923, 0.1941013, -2.1592526, 0.2029386, -2.1221833, 2.0639057
9: -9.3647747, -6.3395576, -9.3684750, -6.3403931, -2.0155559, 2.0293908

Time for backsubstitution: 14.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4639

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8930253, upper bound: 0.9012747
time: 4.67 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8930253, upper bound: 0.9012755
time: 4.91 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9.1724739, -5.8609304, -9.0963058, -5.8753510, -2.4170818, 2.3631177
1: -14.4892502, -11.0271072, -14.3965292, -11.0441856, -2.3090024, 2.2890573
2: 6.3262367, 9.3148003, 6.4048963, 9.3079987, -2.4446268, 2.3777194
3: -5.2273874, -2.4565196, -5.2136011, -2.5530376, -2.4778414, 2.5568836
4: -11.1534767, -7.9222980, -11.1295052, -7.9597263, -2.4649186, 2.4546416
5: -10.7285624, -7.9241686, -10.7184658, -7.9950104, -2.0472050, 2.0891514
6: -13.7165394, -9.5588694, -13.6051569, -9.5626049, -2.8725109, 2.7747636
7: -4.3688574, -1.8311660, -4.3419909, -1.8644320, -2.0621619, 2.0917273
8: -2.3062997, 0.2082930, -2.1592526, 0.2029386, -2.1366482, 2.0696251
9: -9.3729324, -6.3040085, -9.3684750, -6.3403931, -2.0031815, 2.0432711

Time for backsubstitution: 14.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4639

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8930233, upper bound: 0.8980003
time: 4.50 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8930233, upper bound: 0.8979995
time: 5.16 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 24.54 seconds
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 24.54
Output dim: 2, lower bound: -0.8943798, upper bound: 0.8971982
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 24.54
Output dim: 2, lower bound: -0.8943779, upper bound: 0.8971986
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.54
Output dim: 2, lower bound: -0.8962975, upper bound: 0.8979974
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.54
Output dim: 2, lower bound: -0.8962975, upper bound: 0.8979967
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.54
Output dim: 2, lower bound: -0.8930253, upper bound: 0.9012747
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.54
Output dim: 2, lower bound: -0.8930253, upper bound: 0.9012755
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.54
Output dim: 2, lower bound: -0.8930233, upper bound: 0.8980003
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.54
Output dim: 2, lower bound: -0.8930233, upper bound: 0.8979995

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -9.1670799, -5.8715720, -9.0912991, -5.9002066, -2.3836679, 2.3431859
1: -14.4721718, -11.0291767, -14.3597851, -11.0607615, -2.2758255, 2.2658439
2: 6.3422914, 9.3136015, 6.4355936, 9.2987032, -2.4238377, 2.3567209
3: -5.2254553, -2.4715049, -5.2057543, -2.5703781, -2.4369707, 2.4880044
4: -11.1462908, -7.9242034, -11.1100378, -7.9652309, -2.4149952, 2.3815618
5: -10.7237272, -7.9297199, -10.7053604, -7.9982181, -2.0246954, 2.0690413
6: -13.7055874, -9.5819445, -13.5881872, -9.6196785, -2.8120680, 2.6987753
7: -4.3661213, -1.8351593, -4.3338003, -1.8736471, -2.0025840, 2.0254102
8: -2.2708287, 0.2060814, -2.0978031, 0.1817498, -2.0364003, 2.0194983
9: -9.3717852, -6.3222041, -9.3592758, -6.3771677, -1.9808912, 2.0340014

Time for backsubstitution: 14.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4616

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8930220, upper bound: 0.8979951
time: 4.74 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8930220, upper bound: 0.8979954
time: 5.04 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -9.1739683, -5.8690290, -9.1718292, -5.8823447, -2.4113493, 2.4135771
1: -14.4762173, -11.0271177, -14.4553394, -11.0323238, -2.3173132, 2.3166091
2: 6.3351402, 9.3144035, 6.3521729, 9.3099260, -2.4474916, 2.4301729
3: -5.2259474, -2.4572244, -5.2216430, -2.4667351, -2.5312443, 2.5187807
4: -11.1501675, -7.9240556, -11.1412992, -7.9268699, -2.4389653, 2.4285595
5: -10.7249460, -7.9222641, -10.7188492, -7.9241285, -2.0902715, 2.0920095
6: -13.7199459, -9.5818834, -13.7081099, -9.6143055, -2.8322411, 2.7955842
7: -4.3662243, -1.8319743, -4.3610997, -1.8353181, -2.0459752, 2.0376549
8: -2.2893233, 0.2061534, -2.2541308, 0.1940956, -2.0653887, 2.1023190
9: -9.3718977, -6.3162503, -9.3647804, -6.3369570, -2.0302572, 2.0469794

Time for backsubstitution: 14.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4616

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8930220, upper bound: 0.8979960
time: 5.14 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8930220, upper bound: 0.8979994
time: 5.98 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -9.1649513, -5.8848972, -9.0940342, -5.8776674, -2.3998823, 2.3379812
1: -14.4513416, -11.0343895, -14.3950167, -11.0544987, -2.2741261, 2.2952256
2: 6.3593116, 9.3091202, 6.4066296, 9.3040142, -2.4236436, 2.3878179
3: -5.2211471, -2.4810133, -5.2118092, -2.5546496, -2.4430637, 2.5060589
4: -11.1374187, -7.9270029, -11.1238718, -7.9606109, -2.4151707, 2.4207602
5: -10.7176390, -7.9315848, -10.7155809, -7.9956589, -2.0283389, 2.0773702
6: -13.6937733, -9.6143379, -13.6023312, -9.5642166, -2.8513727, 2.7612371
7: -4.3610058, -1.8384943, -4.3414249, -1.8684764, -2.0258341, 2.0510321
8: -2.2357366, 0.1940317, -2.1575828, 0.1959438, -2.1055002, 2.0599885
9: -9.3646755, -6.3429031, -9.3674107, -6.3416414, -2.0140114, 2.0210752

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6184

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8911158, upper bound: 0.9004575
time: 4.69 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8930202, upper bound: 0.9012718
time: 4.79 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -9.1718369, -5.8823404, -9.1746111, -5.8598051, -2.4272299, 2.4159927
1: -14.4553471, -11.0323219, -14.4910002, -11.0261936, -2.3156714, 2.3272796
2: 6.3521647, 9.3099308, 6.3232241, 9.3151550, -2.4382291, 2.4557538
3: -5.2216492, -2.4667296, -5.2276049, -2.4510739, -2.5450816, 2.5366716
4: -11.1413088, -7.9268584, -11.1550627, -7.9222469, -2.4448929, 2.4592137
5: -10.7188530, -7.9241290, -10.7290783, -7.9215631, -2.0952988, 2.1002660
6: -13.7081127, -9.6142759, -13.7222748, -9.5588684, -2.8715968, 2.8647718
7: -4.3611021, -1.8353025, -4.3688560, -1.8302293, -2.0692320, 2.0723538
8: -2.2541590, 0.1941023, -2.3144000, 0.2082934, -2.1344390, 2.1427128
9: -9.3647861, -6.3369546, -9.3729439, -6.3014030, -2.0496187, 2.0371561

Time for backsubstitution: 14.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6184

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8911158, upper bound: 0.9004585
time: 4.84 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8930202, upper bound: 0.9012726
time: 5.06 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -9.1677208, -5.8623400, -9.0940342, -5.8776674, -2.4098530, 2.3591671
1: -14.4869308, -11.0282421, -14.3950167, -11.0544987, -2.2964268, 2.2857027
2: 6.3303757, 9.3143597, 6.4066296, 9.3040142, -2.4365335, 2.3748064
3: -5.2271194, -2.4653516, -5.2118092, -2.5546496, -2.4759521, 2.5477176
4: -11.1511927, -7.9223948, -11.1238718, -7.9606109, -2.4602404, 2.4486735
5: -10.7278614, -7.9290209, -10.7155809, -7.9956589, -2.0345340, 2.0801821
6: -13.7078953, -9.5589294, -13.6023312, -9.5642166, -2.8619146, 2.7722173
7: -4.3687510, -1.8334106, -4.3414249, -1.8684764, -2.0581546, 2.0837188
8: -2.2958450, 0.2082200, -2.1575828, 0.1959438, -2.1199837, 2.0567191
9: -9.3728313, -6.3073578, -9.3674107, -6.3416414, -2.0016370, 2.0349627

Time for backsubstitution: 14.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6184

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8943779, upper bound: 0.8971978
time: 4.57 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8962974, upper bound: 0.8979959
time: 5.16 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -9.1746111, -5.8598051, -9.1746111, -5.8598051, -2.4374723, 2.4374719
1: -14.4910002, -11.0261936, -14.4910002, -11.0261936, -2.3367906, 2.3367901
2: 6.3232241, 9.3151550, 6.3232241, 9.3151550, -2.4486370, 2.4486370
3: -5.2276049, -2.4510739, -5.2276049, -2.4510739, -2.5783296, 2.5783296
4: -11.1550627, -7.9222469, -11.1550627, -7.9222469, -2.4899712, 2.4899712
5: -10.7290783, -7.9215631, -10.7290783, -7.9215631, -2.1014795, 2.1014791
6: -13.7222748, -9.5588684, -13.7222748, -9.5588684, -2.8820539, 2.8820539
7: -4.3688560, -1.8302293, -4.3688560, -1.8302293, -2.1015491, 2.1015491
8: -2.3144000, 0.2082934, -2.3144000, 0.2082934, -2.1488886, 2.1488883
9: -9.3729439, -6.3014030, -9.3729439, -6.3014030, -2.0509825, 2.0509825

Time for backsubstitution: 14.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6184

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8943779, upper bound: 0.8971986
time: 5.04 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8962974, upper bound: 0.8979967
time: 5.22 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 25.13 seconds
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 25.13
Output dim: 2, lower bound: -0.8930220, upper bound: 0.8979951
IS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 25.13
Output dim: 2, lower bound: -0.8930220, upper bound: 0.8979954
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.13
Output dim: 2, lower bound: -0.8930220, upper bound: 0.8979960
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.13
Output dim: 2, lower bound: -0.8930220, upper bound: 0.8979994
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.13
Output dim: 2, lower bound: -0.8911158, upper bound: 0.9004575
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.13
Output dim: 2, lower bound: -0.8930202, upper bound: 0.9012718
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.13
Output dim: 2, lower bound: -0.8911158, upper bound: 0.9004585
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.13
Output dim: 2, lower bound: -0.8930202, upper bound: 0.9012726
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 25.13
Output dim: 2, lower bound: -0.8943779, upper bound: 0.8971978
IS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 25.13
Output dim: 2, lower bound: -0.8962974, upper bound: 0.8979959
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 25.13
Output dim: 2, lower bound: -0.8943779, upper bound: 0.8971986
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.13
Output dim: 2, lower bound: -0.8962974, upper bound: 0.8979967

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -9.1718264, -5.8823490, -9.1718292, -5.8823447, -2.4091549, 2.4050531
1: -14.4553337, -11.0323257, -14.4553394, -11.0323238, -2.3051963, 2.3111141
2: 6.3521786, 9.3099270, 6.3521729, 9.3099260, -2.4332457, 2.4255562
3: -5.2216382, -2.4667366, -5.2216430, -2.4667351, -2.5242023, 2.5088100
4: -11.1412945, -7.9268775, -11.1412992, -7.9268699, -2.4240279, 2.4245484
5: -10.7188444, -7.9241304, -10.7188492, -7.9241285, -2.0852246, 2.0901222
6: -13.7081041, -9.6143274, -13.7081099, -9.6143055, -2.8200760, 2.7845449
7: -4.3610983, -1.8353283, -4.3610997, -1.8353181, -2.0376215, 2.0307524
8: -2.2541103, 0.1940913, -2.2541308, 0.1940956, -2.0536695, 2.0900037
9: -9.3647776, -6.3369589, -9.3647804, -6.3369570, -2.0216885, 2.0339265

Time for backsubstitution: 14.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 470

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8922147, upper bound: 0.8964775
time: 4.67 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8930206, upper bound: 0.8979944
time: 4.88 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9.1746006, -5.8598156, -9.1718292, -5.8823447, -2.4108229, 2.4180546
1: -14.4909830, -11.0261927, -14.4553394, -11.0323238, -2.3215852, 2.3139591
2: 6.3232422, 9.3151503, 6.3521729, 9.3099260, -2.4506502, 2.4310045
3: -5.2275963, -2.4510798, -5.2216430, -2.4667351, -2.5281334, 2.5211167
4: -11.1550503, -7.9222684, -11.1412992, -7.9268699, -2.4426923, 2.4262524
5: -10.7290716, -7.9215655, -10.7188492, -7.9241285, -2.0920773, 2.0918417
6: -13.7222652, -9.5589199, -13.7081099, -9.6143055, -2.8267059, 2.7979102
7: -4.3688517, -1.8302602, -4.3610997, -1.8353181, -2.0459685, 2.0359213
8: -2.3143339, 0.2082834, -2.2541308, 0.1940956, -2.0679364, 2.0960600
9: -9.3729353, -6.3014088, -9.3647804, -6.3369570, -2.0306282, 2.0481701

Time for backsubstitution: 14.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 470

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8922147, upper bound: 0.8964788
time: 5.39 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8930206, upper bound: 0.8979941
time: 5.30 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -9.1499500, -5.8997736, -9.0859718, -5.8819914, -2.3804021, 2.3121104
1: -14.4305725, -11.0416851, -14.3851328, -11.0573053, -2.2469478, 2.2738357
2: 6.3809233, 9.2985783, 6.4162865, 9.3009510, -2.3954039, 2.3638191
3: -5.2087460, -2.5005660, -5.2049274, -2.5604794, -2.4135814, 2.4704165
4: -11.1036396, -7.9513736, -11.1153889, -7.9749775, -2.3554630, 2.3849394
5: -10.7002029, -7.9394045, -10.7071152, -7.9973001, -2.0085325, 2.0591979
6: -13.6430235, -9.6719036, -13.5973425, -9.5973167, -2.7551522, 2.6944571
7: -4.3308439, -1.8736745, -4.3378239, -1.8883622, -1.9667196, 2.0091500
8: -2.1790524, 0.1388721, -2.1250868, 0.1883421, -2.0412683, 1.9631619
9: -9.3522387, -6.3521233, -9.3619041, -6.3444190, -1.9932876, 1.9997301

Time for backsubstitution: 14.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 470

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8903124, upper bound: 0.8989409
time: 5.05 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8911143, upper bound: 0.9004560
time: 4.80 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -9.1649399, -5.8849030, -9.0940285, -5.8776703, -2.3949108, 2.3328238
1: -14.4513226, -11.0343914, -14.3950052, -11.0544987, -2.2685251, 2.2944469
2: 6.3593292, 9.3091135, 6.4066410, 9.3040113, -2.4183435, 2.3812160
3: -5.2211380, -2.4810197, -5.2118039, -2.5546553, -2.4476514, 2.4820666
4: -11.1374044, -7.9270253, -11.1238661, -7.9606228, -2.4050097, 2.3963981
5: -10.7176266, -7.9315872, -10.7155762, -7.9956594, -2.0208178, 2.0728002
6: -13.6937637, -9.6143875, -13.6023293, -9.5642452, -2.8132029, 2.6905556
7: -4.3610001, -1.8385199, -4.3414211, -1.8684936, -1.9994512, 2.0273581
8: -2.2356863, 0.1940203, -2.1575413, 0.1959381, -2.0310111, 2.0216098
9: -9.3646679, -6.3429070, -9.3674068, -6.3416438, -2.0058217, 2.0267951

Time for backsubstitution: 14.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 470

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8922129, upper bound: 0.8997281
time: 4.67 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8930188, upper bound: 0.9012708
time: 4.85 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -9.1568375, -5.8972330, -9.1664867, -5.8641472, -2.4077711, 2.3901315
1: -14.4345980, -11.0396385, -14.4810181, -11.0289154, -2.2885594, 2.3063784
2: 6.3737526, 9.2994070, 6.3328009, 9.3121300, -2.4098949, 2.4317989
3: -5.2092457, -2.4862969, -5.2206793, -2.4569533, -2.5159111, 2.5010924
4: -11.1075363, -7.9512300, -11.1466055, -7.9366074, -2.3851767, 2.4234748
5: -10.7014074, -7.9319439, -10.7205992, -7.9232140, -2.0754957, 2.0820742
6: -13.6573601, -9.6718397, -13.7172489, -9.5919619, -2.7753916, 2.7980399
7: -4.3309364, -1.8704838, -4.3652210, -1.8501024, -2.0099754, 2.0303988
8: -2.1974421, 0.1389418, -2.2818244, 0.2007222, -2.0702586, 2.0458989
9: -9.3523512, -6.3461876, -9.3674431, -6.3042021, -2.0289307, 2.0158074

Time for backsubstitution: 14.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 470

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8903124, upper bound: 0.8989411
time: 4.78 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8911163, upper bound: 0.9004564
time: 5.47 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -9.1718264, -5.8823490, -9.1746035, -5.8598108, -2.4223099, 2.4068193
1: -14.4553337, -11.0323257, -14.4909897, -11.0261936, -2.3099866, 2.3260918
2: 6.3521786, 9.3099270, 6.3232346, 9.3151531, -2.4387531, 2.4454317
3: -5.2216382, -2.4667366, -5.2275982, -2.4510767, -2.5369587, 2.5127306
4: -11.1412945, -7.9268775, -11.1550541, -7.9222593, -2.4257431, 2.4433591
5: -10.7188444, -7.9241304, -10.7290745, -7.9215641, -2.0877771, 2.0957003
6: -13.7081041, -9.6143274, -13.7222672, -9.5588999, -2.8334408, 2.7911496
7: -4.3610983, -1.8353283, -4.3688545, -1.8302475, -2.0428467, 2.0393806
8: -2.2541103, 0.1940913, -2.3143597, 0.2082882, -2.0599942, 2.1043344
9: -9.3647776, -6.3369589, -9.3729401, -6.3014040, -2.0411658, 2.0428643

Time for backsubstitution: 14.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 470

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8922129, upper bound: 0.8997283
time: 4.76 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8930188, upper bound: 0.9012716
time: 4.86 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9.1746006, -5.8598156, -9.1746035, -5.8598108, -2.4324541, 2.4282985
1: -14.4909830, -11.0261927, -14.4909897, -11.0261936, -2.3315763, 2.3367679
2: 6.3232422, 9.3151503, 6.3232346, 9.3151531, -2.4491367, 2.4417958
3: -5.2275963, -2.4510798, -5.2275982, -2.4510767, -2.5702400, 2.5543878
4: -11.1550503, -7.9222684, -11.1550541, -7.9222593, -2.4708910, 2.4715459
5: -10.7290716, -7.9215655, -10.7290745, -7.9215641, -2.0939569, 2.0986009
6: -13.7222652, -9.5589199, -13.7222672, -9.5588999, -2.8439875, 2.8084311
7: -4.3688517, -1.8302602, -4.3688545, -1.8302475, -2.0751643, 2.0685196
8: -2.3143339, 0.2082834, -2.3143597, 0.2082882, -2.0743809, 2.1105099
9: -9.3729353, -6.3014088, -9.3729401, -6.3014040, -2.0444217, 2.0566492

Time for backsubstitution: 14.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 470

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8954712, upper bound: 0.8964788
time: 5.03 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8962960, upper bound: 0.8979952
time: 5.40 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 25.34 seconds
IS_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 25.34
Output dim: 2, lower bound: -0.8922147, upper bound: 0.8964775
IS_A2_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 25.34
Output dim: 2, lower bound: -0.8930206, upper bound: 0.8979944
IS_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 25.34
Output dim: 2, lower bound: -0.8922147, upper bound: 0.8964788
IS_A2_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 25.34
Output dim: 2, lower bound: -0.8930206, upper bound: 0.8979941
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 25.34
Output dim: 2, lower bound: -0.8903124, upper bound: 0.8989409
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 25.34
Output dim: 2, lower bound: -0.8911143, upper bound: 0.9004560
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 25.34
Output dim: 2, lower bound: -0.8922129, upper bound: 0.8997281
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 25.34
Output dim: 2, lower bound: -0.8930188, upper bound: 0.9012708
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 25.34
Output dim: 2, lower bound: -0.8903124, upper bound: 0.8989411
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 25.34
Output dim: 2, lower bound: -0.8911163, upper bound: 0.9004564
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 25.34
Output dim: 2, lower bound: -0.8922129, upper bound: 0.8997283
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 25.34
Output dim: 2, lower bound: -0.8930188, upper bound: 0.9012716
IS_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 25.34
Output dim: 2, lower bound: -0.8954712, upper bound: 0.8964788
IS_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 25.34
Output dim: 2, lower bound: -0.8962960, upper bound: 0.8979952

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -9.1421099, -5.9017181, -9.0728798, -5.8852329, -2.3680615, 2.2972674
1: -14.4196901, -11.0423670, -14.3670654, -11.0584249, -2.2300930, 2.2524595
2: 6.3858709, 9.2976341, 6.4247046, 9.2993898, -2.3869219, 2.3527312
3: -5.2066560, -2.5016570, -5.2013917, -2.5622792, -2.4059162, 2.4618683
4: -11.0886202, -7.9520082, -11.0904465, -7.9760385, -2.3376665, 2.3563914
5: -10.6984634, -7.9430389, -10.7041492, -8.0033655, -1.9996448, 2.0505657
6: -13.6413078, -9.6730900, -13.5944595, -9.5993233, -2.7504287, 2.6894393
7: -4.3303776, -1.8786542, -4.3370805, -1.8967978, -1.9571462, 2.0027781
8: -2.1783864, 0.1335649, -2.1239598, 0.1795669, -2.0324998, 1.9567637
9: -9.3511448, -6.3541036, -9.3600388, -6.3477421, -1.9872484, 1.9941626

Time for backsubstitution: 14.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 470

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8896054, upper bound: 0.8989411
time: 5.23 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8896035, upper bound: 0.8989397
time: 5.66 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -9.1499434, -5.8997793, -9.0879898, -5.8703899, -2.3811722, 2.3085580
1: -14.4305611, -11.0416889, -14.3863344, -11.0467205, -2.2454677, 2.2673936
2: 6.3809266, 9.2985783, 6.4143419, 9.3037491, -2.3955946, 2.3648896
3: -5.2087440, -2.5005679, -5.2079821, -2.5577023, -2.4168415, 2.4708228
4: -11.1036243, -7.9513760, -11.1175251, -7.9559708, -2.3710408, 2.3762107
5: -10.7002010, -7.9394116, -10.7122164, -7.9963803, -2.0067635, 2.0600038
6: -13.6430244, -9.6719036, -13.5994692, -9.5942574, -2.7557683, 2.6958585
7: -4.3308439, -1.8736783, -4.3435645, -1.8863622, -1.9657383, 2.0149984
8: -2.1790519, 0.1388659, -2.1299341, 0.1893377, -2.0410252, 1.9651165
9: -9.3522396, -6.3521252, -9.3653927, -6.3437386, -1.9918098, 2.0020950

Time for backsubstitution: 14.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 470

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8896054, upper bound: 0.8996511
time: 5.43 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8896035, upper bound: 0.9004556
time: 5.15 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -9.1570940, -5.8867750, -9.0809383, -5.8808737, -2.3825431, 2.3180141
1: -14.4405022, -11.0350752, -14.3768177, -11.0556240, -2.2515659, 2.2726583
2: 6.3644075, 9.3081875, 6.4151545, 9.3024616, -2.4098125, 2.3698435
3: -5.2189817, -2.4820976, -5.2082205, -2.5564382, -2.4399891, 2.4734752
4: -11.1224127, -7.9276562, -11.0989008, -7.9616838, -2.3850408, 2.3676772
5: -10.7158737, -7.9352136, -10.7126102, -8.0017223, -2.0118980, 2.0641274
6: -13.6920452, -9.6155815, -13.5994453, -9.5662527, -2.8084822, 2.6855149
7: -4.3605361, -1.8435912, -4.3406858, -1.8769950, -1.9897227, 2.0210857
8: -2.2350216, 0.1887383, -2.1564116, 0.1871581, -2.0220280, 2.0150957
9: -9.3635578, -6.3448968, -9.3655415, -6.3449664, -1.9996295, 2.0212064

Time for backsubstitution: 14.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 470

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8915025, upper bound: 0.8997287
time: 5.07 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8915025, upper bound: 0.8997278
time: 5.49 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -9.1649303, -5.8849053, -9.0960531, -5.8660975, -2.3956900, 2.3292918
1: -14.4513121, -11.0343943, -14.3962650, -11.0439043, -2.2670488, 2.2871039
2: 6.3593321, 9.3091135, 6.4046478, 9.3068085, -2.4185367, 2.3823094
3: -5.2211361, -2.4810212, -5.2149019, -2.5518699, -2.4508743, 2.4824948
4: -11.1373882, -7.9270267, -11.1260891, -7.9416146, -2.4059029, 2.3876832
5: -10.7176266, -7.9315915, -10.7206669, -7.9947410, -2.0190492, 2.0735946
6: -13.6937666, -9.6143875, -13.6044369, -9.5611734, -2.8138208, 2.6919603
7: -4.3610001, -1.8385247, -4.3471727, -1.8664631, -1.9985061, 2.0324612
8: -2.2356882, 0.1940145, -2.1623960, 0.1969852, -2.0307789, 2.0235651
9: -9.3646660, -6.3429089, -9.3708763, -6.3409600, -2.0040393, 2.0291100

Time for backsubstitution: 14.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 470

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8915025, upper bound: 0.9004467
time: 4.52 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8915025, upper bound: 0.9012707
time: 5.62 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -9.1490078, -5.8991756, -9.1534128, -5.8673992, -2.3954654, 2.3753858
1: -14.4237261, -11.0403204, -14.4629154, -11.0300646, -2.2717152, 2.2849619
2: 6.3786983, 9.2984657, 6.3412247, 9.3105621, -2.4016590, 2.4209456
3: -5.2071571, -2.4873884, -5.2171440, -2.4587784, -2.5079880, 2.4925504
4: -11.0925179, -7.9518638, -11.1216221, -7.9376636, -2.3673778, 2.3949487
5: -10.6996727, -7.9355750, -10.7176628, -7.9292583, -2.0666056, 2.0734062
6: -13.6556616, -9.6730270, -13.7144337, -9.5939655, -2.7706809, 2.7929864
7: -4.3304672, -1.8754609, -4.3644667, -1.8584931, -2.0004349, 2.0235028
8: -2.1967831, 0.1336336, -2.2807205, 0.1919503, -2.0614123, 2.0394588
9: -9.3512554, -6.3481736, -9.3656025, -6.3075333, -2.0227594, 2.0102444

Time for backsubstitution: 14.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 470

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8896035, upper bound: 0.8989417
time: 4.83 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8896035, upper bound: 0.8989410
time: 5.24 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -9.1568279, -5.8972316, -9.1685486, -5.8525763, -2.4085441, 2.3866796
1: -14.4345865, -11.0396414, -14.4822769, -11.0183220, -2.2870789, 2.2990594
2: 6.3737545, 9.2994032, 6.3308096, 9.3149242, -2.4126873, 2.4330087
3: -5.2092457, -2.4862981, -5.2237344, -2.4541700, -2.5154920, 2.5014629
4: -11.1075230, -7.9512300, -11.1488018, -7.9176068, -2.3913136, 2.4147863
5: -10.7014046, -7.9319482, -10.7256794, -7.9222789, -2.0737829, 2.0828528
6: -13.6573582, -9.6718388, -13.7194538, -9.5889053, -2.7760086, 2.7989316
7: -4.3309355, -1.8704875, -4.3709688, -1.8480808, -2.0090361, 2.0311708
8: -2.1974421, 0.1389370, -2.2868087, 0.2017622, -2.0700526, 2.0478303
9: -9.3523521, -6.3461876, -9.3709164, -6.3035135, -2.0271482, 2.0181417

Time for backsubstitution: 14.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 470

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8896035, upper bound: 0.8996502
time: 4.81 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8896035, upper bound: 0.9004568
time: 5.02 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -9.1639900, -5.8842392, -9.1615400, -5.8630095, -2.4099798, 2.3920898
1: -14.4445152, -11.0330105, -14.4727678, -11.0273447, -2.2931376, 2.3045025
2: 6.3572578, 9.3090000, 6.3317599, 9.3135948, -2.4305663, 2.4342685
3: -5.2194824, -2.4678175, -5.2240191, -2.4528866, -2.5289984, 2.5041437
4: -11.1263008, -7.9275098, -11.1300497, -7.9233146, -2.4053240, 2.4146199
5: -10.7170906, -7.9277511, -10.7261353, -7.9276066, -2.0788541, 2.0869970
6: -13.7063990, -9.6155205, -13.7194538, -9.5608978, -2.8287234, 2.7860732
7: -4.3606339, -1.8403989, -4.3681064, -1.8387053, -2.0331006, 2.0325270
8: -2.2534499, 0.1888108, -2.3132529, 0.1995068, -2.0509644, 2.0977783
9: -9.3636684, -6.3389492, -9.3710976, -6.3047409, -2.0349650, 2.0372796

Time for backsubstitution: 14.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 470

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8915025, upper bound: 0.8997296
time: 4.74 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8915025, upper bound: 0.8997286
time: 5.84 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -9.1718178, -5.8823500, -9.1766977, -5.8482614, -2.4230947, 2.4033866
1: -14.4553194, -11.0323277, -14.4923038, -11.0155973, -2.3085117, 2.3188057
2: 6.3521862, 9.3099251, 6.3211975, 9.3179417, -2.4415412, 2.4466772
3: -5.2216353, -2.4667382, -5.2306685, -2.4482856, -2.5365219, 2.5131221
4: -11.1412792, -7.9268808, -11.1573391, -7.9032631, -2.4261799, 2.4346776
5: -10.7188396, -7.9241309, -10.7341442, -7.9206295, -2.0860634, 2.0964675
6: -13.7081051, -9.6143265, -13.7244530, -9.5558491, -2.8340569, 2.7920346
7: -4.3610983, -1.8353324, -4.3746099, -1.8281970, -2.0419364, 2.0401361
8: -2.2541103, 0.1940856, -2.3193498, 0.2093768, -2.0598159, 2.1062663
9: -9.3647766, -6.3369608, -9.3763981, -6.3007059, -2.0393882, 2.0445609

Time for backsubstitution: 14.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 470

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8915025, upper bound: 0.9004475
time: 4.48 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8915025, upper bound: 0.9012743
time: 5.45 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 24.84 seconds
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 24.84
Output dim: 2, lower bound: -0.8896054, upper bound: 0.8989411
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 24.84
Output dim: 2, lower bound: -0.8896035, upper bound: 0.8989397
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 24.84
Output dim: 2, lower bound: -0.8896054, upper bound: 0.8996511
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 24.84
Output dim: 2, lower bound: -0.8896035, upper bound: 0.9004556
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 24.84
Output dim: 2, lower bound: -0.8915025, upper bound: 0.8997287
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 24.84
Output dim: 2, lower bound: -0.8915025, upper bound: 0.8997278
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 24.84
Output dim: 2, lower bound: -0.8915025, upper bound: 0.9004467
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 24.84
Output dim: 2, lower bound: -0.8915025, upper bound: 0.9012707
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 24.84
Output dim: 2, lower bound: -0.8896035, upper bound: 0.8989417
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 24.84
Output dim: 2, lower bound: -0.8896035, upper bound: 0.8989410
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 24.84
Output dim: 2, lower bound: -0.8896035, upper bound: 0.8996502
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 24.84
Output dim: 2, lower bound: -0.8896035, upper bound: 0.9004568
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 24.84
Output dim: 2, lower bound: -0.8915025, upper bound: 0.8997296
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 24.84
Output dim: 2, lower bound: -0.8915025, upper bound: 0.8997286
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 24.84
Output dim: 2, lower bound: -0.8915025, upper bound: 0.9004475
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 24.84
Output dim: 2, lower bound: -0.8915025, upper bound: 0.9012743

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -9.1368637, -5.9030428, -9.0728798, -5.8852329, -2.3644185, 2.2960110
1: -14.4124289, -11.0428276, -14.3670654, -11.0584249, -2.2244229, 2.2508469
2: 6.3891659, 9.2969990, 6.4247046, 9.2993898, -2.3836145, 2.3517790
3: -5.2052703, -2.5023942, -5.2013917, -2.5622792, -2.4035769, 2.4604976
4: -11.0785885, -7.9524369, -11.0904465, -7.9760385, -2.3264647, 2.3559585
5: -10.6972885, -7.9454670, -10.7041492, -8.0033655, -1.9980140, 2.0485935
6: -13.6401873, -9.6738863, -13.5944595, -9.5993233, -2.7491660, 2.6884875
7: -4.3300657, -1.8819798, -4.3370805, -1.8967978, -1.9567881, 1.9990554
8: -2.1779366, 0.1300459, -2.1239598, 0.1795669, -2.0320916, 1.9540546
9: -9.3504028, -6.3554387, -9.3600388, -6.3477421, -1.9860373, 1.9924622

Time for backsubstitution: 14.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6184

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8896078, upper bound: 0.8978329
time: 4.94 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8896078, upper bound: 0.8989403
time: 5.17 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 25.00 seconds
IS_A2_B2_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 25.00
Output dim: 2, lower bound: -0.8896078, upper bound: 0.8978329
IS_A2_B2_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 25.00
Output dim: 2, lower bound: -0.8896078, upper bound: 0.8989403
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 25.00
Output dim: 2, lower bound: -0.8896035, upper bound: 0.8989397
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 25.00
Output dim: 2, lower bound: -0.8896054, upper bound: 0.8996511
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 25.00
Output dim: 2, lower bound: -0.8896035, upper bound: 0.9004556
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 25.00
Output dim: 2, lower bound: -0.8915025, upper bound: 0.8997287
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 25.00
Output dim: 2, lower bound: -0.8915025, upper bound: 0.8997278
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 25.00
Output dim: 2, lower bound: -0.8915025, upper bound: 0.9004467
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 25.00
Output dim: 2, lower bound: -0.8915025, upper bound: 0.9012707
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 25.00
Output dim: 2, lower bound: -0.8896035, upper bound: 0.8989417
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 25.00
Output dim: 2, lower bound: -0.8896035, upper bound: 0.8989410
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 25.00
Output dim: 2, lower bound: -0.8896035, upper bound: 0.8996502
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 25.00
Output dim: 2, lower bound: -0.8896035, upper bound: 0.9004568
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 25.00
Output dim: 2, lower bound: -0.8915025, upper bound: 0.8997296
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 25.00
Output dim: 2, lower bound: -0.8915025, upper bound: 0.8997286
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 25.00
Output dim: 2, lower bound: -0.8915025, upper bound: 0.9004475
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 25.00
Output dim: 2, lower bound: -0.8915025, upper bound: 0.9012743
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=2.384212017059326
rel_dist={2: [-0.9012986819155433, 0.9012981480281343]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2414.04 seconds
