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
execution time: IAR + LP analysis = 15.44 + 33.26 = 48.70 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3551.30 seconds, max iter: 100)

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
Binary search time: 201.90 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Relational Split (RS_dual_Z) starts
Time budget: 3349.40 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 6184
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 4639

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4790338, upper bound: 1.4751890
time: 4.46 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4751891, upper bound: 1.4790338
time: 4.73 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 9.41 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 9.41
Output dim: 2, lower bound: -1.4790338, upper bound: 1.4751890
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 9.41
Output dim: 2, lower bound: -1.4751891, upper bound: 1.4790338

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.8130879, 2.8134913
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.7654738, 2.7479062
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.6693897, 2.6631169
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.6605895, 2.6605895
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.8277750, 2.8359210
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.3513994, 2.3593178
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -3.3507934, 3.3480616
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.2869596, 2.2824697
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.3622570, 2.3622570
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.4118528, 2.4109721

Time for backsubstitution: 14.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6184
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6184

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4758859, upper bound: 1.4751823
time: 3.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4790269, upper bound: 1.4720377
time: 3.89 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.8132586, 2.8130884
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.7479062, 2.7555742
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.6631165, 2.6658573
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.6605895, 2.6605895
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.8313398, 2.8277745
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.3548641, 2.3513994
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -3.3480620, 3.3492546
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.2824697, 2.2844286
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.3622570, 2.3622570
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.4109716, 2.4113562

Time for backsubstitution: 14.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6184
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6184

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4720376, upper bound: 1.4790271
time: 4.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4751822, upper bound: 1.4758858
time: 3.93 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 22.99 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 22.99
Output dim: 2, lower bound: -1.4758859, upper bound: 1.4751823
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 22.99
Output dim: 2, lower bound: -1.4790269, upper bound: 1.4720377
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 22.99
Output dim: 2, lower bound: -1.4720376, upper bound: 1.4790271
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 22.99
Output dim: 2, lower bound: -1.4751822, upper bound: 1.4758858

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.8102756, 2.8170152
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.7686510, 2.7487459
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.6675205, 2.6667042
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.6605895, 2.6605895
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.8229880, 2.8139288
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.3428154, 2.3532352
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -3.2935925, 3.2673249
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.2737103, 2.2509770
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.3146157, 2.3243780
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.4197350, 2.4150908

Time for backsubstitution: 14.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 4616

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4686838, upper bound: 1.4751675
time: 4.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4758746, upper bound: 1.4679565
time: 4.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.8166118, 2.8106794
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.7663136, 2.7510824
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.6729774, 2.6612482
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.6605895, 2.6605895
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.8057818, 2.8311341
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.3453169, 2.3507342
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -3.2700558, 3.2908621
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.2554674, 2.2692204
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.3411350, 2.2978590
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.4159718, 2.4188542

Time for backsubstitution: 15.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 4616

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4717991, upper bound: 1.4720264
time: 4.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4790125, upper bound: 1.4648409
time: 6.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.8104463, 2.8166122
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.7510824, 2.7564135
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.6612482, 2.6694431
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.6605895, 2.6605895
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.8265510, 2.8057823
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.3462820, 2.3453174
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -3.2908621, 3.2685175
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.2692204, 2.2529354
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.2978592, 2.3316491
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.4188538, 2.4154751

Time for backsubstitution: 14.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 4616

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4648412, upper bound: 1.4790123
time: 4.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4720263, upper bound: 1.4717991
time: 4.21 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.8167815, 2.8102760
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.7487459, 2.7587504
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.6667042, 2.6639872
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.6605895, 2.6605895
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.8093467, 2.8229873
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.3487835, 2.3428159
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -3.2673254, 3.2920547
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.2509775, 2.2711787
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.3243780, 2.3051300
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.4150906, 2.4192386

Time for backsubstitution: 14.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 4616

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4679565, upper bound: 1.4758744
time: 4.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4751678, upper bound: 1.4686838
time: 4.49 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 23.53 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.53
Output dim: 2, lower bound: -1.4686838, upper bound: 1.4751675
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.53
Output dim: 2, lower bound: -1.4758746, upper bound: 1.4679565
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.53
Output dim: 2, lower bound: -1.4717991, upper bound: 1.4720264
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.53
Output dim: 2, lower bound: -1.4790125, upper bound: 1.4648409
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.53
Output dim: 2, lower bound: -1.4648412, upper bound: 1.4790123
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.53
Output dim: 2, lower bound: -1.4720263, upper bound: 1.4717991
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.53
Output dim: 2, lower bound: -1.4679565, upper bound: 1.4758744
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.53
Output dim: 2, lower bound: -1.4751678, upper bound: 1.4686838

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.8091249, 2.8153906
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.7503753, 2.7357845
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.6463776, 2.6517153
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.6605895, 2.6605895
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.8505540, 2.8324869
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.3375731, 2.3495073
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -3.2561560, 3.2145042
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.2989945, 2.2685299
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.2532191, 2.2812314
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.4054432, 2.3867102

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 903

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4600946, upper bound: 1.4750715
time: 4.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4685967, upper bound: 1.4665836
time: 4.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.8086519, 2.8158641
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.7556911, 2.7304697
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.6525431, 2.6455622
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.6605895, 2.6605895
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.8415456, 2.8415201
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.3390894, 2.3479929
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -3.2407722, 3.2299147
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.2912631, 2.2762780
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.2715268, 2.2629817
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.3913536, 2.4008183

Time for backsubstitution: 14.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 903

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4672997, upper bound: 1.4678680
time: 4.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4757758, upper bound: 1.4593687
time: 4.13 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.8154612, 2.8090549
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.7480378, 2.7381234
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.6518345, 2.6462708
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.6605895, 2.6605895
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.8333735, 2.8496921
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.3400745, 2.3470078
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -3.2326450, 3.2380414
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.2807679, 2.2867732
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.2797384, 2.2547698
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.4016991, 2.3904738

Time for backsubstitution: 14.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 903

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4632114, upper bound: 1.4719274
time: 4.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4717108, upper bound: 1.4634520
time: 4.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.8149881, 2.8095284
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.7533526, 2.7328072
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.6579876, 2.6401057
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.6605895, 2.6605895
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.8243403, 2.8587010
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.3415890, 2.3454914
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -3.2172356, 3.2534256
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.2730193, 2.2945051
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.2979884, 2.2364626
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.3875904, 2.4045625

Time for backsubstitution: 14.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 903

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4704287, upper bound: 1.4647540
time: 4.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4789194, upper bound: 1.4562518
time: 7.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.8092947, 2.8149877
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.7328076, 2.7434525
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.6401062, 2.6544557
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.6605895, 2.6605895
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.8541188, 2.8243403
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.3410387, 2.3415895
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -3.2534256, 3.2156968
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.2945046, 2.2704887
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.2364626, 2.2885029
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.4045620, 2.3870952

Time for backsubstitution: 14.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 903

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4562520, upper bound: 1.4789192
time: 4.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4647542, upper bound: 1.4704285
time: 4.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.8088217, 2.8154612
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.7381234, 2.7381377
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.6462708, 2.6483021
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.6605895, 2.6605895
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.8451104, 2.8333735
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.3425550, 2.3400745
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -3.2380409, 3.2311063
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.2867732, 2.2782369
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.2547698, 2.2702532
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.3904743, 2.4012034

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 903

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4634520, upper bound: 1.4717105
time: 4.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4719276, upper bound: 1.4632115
time: 4.22 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.8156309, 2.8086519
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.7304702, 2.7457914
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.6455622, 2.6490107
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.6605895, 2.6605895
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.8369374, 2.8415453
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.3435392, 2.3390894
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -3.2299137, 3.2392335
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.2762780, 2.2887321
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.2629819, 2.2620411
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.4008179, 2.3908587

Time for backsubstitution: 14.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 903

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4593688, upper bound: 1.4757756
time: 3.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4678682, upper bound: 1.4672995
time: 4.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.8151579, 2.8091249
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.7357850, 2.7404747
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.6517153, 2.6428461
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.6605895, 2.6605895
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.8279042, 2.8505542
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.3450537, 2.3375731
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -3.2145042, 3.2546177
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.2685294, 2.2964640
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.2812314, 2.2437339
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.3867102, 2.4049473

Time for backsubstitution: 14.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 903

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4665839, upper bound: 1.4685966
time: 4.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4750717, upper bound: 1.4600944
time: 4.09 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 23.27 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.27
Output dim: 2, lower bound: -1.4600946, upper bound: 1.4750715
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.27
Output dim: 2, lower bound: -1.4685967, upper bound: 1.4665836
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.27
Output dim: 2, lower bound: -1.4672997, upper bound: 1.4678680
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.27
Output dim: 2, lower bound: -1.4757758, upper bound: 1.4593687
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.27
Output dim: 2, lower bound: -1.4632114, upper bound: 1.4719274
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.27
Output dim: 2, lower bound: -1.4717108, upper bound: 1.4634520
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.27
Output dim: 2, lower bound: -1.4704287, upper bound: 1.4647540
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.27
Output dim: 2, lower bound: -1.4789194, upper bound: 1.4562518
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.27
Output dim: 2, lower bound: -1.4562520, upper bound: 1.4789192
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.27
Output dim: 2, lower bound: -1.4647542, upper bound: 1.4704285
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.27
Output dim: 2, lower bound: -1.4634520, upper bound: 1.4717105
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.27
Output dim: 2, lower bound: -1.4719276, upper bound: 1.4632115
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.27
Output dim: 2, lower bound: -1.4593688, upper bound: 1.4757756
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.27
Output dim: 2, lower bound: -1.4678682, upper bound: 1.4672995
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.27
Output dim: 2, lower bound: -1.4665839, upper bound: 1.4685966
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.27
Output dim: 2, lower bound: -1.4750717, upper bound: 1.4600944

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.8025217, 2.8107119
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.7492633, 2.7343512
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.6478634, 2.6539459
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.6605895, 2.6605895
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.8523178, 2.8348413
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.3352213, 2.3464732
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -3.2524891, 3.2121353
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.2985020, 2.2678871
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.2526131, 2.2804482
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.4087048, 2.3910673

Time for backsubstitution: 14.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 850

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4600944, upper bound: 1.4608561
time: 8.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4458710, upper bound: 1.4750717
time: 4.23 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.8044462, 2.8087873
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.7489409, 2.7346740
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.6486092, 2.6532001
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.6605895, 2.6605895
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.8529091, 2.8342500
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.3345394, 2.3471560
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -3.2537870, 3.2108374
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.2983522, 2.2680359
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.2524366, 2.2806249
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.4097996, 2.3899732

Time for backsubstitution: 14.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 850

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4685966, upper bound: 1.4523630
time: 8.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4543731, upper bound: 1.4665838
time: 4.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.8020487, 2.8111854
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.7545800, 2.7290363
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.6540279, 2.6477928
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.6605895, 2.6605895
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.8433084, 2.8438745
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.3367376, 2.3449588
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -3.2371054, 3.2275453
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.2907696, 2.2756352
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.2709203, 2.2621984
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.3946171, 2.4051752

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 850

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4672995, upper bound: 1.4536444
time: 4.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4530871, upper bound: 1.4678678
time: 4.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.8039732, 2.8092608
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.7542567, 2.7293596
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.6547737, 2.6470470
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.6605895, 2.6605895
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.8439007, 2.8432832
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.3360548, 2.3456411
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -3.2384024, 3.2262473
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.2906199, 2.2757845
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.2707434, 2.2623749
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.3957109, 2.4040813

Time for backsubstitution: 14.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 850

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4757756, upper bound: 1.4451424
time: 8.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4615647, upper bound: 1.4593687
time: 4.46 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.8088579, 2.8043761
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.7469268, 2.7366896
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.6533194, 2.6485014
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.6605895, 2.6605895
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.8351364, 2.8520465
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.3377228, 2.3439736
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -3.2289782, 3.2356725
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.2802744, 2.2861304
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.2791324, 2.2539866
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.4049616, 2.3948307

Time for backsubstitution: 14.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 850

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4632112, upper bound: 1.4577182
time: 5.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4489836, upper bound: 1.4719268
time: 4.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.8107824, 2.8024516
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.7466035, 2.7370129
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.6540651, 2.6477556
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.6605895, 2.6605895
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.8357277, 2.8514552
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.3370409, 2.3446565
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -3.2302752, 3.2343745
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.2801256, 2.2862797
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.2789555, 2.2541630
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.4060555, 2.3937366

Time for backsubstitution: 14.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 850

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4717106, upper bound: 1.4492411
time: 5.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4574872, upper bound: 1.4634520
time: 4.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.8083839, 2.8048496
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.7522407, 2.7313733
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.6594734, 2.6423364
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.6605895, 2.6605895
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.8261032, 2.8610554
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.3392382, 2.3424573
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -3.2135677, 3.2510567
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.2725258, 2.2938623
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.2973819, 2.2356794
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.3908539, 2.4089193

Time for backsubstitution: 14.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 850

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4704285, upper bound: 1.4505305
time: 4.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4562079, upper bound: 1.4647537
time: 7.46 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.8103085, 2.8029251
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.7519183, 2.7316961
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.6602182, 2.6415906
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.6605895, 2.6605895
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.8266945, 2.8604641
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.3385553, 2.3431401
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -3.2148657, 3.2497587
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.2723770, 2.2940116
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.2972054, 2.2358561
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.3919468, 2.4078252

Time for backsubstitution: 14.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 850

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4789192, upper bound: 1.4420301
time: 4.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4647006, upper bound: 1.4562517
time: 10.21 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.8026915, 2.8103089
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.7316966, 2.7420187
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.6415901, 2.6566858
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.6605895, 2.6605895
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.8558817, 2.8266947
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.3386869, 2.3385553
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -3.2497587, 3.2133279
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.2940111, 2.2698450
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.2358561, 2.2877197
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.4078255, 2.3914516

Time for backsubstitution: 14.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 850

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4562519, upper bound: 1.4647004
time: 4.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4420284, upper bound: 1.4789193
time: 4.23 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.8046160, 2.8083844
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.7313733, 2.7423420
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.6423359, 2.6559401
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.6605895, 2.6605895
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.8564730, 2.8261034
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.3380041, 2.3392377
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -3.2510567, 3.2120299
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.2938623, 2.2699947
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.2356796, 2.2878964
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.4089193, 2.3903575

Time for backsubstitution: 14.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 850

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4647540, upper bound: 1.4562077
time: 4.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4505305, upper bound: 1.4704285
time: 4.18 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.8022184, 2.8107824
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.7370124, 2.7367039
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.6477556, 2.6505327
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.6605895, 2.6605895
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.8468733, 2.8357279
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.3402033, 2.3370404
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -3.2343740, 3.2287378
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.2862797, 2.2775936
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.2541633, 2.2694700
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.3937359, 2.4055598

Time for backsubstitution: 14.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 850

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4634518, upper bound: 1.4574867
time: 4.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4492394, upper bound: 1.4717104
time: 4.46 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.8041430, 2.8088579
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.7366891, 2.7370272
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.6485014, 2.6497869
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.6605895, 2.6605895
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.8474646, 2.8351364
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.3395205, 2.3377233
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -3.2356720, 3.2274399
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.2861300, 2.2777429
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.2539868, 2.2696466
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.3948307, 2.4044657

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 850

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4719274, upper bound: 1.4489834
time: 4.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4577164, upper bound: 1.4632113
time: 4.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.8090277, 2.8039732
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.7293591, 2.7443571
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.6470470, 2.6512413
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.6605895, 2.6605895
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.8387012, 2.8438997
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.3411884, 2.3360553
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -3.2262468, 3.2368650
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.2757845, 2.2880888
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.2623754, 2.2612581
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.4040813, 2.3952150

Time for backsubstitution: 14.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 850

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4593686, upper bound: 1.4615649
time: 4.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4451426, upper bound: 1.4757755
time: 6.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.8109522, 2.8020487
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.7290359, 2.7446804
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.6477928, 2.6504955
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.6605895, 2.6605895
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.8392925, 2.8433084
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.3405066, 2.3367381
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -3.2275457, 3.2355671
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.2756357, 2.2882380
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.2621989, 2.2614346
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.4051752, 2.3941212

Time for backsubstitution: 14.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 850

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4678681, upper bound: 1.4530873
time: 4.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4536446, upper bound: 1.4672996
time: 4.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.8085546, 2.8044467
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.7346740, 2.7390413
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.6532001, 2.6450763
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.6605895, 2.6605895
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.8296680, 2.8529086
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.3427038, 2.3345394
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -3.2108374, 3.2522492
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.2680359, 2.2958207
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.2806253, 2.2429509
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.3899727, 2.4093037

Time for backsubstitution: 14.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 850

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4665837, upper bound: 1.4543729
time: 4.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4523630, upper bound: 1.4685966
time: 4.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.8104792, 2.8025217
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.7343507, 2.7393641
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.6539459, 2.6443305
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.6605895, 2.6605895
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.8302593, 2.8523173
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.3420210, 2.3352218
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -3.2121353, 3.2509513
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.2678871, 2.2959700
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.2804484, 2.2431273
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.3910675, 2.4082098

Time for backsubstitution: 14.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 850

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4750715, upper bound: 1.4458708
time: 4.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4608559, upper bound: 1.4600944
time: 4.20 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 23.30 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4600944, upper bound: 1.4608561
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4458710, upper bound: 1.4750717
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4685966, upper bound: 1.4523630
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4543731, upper bound: 1.4665838
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4672995, upper bound: 1.4536444
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4530871, upper bound: 1.4678678
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4757756, upper bound: 1.4451424
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4615647, upper bound: 1.4593687
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4632112, upper bound: 1.4577182
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4489836, upper bound: 1.4719268
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4717106, upper bound: 1.4492411
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4574872, upper bound: 1.4634520
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4704285, upper bound: 1.4505305
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4562079, upper bound: 1.4647537
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4789192, upper bound: 1.4420301
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4647006, upper bound: 1.4562517
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4562519, upper bound: 1.4647004
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4420284, upper bound: 1.4789193
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4647540, upper bound: 1.4562077
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4505305, upper bound: 1.4704285
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4634518, upper bound: 1.4574867
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4492394, upper bound: 1.4717104
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4719274, upper bound: 1.4489834
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4577164, upper bound: 1.4632113
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4593686, upper bound: 1.4615649
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4451426, upper bound: 1.4757755
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4678681, upper bound: 1.4530873
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4536446, upper bound: 1.4672996
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4665837, upper bound: 1.4543729
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4523630, upper bound: 1.4685966
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4750715, upper bound: 1.4458708
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4608559, upper bound: 1.4600944

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.7965980, 2.8015313
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.7533636, 2.7375073
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.6421280, 2.6438751
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.6605895, 2.6605895
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.8540688, 2.8360696
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.3097706, 2.3284369
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -3.2223530, 3.1908054
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.2964544, 2.2649946
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.2521276, 2.2798212
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.4076185, 2.3902256

Time for backsubstitution: 14.69 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=2.6658577919006348
rel_dist={2: [-1.4790870249746977, 1.4790871137342352]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 6184
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 4639

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0555282, upper bound: 1.0504953
time: 4.55 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0504956, upper bound: 1.0555279
time: 4.65 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 9.39 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 9.39
Output dim: 2, lower bound: -1.0555282, upper bound: 1.0504953
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 9.39
Output dim: 2, lower bound: -1.0504956, upper bound: 1.0555279

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4643559, 2.4645872
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.4010940, 2.3910556
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4554672, 2.4518828
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5078921, 2.5079112
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.5071583, 2.5118134
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.1087637, 2.1132884
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.9497619, 2.9482007
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0935440, 2.0909781
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.1804099, 2.1708338
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.1188755, 2.1183722

Time for backsubstitution: 14.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6184
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6184

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0538949, upper bound: 1.0504916
time: 4.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0555243, upper bound: 1.0488621
time: 4.38 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4645267, 2.4643564
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.3910556, 2.3987236
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4518833, 2.4546232
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5079074, 2.5078921
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.5107222, 2.5071583
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.1122284, 2.1087637
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.9482017, 2.9493933
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0909777, 2.0929370
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.1708341, 2.1781058
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.1183720, 2.1187561

Time for backsubstitution: 14.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6184
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6184

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0488623, upper bound: 1.0555241
time: 4.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0504917, upper bound: 1.0538963
time: 5.41 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 24.62 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.62
Output dim: 2, lower bound: -1.0538949, upper bound: 1.0504916
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.62
Output dim: 2, lower bound: -1.0555243, upper bound: 1.0488621
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.62
Output dim: 2, lower bound: -1.0488623, upper bound: 1.0555241
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.62
Output dim: 2, lower bound: -1.0504917, upper bound: 1.0538963

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4615436, 2.4653955
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.4032688, 2.3918943
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4535990, 2.4531322
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5080376, 2.5140052
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.4949965, 2.4898212
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.1001797, 2.1061344
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.8824739, 2.8674645
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0724764, 2.0594854
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0976090, 2.1031876
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.1251445, 2.1224906

Time for backsubstitution: 16.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 4616

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0497389, upper bound: 1.0504825
time: 4.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0538884, upper bound: 1.0463525
time: 4.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4651656, 2.4617748
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.4019337, 2.3932304
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4567165, 2.4500141
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5139866, 2.5080564
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.4851661, 2.4996529
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.1016092, 2.1047053
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.8690243, 2.8809142
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0620518, 2.0699100
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.1127629, 2.0880337
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.1229939, 2.1246414

Time for backsubstitution: 14.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 4616

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0513717, upper bound: 1.0488545
time: 4.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0555160, upper bound: 1.0447163
time: 4.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4617143, 2.4651651
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.3932304, 2.3995624
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4500141, 2.4558711
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5080519, 2.5139861
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.4985614, 2.4851661
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.1036463, 2.1016097
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.8809137, 2.8686571
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0699100, 2.0614438
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0880337, 2.1104584
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.1246414, 2.1228750

Time for backsubstitution: 14.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 4616

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0447144, upper bound: 1.0555157
time: 4.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0488558, upper bound: 1.0513716
time: 4.44 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4653344, 2.4615445
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.3918943, 2.4008980
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4531326, 2.4527531
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5139999, 2.5080373
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.4887300, 2.4949975
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.1050758, 2.1001801
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.8674641, 2.8821068
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0594854, 2.0718689
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.1031876, 2.0953047
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.1224904, 2.1250257

Time for backsubstitution: 14.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 4616

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0463525, upper bound: 1.0538871
time: 4.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0504834, upper bound: 1.0497388
time: 4.41 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 23.69 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.69
Output dim: 2, lower bound: -1.0497389, upper bound: 1.0504825
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.69
Output dim: 2, lower bound: -1.0538884, upper bound: 1.0463525
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.69
Output dim: 2, lower bound: -1.0513717, upper bound: 1.0488545
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.69
Output dim: 2, lower bound: -1.0555160, upper bound: 1.0447163
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.69
Output dim: 2, lower bound: -1.0447144, upper bound: 1.0555157
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.69
Output dim: 2, lower bound: -1.0488558, upper bound: 1.0513716
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.69
Output dim: 2, lower bound: -1.0463525, upper bound: 1.0538871
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.69
Output dim: 2, lower bound: -1.0504834, upper bound: 1.0497388

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4601908, 2.4637709
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.3849940, 2.3766565
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4324560, 2.4355059
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5293698, 2.5393631
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.5187039, 2.5083792
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0949373, 2.1017570
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.8384447, 2.8146439
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0944467, 2.0770378
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0362124, 2.0522194
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.1048145, 2.0941105

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 903

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0448934, upper bound: 1.0503337
time: 4.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0496011, upper bound: 1.0456468
time: 4.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4599199, 2.4640417
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.3880315, 2.3736191
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4359789, 2.4319897
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5334020, 2.5353374
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.5135560, 2.5135410
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0958033, 2.1008916
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.8296537, 2.8234491
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0900283, 2.0814657
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0466738, 2.0417910
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.0967641, 2.1021719

Time for backsubstitution: 14.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 903

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0490519, upper bound: 1.0462112
time: 4.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0537395, upper bound: 1.0415072
time: 4.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4638109, 2.4601507
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.3836579, 2.3779926
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4355736, 2.4323945
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5353189, 2.5334208
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.5088859, 2.5182109
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0963669, 2.1003284
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.8250093, 2.8280935
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0840316, 2.0874629
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0513663, 2.0370984
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.1026750, 2.0962608

Time for backsubstitution: 14.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 903

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0465262, upper bound: 1.0487056
time: 4.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0512380, upper bound: 1.0440182
time: 4.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4635401, 2.4604211
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.3866954, 2.3749547
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4390898, 2.4288716
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5393443, 2.5293887
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.5037236, 2.5233588
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0972319, 2.0994625
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.8162041, 2.8368845
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0796037, 2.0918813
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0617948, 2.0266373
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.0946136, 2.1043115

Time for backsubstitution: 14.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 903

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0506797, upper bound: 1.0445704
time: 4.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0553671, upper bound: 1.0398748
time: 4.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4603605, 2.4635410
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.3749547, 2.3843241
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4288721, 2.4382463
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5293832, 2.5393438
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.5222688, 2.5037241
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0984030, 2.0972328
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.8368845, 2.8158355
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0918813, 2.0789967
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0266376, 2.0594909
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.1043115, 2.0944953

Time for backsubstitution: 14.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 903

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0398736, upper bound: 1.0553668
time: 4.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0445703, upper bound: 1.0506793
time: 4.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4600897, 2.4638114
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.3779931, 2.3812871
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4323940, 2.4347301
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5334153, 2.5353184
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.5171208, 2.5088859
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0992689, 2.0963669
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.8280935, 2.8246417
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0874629, 2.0834246
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0370984, 2.0490625
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.0962605, 2.1025569

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 903

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0440194, upper bound: 1.0512381
time: 4.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0487069, upper bound: 1.0465262
time: 4.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4639807, 2.4599204
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.3736196, 2.3856602
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4319897, 2.4351349
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5353312, 2.5334015
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.5124497, 2.5135555
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0998316, 2.0958042
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.8234491, 2.8292851
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0814662, 2.0894217
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0417910, 2.0443699
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.1021719, 2.0966458

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 903

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0415071, upper bound: 1.0537383
time: 4.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0462114, upper bound: 1.0490514
time: 4.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4637098, 2.4601908
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.3766570, 2.3826227
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4355059, 2.4316120
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5393577, 2.5293696
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.5072885, 2.5187035
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.1006975, 2.0949378
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.8146439, 2.8380761
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0770373, 2.0938401
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0522194, 2.0339086
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.0941100, 2.1046963

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 903

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0456471, upper bound: 1.0496030
time: 4.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0503345, upper bound: 1.0448947
time: 4.46 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 23.62 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.62
Output dim: 2, lower bound: -1.0448934, upper bound: 1.0503337
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.62
Output dim: 2, lower bound: -1.0496011, upper bound: 1.0456468
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.62
Output dim: 2, lower bound: -1.0490519, upper bound: 1.0462112
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.62
Output dim: 2, lower bound: -1.0537395, upper bound: 1.0415072
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.62
Output dim: 2, lower bound: -1.0465262, upper bound: 1.0487056
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.62
Output dim: 2, lower bound: -1.0512380, upper bound: 1.0440182
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.62
Output dim: 2, lower bound: -1.0506797, upper bound: 1.0445704
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.62
Output dim: 2, lower bound: -1.0553671, upper bound: 1.0398748
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.62
Output dim: 2, lower bound: -1.0398736, upper bound: 1.0553668
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.62
Output dim: 2, lower bound: -1.0445703, upper bound: 1.0506793
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.62
Output dim: 2, lower bound: -1.0440194, upper bound: 1.0512381
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.62
Output dim: 2, lower bound: -1.0487069, upper bound: 1.0465262
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.62
Output dim: 2, lower bound: -1.0415071, upper bound: 1.0537383
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.62
Output dim: 2, lower bound: -1.0462114, upper bound: 1.0490514
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.62
Output dim: 2, lower bound: -1.0456471, upper bound: 1.0496030
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.62
Output dim: 2, lower bound: -1.0503345, upper bound: 1.0448947

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4535875, 2.4582677
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.3838820, 2.3753605
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4342604, 2.4377365
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5288811, 2.5387766
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.5204668, 2.5104804
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0925856, 2.0990157
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.8347769, 2.8117180
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0938892, 2.0763950
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0356064, 2.0515120
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.1080775, 2.0979981

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 850

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0448933, upper bound: 1.0422842
time: 4.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0368032, upper bound: 1.0503328
time: 4.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4546871, 2.4571676
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.3836970, 2.3755455
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4346867, 2.4373102
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5287838, 2.5388741
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.5208044, 2.5101423
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0921965, 2.0994058
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.8355188, 2.8109765
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0938044, 2.0764809
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0355053, 2.0516129
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.1087027, 2.0973730

Time for backsubstitution: 14.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 850

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0496010, upper bound: 1.0375977
time: 4.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0415109, upper bound: 1.0456459
time: 4.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4533167, 2.4585381
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.3869205, 2.3723240
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4377832, 2.4342203
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5329132, 2.5347509
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.5153189, 2.5156422
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0934525, 2.0981503
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.8259859, 2.8205237
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0894709, 2.0808229
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0460672, 2.0410836
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.1000266, 2.1060600

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 850

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0490518, upper bound: 1.0381474
time: 4.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0409616, upper bound: 1.0462113
time: 4.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4544163, 2.4574385
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.3867354, 2.3725080
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4382095, 2.4337940
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5328159, 2.5348487
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.5156574, 2.5153041
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0930624, 2.0985398
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.8267279, 2.8197823
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0893860, 2.0809083
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0459666, 2.0411844
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.1006517, 2.1054349

Time for backsubstitution: 14.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 850

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0537394, upper bound: 1.0334417
time: 4.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0456496, upper bound: 1.0415061
time: 4.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4572077, 2.4546471
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.3825469, 2.3766971
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4373779, 2.4346251
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5348301, 2.5328343
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.5106487, 2.5203118
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0940151, 2.0975871
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.8213425, 2.8251677
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0834742, 2.0868201
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0507603, 2.0363910
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.1059380, 2.1001487

Time for backsubstitution: 14.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 850

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0465252, upper bound: 1.0406161
time: 4.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0384608, upper bound: 1.0487062
time: 4.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4583073, 2.4535475
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.3823619, 2.3768816
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4378042, 2.4341993
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5347328, 2.5329320
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.5109873, 2.5199738
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0936251, 2.0979772
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.8220844, 2.8244262
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0833883, 2.0869055
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0506592, 2.0364919
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.1065631, 2.0995235

Time for backsubstitution: 15.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.25 seconds

### Candidate
type: RSZ, layer: 1, pos: 850

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0512370, upper bound: 1.0359288
time: 5.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0431727, upper bound: 1.0440186
time: 5.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4569378, 2.4549174
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.3855834, 2.3736591
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4408951, 2.4311023
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5388556, 2.5288022
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.5054874, 2.5254598
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0948811, 2.0967207
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.8125372, 2.8339586
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0790462, 2.0912385
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0611887, 2.0259297
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.0978761, 2.1081994

Time for backsubstitution: 15.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.26 seconds

### Candidate
type: RSZ, layer: 1, pos: 850

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0506786, upper bound: 1.0364821
time: 5.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0426304, upper bound: 1.0445701
time: 5.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4580374, 2.4538178
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.3853993, 2.3738441
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4413204, 2.4306765
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5387573, 2.5288999
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.5058250, 2.5251217
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0944910, 2.0971107
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.8132782, 2.8332171
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0789604, 2.0913239
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0610876, 2.0260305
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.0985012, 2.1075742

Time for backsubstitution: 15.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.25 seconds

### Candidate
type: RSZ, layer: 1, pos: 850

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0553661, upper bound: 1.0317842
time: 5.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0473180, upper bound: 1.0398736
time: 5.45 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4537573, 2.4580374
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.3738437, 2.3830285
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4306765, 2.4404764
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5288954, 2.5387573
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.5240316, 2.5058250
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0960512, 2.0944910
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.8332167, 2.8129106
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0913239, 2.0783539
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0260310, 2.0587835
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.1075740, 2.0983829

Time for backsubstitution: 15.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 850

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0398735, upper bound: 1.0473177
time: 4.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0317834, upper bound: 1.0553660
time: 4.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4548569, 2.4569378
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.3736587, 2.3832130
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4311028, 2.4400501
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5287981, 2.5388551
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.5243692, 2.5054872
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0956621, 2.0948811
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.8339586, 2.8121691
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0912390, 2.0784388
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0259304, 2.0588844
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.1081996, 2.0977573

Time for backsubstitution: 14.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 850

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0445702, upper bound: 1.0426303
time: 4.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0364800, upper bound: 1.0506779
time: 4.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4534864, 2.4583077
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.3768821, 2.3799915
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4341993, 2.4369602
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5329275, 2.5347319
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.5188837, 2.5109868
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0969172, 2.0936255
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.8244257, 2.8217163
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0869055, 2.0827813
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0364923, 2.0483551
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.0995235, 2.1064444

Time for backsubstitution: 14.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 850

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0440194, upper bound: 1.0431729
time: 4.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0359294, upper bound: 1.0512364
time: 7.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4545860, 2.4572082
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.3766971, 2.3801761
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4346256, 2.4365339
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5328293, 2.5348294
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.5192213, 2.5106490
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0965281, 2.0940156
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.8251677, 2.8209748
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0868196, 2.0828667
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0363913, 2.0484560
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.1001487, 2.1058192

Time for backsubstitution: 14.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 850

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0487068, upper bound: 1.0384610
time: 4.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0406170, upper bound: 1.0465252
time: 4.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4573774, 2.4544168
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.3725085, 2.3843651
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4337940, 2.4373651
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5348444, 2.5328150
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.5142136, 2.5156565
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0974808, 2.0930629
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.8197823, 2.8263602
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0809078, 2.0887785
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0411849, 2.0436625
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.1054349, 2.1005335

Time for backsubstitution: 14.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 850

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0415060, upper bound: 1.0456488
time: 4.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0334417, upper bound: 1.0537386
time: 4.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4584770, 2.4533172
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.3723235, 2.3845496
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4342203, 2.4369392
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5347462, 2.5329127
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.5145521, 2.5153186
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0970907, 2.0934525
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.8205233, 2.8256187
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0808229, 2.0888638
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0410838, 2.0437634
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.1060600, 2.0999079

Time for backsubstitution: 14.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 850

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0462104, upper bound: 1.0409614
time: 5.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0381479, upper bound: 1.0490516
time: 4.20 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4571066, 2.4546871
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.3755450, 2.3813272
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4373102, 2.4338422
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5388699, 2.5287831
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.5090513, 2.5208044
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0983458, 2.0921960
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.8109770, 2.8351512
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0764809, 2.0931969
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0516133, 2.0332012
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.0973730, 2.1085839

Time for backsubstitution: 14.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 850

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0456460, upper bound: 1.0415106
time: 4.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0375979, upper bound: 1.0496010
time: 4.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4582071, 2.4535875
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.3753600, 2.3815117
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4377365, 2.4334164
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5387726, 2.5288806
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.5093899, 2.5204666
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0979567, 2.0925860
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.8117180, 2.8344097
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0763950, 2.0932822
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0515122, 2.0333021
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.0979981, 2.1079588

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 850

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0503334, upper bound: 1.0368031
time: 4.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0422853, upper bound: 1.0448935
time: 4.49 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 24.23 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 2, lower bound: -1.0448933, upper bound: 1.0422842
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 2, lower bound: -1.0368032, upper bound: 1.0503328
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 2, lower bound: -1.0496010, upper bound: 1.0375977
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 2, lower bound: -1.0415109, upper bound: 1.0456459
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 2, lower bound: -1.0490518, upper bound: 1.0381474
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 2, lower bound: -1.0409616, upper bound: 1.0462113
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 2, lower bound: -1.0537394, upper bound: 1.0334417
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 2, lower bound: -1.0456496, upper bound: 1.0415061
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 2, lower bound: -1.0465252, upper bound: 1.0406161
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 2, lower bound: -1.0384608, upper bound: 1.0487062
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 2, lower bound: -1.0512370, upper bound: 1.0359288
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 2, lower bound: -1.0431727, upper bound: 1.0440186
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 2, lower bound: -1.0506786, upper bound: 1.0364821
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 2, lower bound: -1.0426304, upper bound: 1.0445701
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 2, lower bound: -1.0553661, upper bound: 1.0317842
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 2, lower bound: -1.0473180, upper bound: 1.0398736
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 2, lower bound: -1.0398735, upper bound: 1.0473177
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 2, lower bound: -1.0317834, upper bound: 1.0553660
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 2, lower bound: -1.0445702, upper bound: 1.0426303
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 2, lower bound: -1.0364800, upper bound: 1.0506779
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 2, lower bound: -1.0440194, upper bound: 1.0431729
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 2, lower bound: -1.0359294, upper bound: 1.0512364
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 2, lower bound: -1.0487068, upper bound: 1.0384610
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 2, lower bound: -1.0406170, upper bound: 1.0465252
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 2, lower bound: -1.0415060, upper bound: 1.0456488
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 2, lower bound: -1.0334417, upper bound: 1.0537386
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 2, lower bound: -1.0462104, upper bound: 1.0409614
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 2, lower bound: -1.0381479, upper bound: 1.0490516
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 2, lower bound: -1.0456460, upper bound: 1.0415106
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 2, lower bound: -1.0375979, upper bound: 1.0496010
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 2, lower bound: -1.0503334, upper bound: 1.0368031
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.23
Output dim: 2, lower bound: -1.0422853, upper bound: 1.0448935

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4462676, 2.4490871
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.3875780, 2.3785172
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4266682, 2.4276657
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5357728, 2.5446613
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.5222187, 2.5119328
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0671344, 2.0778012
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.8046417, 2.7866144
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0914803, 2.0735030
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0351210, 2.0509455
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.1070957, 2.0971565

Time for backsubstitution: 14.67 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=2.4546236991882324
rel_dist={2: [-1.0555928908856513, 1.0555926528901223]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 6184
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 4639

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9012838, upper bound: 0.8963065
time: 5.17 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8963070, upper bound: 0.9012833
time: 5.45 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 10.81 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 10.81
Output dim: 2, lower bound: -0.9012838, upper bound: 0.8963065
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 10.81
Output dim: 2, lower bound: -0.8963070, upper bound: 0.9012833

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.3481126, 2.3482857
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.2796340, 2.2721052
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.3841600, 2.3814716
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.4363551, 2.4363697
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.4002867, 2.4037776
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0278854, 2.0312786
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.8160853, 2.8149142
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0290718, 2.0271478
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.1080737, 2.1008921
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.0212164, 2.0208387

Time for backsubstitution: 14.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6184
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6184

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9004148, upper bound: 0.8963039
time: 4.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9012809, upper bound: 0.8954370
time: 5.03 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.3482852, 2.3481131
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.2721047, 2.2796340
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.3814716, 2.3841600
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.4363704, 2.4363551
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.4037781, 2.4002862
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0312786, 2.0278854
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.8149142, 2.8160844
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0271482, 2.0290718
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.1008925, 2.1080735
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.0208387, 2.0212162

Time for backsubstitution: 14.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6184
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 6184

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8954377, upper bound: 0.9012806
time: 5.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8963041, upper bound: 0.9004141
time: 4.69 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 24.73 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.73
Output dim: 2, lower bound: -0.9004148, upper bound: 0.8963039
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.73
Output dim: 2, lower bound: -0.9012809, upper bound: 0.8954370
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.73
Output dim: 2, lower bound: -0.8954377, upper bound: 0.9012806
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.73
Output dim: 2, lower bound: -0.8963041, upper bound: 0.9004141

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.3453002, 2.3481889
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.2814751, 2.2729445
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.3822918, 2.3819413
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.4365005, 2.4409764
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.3856673, 2.3817854
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0193014, 2.0237675
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.7454348, 2.7341771
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0053978, 1.9956551
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0252738, 2.0294573
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.0269475, 2.0249572

Time for backsubstitution: 14.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 4616

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8971448, upper bound: 0.8962970
time: 5.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9004092, upper bound: 0.8930195
time: 4.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.3480163, 2.3454733
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.2804737, 2.2739458
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.3846292, 2.3796029
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.4409618, 2.4365149
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.3782935, 2.3891590
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0203733, 2.0226951
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.7353477, 2.7442646
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -1.9975796, 2.0034733
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0366387, 2.0180919
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.0253348, 2.0265703

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 4616

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8979974, upper bound: 0.8954314
time: 4.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9012743, upper bound: 0.8921674
time: 4.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.3454738, 2.3480163
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.2739458, 2.2804737
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.3796034, 2.3846297
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.4365149, 2.4409621
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.3891597, 2.3782940
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0226955, 2.0203738
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.7442646, 2.7353473
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0034733, 1.9975791
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0180917, 2.0366387
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.0265703, 2.0253348

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 4616

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8921676, upper bound: 0.9012737
time: 4.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8954321, upper bound: 0.8979993
time: 4.45 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.3481879, 2.3453007
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.2729445, 2.2814751
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.3819418, 2.3822913
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.4409771, 2.4365005
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.3817859, 2.3856676
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0237675, 2.0193019
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.7341776, 2.7454348
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -1.9956551, 2.0053978
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0294576, 2.0252733
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.0249572, 2.0269475

Time for backsubstitution: 14.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 4616

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8930202, upper bound: 0.9004084
time: 4.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8962975, upper bound: 0.8971446
time: 4.86 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 24.73 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 24.73
Output dim: 2, lower bound: -0.8971448, upper bound: 0.8962970
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.73
Output dim: 2, lower bound: -0.9004092, upper bound: 0.8930195
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.73
Output dim: 2, lower bound: -0.8979974, upper bound: 0.8954314
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.73
Output dim: 2, lower bound: -0.9012743, upper bound: 0.8921674
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.73
Output dim: 2, lower bound: -0.8921676, upper bound: 0.9012737
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.73
Output dim: 2, lower bound: -0.8954321, upper bound: 0.8979993
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.73
Output dim: 2, lower bound: -0.8930202, upper bound: 0.9004084
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 24.73
Output dim: 2, lower bound: -0.8962975, upper bound: 0.8971446

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.3436766, 2.3467674
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.2654777, 2.2546692
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.3637905, 2.3607988
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.4608569, 2.4623086
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.4042258, 2.4042149
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0147085, 2.0185246
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.6926136, 2.6879606
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0229506, 2.0165281
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -1.9717231, 1.9680610
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -1.9985671, 2.0026231

Time for backsubstitution: 14.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 903

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8967383, upper bound: 0.8928966
time: 4.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9002651, upper bound: 0.8893469
time: 4.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.3465948, 2.3438492
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.2621980, 2.2579489
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.3634872, 2.3611026
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.4622941, 2.4608712
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.4007230, 2.4077170
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0151310, 2.0181026
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.6891308, 2.6914439
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0184531, 2.0210261
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -1.9752421, 1.9645414
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.0030007, 1.9981899

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 903

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8943248, upper bound: 0.8952883
time: 7.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8978749, upper bound: 0.8917606
time: 5.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.3463917, 2.3440523
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.2644763, 2.2556705
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.3661242, 2.3584604
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.4653134, 2.4578471
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.3968520, 2.4115779
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0157804, 2.0174527
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.6825266, 2.6980371
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0151315, 2.0243397
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -1.9830637, 1.9566956
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -1.9969544, 2.0042276

Time for backsubstitution: 14.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 903

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8976066, upper bound: 0.8920445
time: 9.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9011273, upper bound: 0.8884943
time: 4.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.3440523, 2.3463917
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.2556710, 2.2644758
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.3584604, 2.3661242
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.4578471, 2.4653134
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.4115777, 2.3968520
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0174522, 2.0157804
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.6980371, 2.6825271
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0243402, 2.0151320
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -1.9566956, 1.9830637
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.0042276, 1.9969540

Time for backsubstitution: 14.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 903

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8884951, upper bound: 0.9011269
time: 5.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8920447, upper bound: 0.8976062
time: 4.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.3438492, 2.3465948
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.2579494, 2.2621975
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.3611031, 2.3634872
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.4608712, 2.4622943
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.4077172, 2.4007235
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0181017, 2.0151310
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.6914444, 2.6891313
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0210261, 2.0184526
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -1.9645414, 1.9752424
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -1.9981894, 2.0030003

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 903

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8917611, upper bound: 0.8978742
time: 4.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8952883, upper bound: 0.8943240
time: 4.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.3467674, 2.3436766
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.2546687, 2.2654781
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.3607988, 2.3637905
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.4623094, 2.4608567
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.4042153, 2.4042256
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0185242, 2.0147090
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.6879606, 2.6926146
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0165286, 2.0229506
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -1.9680610, 1.9717231
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.0026231, 1.9985671

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 903

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8893476, upper bound: 0.9002644
time: 4.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8928972, upper bound: 0.8967377
time: 5.18 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 24.87 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 24.87
Output dim: 2, lower bound: -0.8967383, upper bound: 0.8928966
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.87
Output dim: 2, lower bound: -0.9002651, upper bound: 0.8893469
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 24.87
Output dim: 2, lower bound: -0.8943248, upper bound: 0.8952883
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 24.87
Output dim: 2, lower bound: -0.8978749, upper bound: 0.8917606
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 24.87
Output dim: 2, lower bound: -0.8976066, upper bound: 0.8920445
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.87
Output dim: 2, lower bound: -0.9011273, upper bound: 0.8884943
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.87
Output dim: 2, lower bound: -0.8884951, upper bound: 0.9011269
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 24.87
Output dim: 2, lower bound: -0.8920447, upper bound: 0.8976062
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 24.87
Output dim: 2, lower bound: -0.8917611, upper bound: 0.8978742
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 24.87
Output dim: 2, lower bound: -0.8952883, upper bound: 0.8943240
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.87
Output dim: 2, lower bound: -0.8893476, upper bound: 0.9002644
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 24.87
Output dim: 2, lower bound: -0.8928972, upper bound: 0.8967377

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.3378983, 2.3401642
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.2642283, 2.2535582
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.3660212, 2.3627100
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.4602709, 2.4617953
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.4062424, 2.4059777
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0120649, 2.0161729
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.6895037, 2.6842937
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0223083, 2.0159492
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -1.9710407, 1.9674542
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.0022988, 2.0058861

Time for backsubstitution: 14.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 850

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9002473, upper bound: 0.8833718
time: 4.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8942893, upper bound: 0.8893291
time: 4.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.3406134, 2.3374491
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.2632260, 2.2545595
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.3683548, 2.3603716
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.4647274, 2.4573338
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.3988686, 2.4133410
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0131359, 2.0151010
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.6794157, 2.6943703
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0144892, 2.0237613
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -1.9823818, 1.9560888
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.0006862, 2.0074906

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 850

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9011094, upper bound: 0.8825193
time: 4.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8951535, upper bound: 0.8884765
time: 5.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.3374481, 2.3406134
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.2545600, 2.2632265
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.3603716, 2.3683548
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.4573345, 2.4647269
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.4133415, 2.3988686
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0151014, 2.0131364
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.6943703, 2.6794162
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0237617, 2.0144892
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -1.9560890, 1.9823813
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.0074906, 2.0006862

Time for backsubstitution: 14.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 850

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8884771, upper bound: 0.8951534
time: 4.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8825200, upper bound: 0.9011089
time: 4.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.3401642, 2.3378983
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.2535577, 2.2642288
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.3627100, 2.3660212
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.4617958, 2.4602702
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.4059782, 2.4062421
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0161734, 2.0120649
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.6842937, 2.6895037
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0159492, 2.0223079
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -1.9674549, 1.9710407
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.0058861, 2.0022988

Time for backsubstitution: 14.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 850

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8893296, upper bound: 0.8942888
time: 5.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8833723, upper bound: 0.9002493
time: 4.95 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 25.05 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.05
Output dim: 2, lower bound: -0.9002473, upper bound: 0.8833718
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 25.05
Output dim: 2, lower bound: -0.8942893, upper bound: 0.8893291
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.05
Output dim: 2, lower bound: -0.9011094, upper bound: 0.8825193
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 25.05
Output dim: 2, lower bound: -0.8951535, upper bound: 0.8884765
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 25.05
Output dim: 2, lower bound: -0.8884771, upper bound: 0.8951534
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.05
Output dim: 2, lower bound: -0.8825200, upper bound: 0.9011089
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 25.05
Output dim: 2, lower bound: -0.8893296, upper bound: 0.8942888
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.05
Output dim: 2, lower bound: -0.8833723, upper bound: 0.9002493

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.3301129, 2.3309836
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.2677889, 2.2567143
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.3578091, 2.3526392
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.4669108, 2.4676800
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.4079933, 2.4075053
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -1.9866138, 1.9938993
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.6593685, 2.6579318
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0197773, 2.0130572
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -1.9705553, 1.9669080
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.0013523, 2.0050445

Time for backsubstitution: 14.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 470

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8998092, upper bound: 0.8833702
time: 5.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9002459, upper bound: 0.8829345
time: 4.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.3328290, 2.3282681
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.2667866, 2.2577162
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.3601427, 2.3503008
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.4713674, 2.4632185
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.4006205, 2.4148684
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -1.9876847, 1.9928274
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.6492805, 2.6680083
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0119591, 2.0208688
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -1.9818964, 1.9555428
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -1.9997392, 2.0066490

Time for backsubstitution: 14.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 470

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9006713, upper bound: 0.8825180
time: 4.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9011080, upper bound: 0.8820817
time: 4.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.3282685, 2.3328285
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.2577162, 2.2667871
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.3503008, 2.3601427
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.4632182, 2.4713674
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.4148684, 2.4006205
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -1.9928274, 1.9876847
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.6680088, 2.6492801
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0208683, 2.0119591
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -1.9555430, 1.9818959
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.0066490, 1.9997392

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 470

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8820825, upper bound: 0.9011077
time: 4.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8825186, upper bound: 0.9006707
time: 4.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.3309836, 2.3301134
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.2567148, 2.2677894
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.3526392, 2.3578091
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.4676805, 2.4669106
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.4075050, 2.4079940
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -1.9938993, 1.9866138
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.6579323, 2.6593671
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0130577, 2.0197778
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -1.9669085, 1.9705553
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.0050445, 2.0013523

Time for backsubstitution: 14.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 470

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8829349, upper bound: 0.9002451
time: 5.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8833710, upper bound: 0.8998085
time: 4.52 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 24.73 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 24.73
Output dim: 2, lower bound: -0.8998092, upper bound: 0.8833702
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 24.73
Output dim: 2, lower bound: -0.9002459, upper bound: 0.8829345
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 24.73
Output dim: 2, lower bound: -0.9006713, upper bound: 0.8825180
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 24.73
Output dim: 2, lower bound: -0.9011080, upper bound: 0.8820817
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 24.73
Output dim: 2, lower bound: -0.8820825, upper bound: 0.9011077
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 24.73
Output dim: 2, lower bound: -0.8825186, upper bound: 0.9006707
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 24.73
Output dim: 2, lower bound: -0.8829349, upper bound: 0.9002451
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 24.73
Output dim: 2, lower bound: -0.8833710, upper bound: 0.8998085

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.3235040, 2.3260555
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.2555237, 2.2475724
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.3574409, 2.3521481
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.4654474, 2.4657199
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.3941069, 2.3971624
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -1.9833860, 1.9895663
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.6580944, 2.6569681
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0167837, 2.0090365
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -1.9692678, 1.9651842
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -1.9989223, 2.0017791

Time for backsubstitution: 14.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 65

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8998084, upper bound: 0.8833700
time: 4.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8998072, upper bound: 0.8829412
time: 4.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.3251853, 2.3243742
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.2586479, 2.2444491
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.3573179, 2.3522706
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.4649506, 2.4662170
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.3976517, 2.3936181
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -1.9822803, 1.9906721
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.6584034, 2.6566596
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0157576, 2.0100632
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -1.9688315, 1.9656205
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -1.9980869, 2.0026145

Time for backsubstitution: 14.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 65

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8998192, upper bound: 0.8829336
time: 4.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9002456, upper bound: 0.8829333
time: 4.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.3262191, 2.3233404
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.2545223, 2.2485743
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.3597736, 2.3498096
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.4699049, 2.4612584
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.3867331, 2.4045258
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -1.9844575, 1.9884944
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.6480083, 2.6670442
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0089655, 2.0168481
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -1.9806085, 1.9538190
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -1.9973097, 2.0033836

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 65

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9006705, upper bound: 0.8825176
time: 4.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9006694, upper bound: 0.8820890
time: 4.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.3279004, 2.3216591
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.2576447, 2.2454510
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.3596516, 2.3499327
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.4694071, 2.4617555
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.3902779, 2.4009814
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -1.9833512, 1.9896002
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.6483154, 2.6667361
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0079384, 2.0178747
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -1.9801722, 1.9542551
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -1.9964738, 2.0042191

Time for backsubstitution: 14.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 65

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9006813, upper bound: 0.8820811
time: 4.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9011078, upper bound: 0.8820809
time: 5.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.3216586, 2.3279004
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.2454510, 2.2576447
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.3499327, 2.3596511
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.4617558, 2.4694073
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.4009819, 2.3902779
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -1.9895997, 1.9833517
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.6667366, 2.6483159
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0178747, 2.0079384
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -1.9542551, 1.9801722
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.0042191, 1.9964738

Time for backsubstitution: 14.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 65

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8820817, upper bound: 0.9011076
time: 9.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8820815, upper bound: 0.9006833
time: 5.23 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.3233399, 2.3262196
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.2485733, 2.2545214
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.3498096, 2.3597741
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.4612589, 2.4699044
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.4045258, 2.3867333
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -1.9884939, 1.9844580
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.6670437, 2.6480079
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0168486, 2.0089650
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -1.9538193, 1.9806085
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.0033836, 1.9973092

Time for backsubstitution: 14.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 65

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8820892, upper bound: 0.9006686
time: 5.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8825184, upper bound: 0.9006698
time: 4.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.3243747, 2.3251848
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.2444496, 2.2586470
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.3522711, 2.3573179
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.4662170, 2.4649506
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.3936176, 2.3976514
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -1.9906716, 1.9822807
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.6566601, 2.6584034
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0100632, 2.0157571
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -1.9656205, 1.9688315
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.0026145, 1.9980869

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 65

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8829341, upper bound: 0.9002450
time: 4.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8829339, upper bound: 0.8998185
time: 5.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.3260550, 2.3235040
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.2475719, 2.2555237
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.3521481, 2.3574405
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.4657202, 2.4654477
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.3971624, 2.3941069
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -1.9895663, 1.9833865
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.6569672, 2.6580954
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0090361, 2.0167837
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -1.9651842, 1.9692676
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.0017791, 1.9989223

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 65

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8829416, upper bound: 0.8998067
time: 4.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8833707, upper bound: 0.8998077
time: 4.76 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 24.26 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 24.26
Output dim: 2, lower bound: -0.8998084, upper bound: 0.8833700
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 24.26
Output dim: 2, lower bound: -0.8998072, upper bound: 0.8829412
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 24.26
Output dim: 2, lower bound: -0.8998192, upper bound: 0.8829336
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 24.26
Output dim: 2, lower bound: -0.9002456, upper bound: 0.8829333
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 24.26
Output dim: 2, lower bound: -0.9006705, upper bound: 0.8825176
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 24.26
Output dim: 2, lower bound: -0.9006694, upper bound: 0.8820890
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 24.26
Output dim: 2, lower bound: -0.9006813, upper bound: 0.8820811
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 24.26
Output dim: 2, lower bound: -0.9011078, upper bound: 0.8820809
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 24.26
Output dim: 2, lower bound: -0.8820817, upper bound: 0.9011076
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 24.26
Output dim: 2, lower bound: -0.8820815, upper bound: 0.9006833
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 24.26
Output dim: 2, lower bound: -0.8820892, upper bound: 0.9006686
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 24.26
Output dim: 2, lower bound: -0.8825184, upper bound: 0.9006698
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 24.26
Output dim: 2, lower bound: -0.8829341, upper bound: 0.9002450
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 24.26
Output dim: 2, lower bound: -0.8829339, upper bound: 0.8998185
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 24.26
Output dim: 2, lower bound: -0.8829416, upper bound: 0.8998067
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 24.26
Output dim: 2, lower bound: -0.8833707, upper bound: 0.8998077

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.3234949, 2.3260398
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.2555170, 2.2475686
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.3574381, 2.3521466
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.4654465, 2.4657180
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.3940973, 2.3971567
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -1.9833822, 1.9895639
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.6580849, 2.6569624
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0167809, 2.0090361
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -1.9692597, 1.9651797
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -1.9989138, 2.0017648

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2137
type: RSZ, layer: 3, pos: 3119
type: RSZ, layer: 3, pos: 2819
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 1104
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 711
type: RSZ, layer: 3, pos: 1437
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 745
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 571
type: RSZ, layer: 3, pos: 1921
type: RSZ, layer: 3, pos: 898
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1703
type: RSZ, layer: 3, pos: 962
type: RSZ, layer: 3, pos: 947
type: RSZ, layer: 3, pos: 2573
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 744
type: RSZ, layer: 3, pos: 1699
type: RSZ, layer: 3, pos: 569
type: RSZ, layer: 3, pos: 599
type: RSZ, layer: 3, pos: 2921
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 163
type: RSZ, layer: 3, pos: 2318
type: RSZ, layer: 3, pos: 1940
type: RSZ, layer: 3, pos: 897
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1919
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 407
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 2069
type: RSZ, layer: 3, pos: 2319
type: RSZ, layer: 3, pos: 1493
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 2389
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 1410
type: RSZ, layer: 3, pos: 2238
type: RSZ, layer: 3, pos: 969
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 2348
type: RSZ, layer: 3, pos: 181
type: RSZ, layer: 3, pos: 1123
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 695
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1968
type: RSZ, layer: 3, pos: 2857
type: RSZ, layer: 3, pos: 2582
type: RSZ, layer: 3, pos: 1797
type: RSZ, layer: 3, pos: 174

Time for candidate selection: 0.33 seconds

### Candidate
type: RSZ, layer: 3, pos: 2137

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8735955, upper bound: 0.8650762
time: 4.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8817199, upper bound: 0.8570923
time: 4.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.3234873, 2.3260336
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.2555151, 2.2475662
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.3574371, 2.3521452
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.4654455, 2.4657171
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.3940964, 2.3971524
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -1.9833808, 1.9895625
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.6580811, 2.6569576
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0167789, 2.0090337
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -1.9692569, 1.9651763
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -1.9989080, 2.0017710

Time for backsubstitution: 14.72 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2137
type: RSZ, layer: 3, pos: 3119
type: RSZ, layer: 3, pos: 2819
type: RSZ, layer: 3, pos: 1704
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 1794
type: RSZ, layer: 3, pos: 1104
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 711
type: RSZ, layer: 3, pos: 1437
type: RSZ, layer: 3, pos: 900
type: RSZ, layer: 3, pos: 745
type: RSZ, layer: 3, pos: 912
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 571
type: RSZ, layer: 3, pos: 1921
type: RSZ, layer: 3, pos: 898
type: RSZ, layer: 3, pos: 1402
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1703
type: RSZ, layer: 3, pos: 962
type: RSZ, layer: 3, pos: 947
type: RSZ, layer: 3, pos: 2573
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 744
type: RSZ, layer: 3, pos: 1699
type: RSZ, layer: 3, pos: 569
type: RSZ, layer: 3, pos: 599
type: RSZ, layer: 3, pos: 2921
type: RSZ, layer: 3, pos: 760
type: RSZ, layer: 3, pos: 163
type: RSZ, layer: 3, pos: 2318
type: RSZ, layer: 3, pos: 1940
type: RSZ, layer: 3, pos: 897
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 2852
type: RSZ, layer: 3, pos: 914
type: RSZ, layer: 3, pos: 418
type: RSZ, layer: 3, pos: 1919
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 407
type: RSZ, layer: 3, pos: 2220
type: RSZ, layer: 3, pos: 1383
type: RSZ, layer: 3, pos: 2069
type: RSZ, layer: 3, pos: 2319
type: RSZ, layer: 3, pos: 1493
type: RSZ, layer: 3, pos: 2488
type: RSZ, layer: 3, pos: 2389
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 1410
type: RSZ, layer: 3, pos: 2238
type: RSZ, layer: 3, pos: 969
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1248
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 2348
type: RSZ, layer: 3, pos: 181
type: RSZ, layer: 3, pos: 1123
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 695
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1968
type: RSZ, layer: 3, pos: 2857
type: RSZ, layer: 3, pos: 2582
type: RSZ, layer: 3, pos: 1797
type: RSZ, layer: 3, pos: 174

Time for candidate selection: 0.33 seconds

### Candidate
type: RSZ, layer: 3, pos: 2137

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8735944, upper bound: 0.8646491
time: 4.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8817187, upper bound: 0.8566653
time: 4.59 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 24.58 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 24.58
Output dim: 2, lower bound: -0.8735955, upper bound: 0.8650762
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 24.58
Output dim: 2, lower bound: -0.8817199, upper bound: 0.8570923
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 24.58
Output dim: 2, lower bound: -0.8735944, upper bound: 0.8646491
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 24.58
Output dim: 2, lower bound: -0.8817187, upper bound: 0.8566653
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 24.58
Output dim: 2, lower bound: -0.8998192, upper bound: 0.8829336
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 24.58
Output dim: 2, lower bound: -0.9002456, upper bound: 0.8829333
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 24.58
Output dim: 2, lower bound: -0.9006705, upper bound: 0.8825176
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 24.58
Output dim: 2, lower bound: -0.9006694, upper bound: 0.8820890
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 24.58
Output dim: 2, lower bound: -0.9006813, upper bound: 0.8820811
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 24.58
Output dim: 2, lower bound: -0.9011078, upper bound: 0.8820809
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 24.58
Output dim: 2, lower bound: -0.8820817, upper bound: 0.9011076
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 24.58
Output dim: 2, lower bound: -0.8820815, upper bound: 0.9006833
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 24.58
Output dim: 2, lower bound: -0.8820892, upper bound: 0.9006686
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 24.58
Output dim: 2, lower bound: -0.8825184, upper bound: 0.9006698
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 24.58
Output dim: 2, lower bound: -0.8829341, upper bound: 0.9002450
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 24.58
Output dim: 2, lower bound: -0.8829339, upper bound: 0.8998185
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 24.58
Output dim: 2, lower bound: -0.8829416, upper bound: 0.8998067
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 24.58
Output dim: 2, lower bound: -0.8833707, upper bound: 0.8998077
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=2.384212017059326
rel_dist={2: [-0.9012986819155433, 0.9012981480281343]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2426.77 seconds
