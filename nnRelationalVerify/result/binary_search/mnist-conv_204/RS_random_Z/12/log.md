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
execution time: IAR + LP analysis = 15.22 + 32.20 = 47.42 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3552.58 seconds, max iter: 100)

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
Binary search time: 198.70 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Relational Split (RS_random_Z) starts
Time budget: 3353.88 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6184
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 4616

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6184

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4759393, upper bound: 1.4790803
time: 4.18 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4790802, upper bound: 1.4759393
time: 4.09 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 8.29 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 8.29
Output dim: 2, lower bound: -1.4759393, upper bound: 1.4790803
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 8.29
Output dim: 2, lower bound: -1.4790802, upper bound: 1.4759393

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.8104463, 2.8167820
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.7587509, 2.7564135
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.6639872, 2.6694431
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.6605895, 2.6605895
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.8265510, 2.8093462
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.3462820, 2.3487835
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -3.2920537, 3.2685175
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.2711792, 2.2529354
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.3051300, 2.3316491
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.4192386, 2.4154751

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 4616

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 470

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4749106, upper bound: 1.4790783
time: 4.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4759374, upper bound: 1.4780520
time: 3.98 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.8167815, 2.8104458
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.7564144, 2.7587504
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.6694431, 2.6639872
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.6605895, 2.6605895
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.8093467, 2.8265512
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.3487835, 2.3462820
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -3.2685170, 3.2920547
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.2529354, 2.2711787
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.3316493, 2.3051300
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.4154754, 2.4192386

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4616

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4718532, upper bound: 1.4759278
time: 4.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4790657, upper bound: 1.4687378
time: 4.48 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 23.11 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 23.11
Output dim: 2, lower bound: -1.4749106, upper bound: 1.4790783
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 23.11
Output dim: 2, lower bound: -1.4759374, upper bound: 1.4780520
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 23.11
Output dim: 2, lower bound: -1.4718532, upper bound: 1.4759278
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 23.11
Output dim: 2, lower bound: -1.4790657, upper bound: 1.4687378

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.8038354, 2.8140950
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.7464857, 2.7514367
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.6637831, 2.6689529
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.6605895, 2.6605895
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.8126636, 2.8037295
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.3445292, 2.3444505
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -3.2907834, 3.2679648
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.2695532, 2.2489142
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.3044248, 2.3299260
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.4179235, 2.4122102

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 4616

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 903

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4663357, upper bound: 1.4789852
time: 4.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4748118, upper bound: 1.4704944
time: 4.25 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.8077588, 2.8101721
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.7537737, 2.7441487
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.6634970, 2.6692395
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.6605895, 2.6605895
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.8209348, 2.7954590
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.3419485, 2.3470306
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -3.2915025, 3.2672458
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.2671576, 2.2513099
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.3034072, 2.3309441
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.4159732, 2.4141598

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 850

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4639

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4758841, upper bound: 1.4741541
time: 3.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4720358, upper bound: 1.4779988
time: 4.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.8156309, 2.8088217
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.7381377, 2.7457914
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.6483026, 2.6490107
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.6605895, 2.6605895
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.8369374, 2.8451099
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.3435392, 2.3425546
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -3.2311068, 3.2392335
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.2782364, 2.2887321
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.2702532, 2.2620411
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.4012032, 2.3908587

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 4639

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 903

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4632661, upper bound: 1.4758289
time: 4.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4717649, upper bound: 1.4673532
time: 3.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.8151579, 2.8092952
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.7434516, 2.7404747
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.6544557, 2.6428461
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.6605895, 2.6605895
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.8279042, 2.8541188
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.3450537, 2.3410382
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -3.2156954, 3.2546177
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.2704887, 2.2964640
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.2885027, 2.2437339
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.3870955, 2.4049473

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4639

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4790125, upper bound: 1.4648409
time: 6.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4751678, upper bound: 1.4686838
time: 4.40 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 25.27 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.27
Output dim: 2, lower bound: -1.4663357, upper bound: 1.4789852
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.27
Output dim: 2, lower bound: -1.4748118, upper bound: 1.4704944
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.27
Output dim: 2, lower bound: -1.4758841, upper bound: 1.4741541
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.27
Output dim: 2, lower bound: -1.4720358, upper bound: 1.4779988
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.27
Output dim: 2, lower bound: -1.4632661, upper bound: 1.4758289
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.27
Output dim: 2, lower bound: -1.4717649, upper bound: 1.4673532
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.27
Output dim: 2, lower bound: -1.4790125, upper bound: 1.4648409
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.27
Output dim: 2, lower bound: -1.4751678, upper bound: 1.4686838

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.7972341, 2.8094182
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.7453742, 2.7500024
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.6652679, 2.6711836
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.6605895, 2.6605895
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.8144264, 2.8060830
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.3421783, 2.3414168
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -3.2871165, 3.2655969
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.2690620, 2.2482738
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.3038177, 2.3291426
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.4211855, 2.4165666

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 850

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4616

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4591207, upper bound: 1.4789709
time: 3.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4663245, upper bound: 1.4717628
time: 3.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.7991586, 2.8074937
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.7450519, 2.7503257
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.6660137, 2.6704378
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.6605895, 2.6605895
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.8150177, 2.8054914
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.3414965, 2.3420992
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -3.2884154, 3.2642989
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.2689123, 2.2484226
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.3036413, 2.3293190
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.4222794, 2.4154725

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 850

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4748117, upper bound: 1.4562737
time: 4.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4606007, upper bound: 1.4704943
time: 4.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.8075891, 2.8104057
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.7636728, 2.7364807
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.6670294, 2.6664991
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.6605895, 2.6605895
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.8173699, 2.8000414
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.3384829, 2.3514829
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -3.2930403, 3.2660532
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.2696900, 2.2493525
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.3128920, 2.3236721
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.4164705, 2.4137759

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 4616

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 65

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4748819, upper bound: 1.4741505
time: 3.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4758834, upper bound: 1.4741532
time: 4.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.8077588, 2.8100028
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.7461061, 2.7441487
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.6607571, 2.6692395
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.6605895, 2.6605895
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.8209348, 2.7918949
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.3419485, 2.3435645
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -3.2903090, 3.2672458
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.2652001, 2.2513099
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.2961354, 2.3309441
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.4155893, 2.4141598

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 903

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4634615, upper bound: 1.4779051
time: 3.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4719370, upper bound: 1.4694149
time: 3.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.8090277, 2.8041430
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.7370267, 2.7443571
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.6497869, 2.6512413
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.6605895, 2.6605895
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.8387012, 2.8474646
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.3411884, 2.3395209
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -3.2274404, 3.2368650
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.2777424, 2.2880888
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.2696466, 2.2612581
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.4044652, 2.3952150

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4639

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4632114, upper bound: 1.4719274
time: 4.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4593688, upper bound: 1.4757756
time: 3.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.8109522, 2.8022184
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.7367043, 2.7446804
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.6505327, 2.6504955
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.6605895, 2.6605895
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.8392925, 2.8468730
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.3405066, 2.3402038
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -3.2287374, 3.2355671
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.2775936, 2.2882380
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.2694702, 2.2614346
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.4055600, 2.3941212

Time for backsubstitution: 14.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 850

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4639

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4717108, upper bound: 1.4634520
time: 4.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4678682, upper bound: 1.4672995
time: 4.25 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 903

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 65

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4781739, upper bound: 1.4648407
time: 5.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4790118, upper bound: 1.4640086
time: 5.93 seconds

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

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 850

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 470

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4741396, upper bound: 1.4686820
time: 4.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4751660, upper bound: 1.4676556
time: 3.95 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 22.68 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.68
Output dim: 2, lower bound: -1.4591207, upper bound: 1.4789709
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.68
Output dim: 2, lower bound: -1.4663245, upper bound: 1.4717628
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.68
Output dim: 2, lower bound: -1.4748117, upper bound: 1.4562737
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.68
Output dim: 2, lower bound: -1.4606007, upper bound: 1.4704943
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.68
Output dim: 2, lower bound: -1.4748819, upper bound: 1.4741505
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.68
Output dim: 2, lower bound: -1.4758834, upper bound: 1.4741532
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.68
Output dim: 2, lower bound: -1.4634615, upper bound: 1.4779051
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.68
Output dim: 2, lower bound: -1.4719370, upper bound: 1.4694149
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.68
Output dim: 2, lower bound: -1.4632114, upper bound: 1.4719274
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.68
Output dim: 2, lower bound: -1.4593688, upper bound: 1.4757756
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.68
Output dim: 2, lower bound: -1.4717108, upper bound: 1.4634520
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.68
Output dim: 2, lower bound: -1.4678682, upper bound: 1.4672995
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.68
Output dim: 2, lower bound: -1.4781739, upper bound: 1.4648407
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.68
Output dim: 2, lower bound: -1.4790118, upper bound: 1.4640086
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.68
Output dim: 2, lower bound: -1.4741396, upper bound: 1.4686820
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.68
Output dim: 2, lower bound: -1.4751660, upper bound: 1.4676556

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.7960825, 2.8077922
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.7270999, 2.7370420
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.6441259, 2.6561947
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.6605895, 2.6605895
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.8419943, 2.8246422
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.3369341, 2.3376875
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -3.2496781, 3.2127743
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.2943459, 2.2658257
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.2424231, 2.2859974
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.4068942, 2.3881872

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 850

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4639

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4590665, upper bound: 1.4750700
time: 4.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4552240, upper bound: 1.4789176
time: 4.22 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.7956085, 2.8082657
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.7324157, 2.7317271
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.6502905, 2.6500411
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.6605895, 2.6605895
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.8329859, 2.8336754
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.3384504, 2.3361726
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -3.2342935, 3.2281847
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.2866144, 2.2735739
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.2607303, 2.2677474
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.3928065, 2.4022954

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 4639

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 65

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4663236, upper bound: 1.4717622
time: 3.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4663210, upper bound: 1.4707615
time: 3.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.7932334, 2.7983112
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.7491503, 2.7534814
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.6602774, 2.6603656
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.6605895, 2.6605895
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.8167715, 2.8067212
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.3160448, 2.3240633
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -3.2582779, 3.2429690
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.2668643, 2.2455287
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.3031564, 2.3286924
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.4211922, 2.4146304

Time for backsubstitution: 14.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 4616

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 65

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4748108, upper bound: 1.4562731
time: 3.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4748082, upper bound: 1.4552641
time: 4.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.7899766, 2.8015676
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.7482071, 2.7544241
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.6559410, 2.6647010
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.6605895, 2.6605895
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.8162470, 2.8072453
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.3234596, 2.3166485
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -3.2670841, 3.2341628
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.2660193, 2.2463737
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.3030148, 2.3288341
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.4214373, 2.4143853

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 4616

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4639

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4605474, upper bound: 1.4665965
time: 4.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4566991, upper bound: 1.4704412
time: 4.27 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.8075595, 2.8103900
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.7636662, 2.7364674
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.6670265, 2.6664939
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.6605895, 2.6605895
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.8173604, 2.8000295
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.3384790, 2.3514752
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -3.2930288, 3.2660317
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.2696862, 2.2493439
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.3128843, 2.3236570
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.4164681, 2.4137602

Time for backsubstitution: 14.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 903

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4616

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4676804, upper bound: 1.4741361
time: 4.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4748706, upper bound: 1.4669248
time: 4.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.8075738, 2.8104048
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.7636719, 2.7364740
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.6670284, 2.6664963
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.6605895, 2.6605895
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.8173699, 2.8000312
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.3384829, 2.3514786
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -3.2930403, 3.2660427
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.2696919, 2.2493486
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.3128920, 2.3236642
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.4164548, 2.4137740

Time for backsubstitution: 14.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 850

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4616

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4686814, upper bound: 1.4741390
time: 4.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4758721, upper bound: 1.4669275
time: 4.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.8011575, 2.8053260
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.7449942, 2.7427144
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.6622410, 2.6714702
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.6605895, 2.6605895
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.8226967, 2.7942483
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.3395977, 2.3405318
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -3.2866440, 3.2648778
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.2647071, 2.2506695
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.2955289, 2.3301601
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.4188523, 2.4185162

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 850

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4616

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4562502, upper bound: 1.4778909
time: 4.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4634502, upper bound: 1.4706825
time: 4.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.8030820, 2.8034015
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.7446709, 2.7430377
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.6629868, 2.6707244
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.6605895, 2.6605895
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.8232880, 2.7936571
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.3389158, 2.3412142
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -3.2879410, 3.2635798
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.2645583, 2.2508183
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.2953520, 2.3303368
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.4199452, 2.4174221

Time for backsubstitution: 14.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 4616

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 65

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4709349, upper bound: 1.4694114
time: 4.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4719364, upper bound: 1.4694141
time: 4.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 14.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 470

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4621833, upper bound: 1.4719259
time: 4.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4632096, upper bound: 1.4708990
time: 4.17 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 14.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 850

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 65

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4585345, upper bound: 1.4757751
time: 4.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4593682, upper bound: 1.4749295
time: 3.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 850

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 470

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4706827, upper bound: 1.4634504
time: 4.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4717090, upper bound: 1.4624234
time: 3.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 14.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 470

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 850

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4678681, upper bound: 1.4530873
time: 4.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4536446, upper bound: 1.4672996
time: 4.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.8149872, 2.8095121
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.7533460, 2.7327948
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.6579847, 2.6401052
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.6605895, 2.6605895
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.8243308, 2.8586891
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.3415842, 2.3454905
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -3.2172241, 3.2534037
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.2730155, 2.2945065
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.2979803, 2.2364619
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.3875904, 2.4045482

Time for backsubstitution: 14.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 470

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 903

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4695916, upper bound: 1.4647533
time: 4.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4780745, upper bound: 1.4562512
time: 4.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.8149719, 2.8095274
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.7533402, 2.7328010
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.6579876, 2.6401029
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.6605895, 2.6605895
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.8243289, 2.8586910
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.3415880, 2.3454866
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -3.2172127, 3.2534146
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.2730212, 2.2945018
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.2979879, 2.2364545
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.3875771, 2.4045620

Time for backsubstitution: 14.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 470

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 850

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4790116, upper bound: 1.4497783
time: 6.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4647911, upper bound: 1.4640084
time: 4.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.8085480, 2.8064384
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.7235198, 2.7354980
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.6515107, 2.6423540
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.6605895, 2.6605895
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.8140168, 2.8449368
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.3433008, 2.3332400
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -3.2132311, 3.2540646
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.2669048, 2.2924433
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.2805266, 2.2420111
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.3853941, 2.4016824

Time for backsubstitution: 14.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 903

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 65

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4741387, upper bound: 1.4686813
time: 4.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4741361, upper bound: 1.4676803
time: 4.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.8124714, 2.8025160
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.7308078, 2.7282100
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.6512246, 2.6426406
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.6605895, 2.6605895
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.8222871, 2.8366663
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.3407211, 2.3358207
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -3.2139511, 3.2533455
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.2645092, 2.2948389
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.2795086, 2.2430289
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.3834448, 2.4036317

Time for backsubstitution: 14.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 850

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4751658, upper bound: 1.4534322
time: 4.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4609451, upper bound: 1.4676554
time: 4.03 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 22.97 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.97
Output dim: 2, lower bound: -1.4590665, upper bound: 1.4750700
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.97
Output dim: 2, lower bound: -1.4552240, upper bound: 1.4789176
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.97
Output dim: 2, lower bound: -1.4663236, upper bound: 1.4717622
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.97
Output dim: 2, lower bound: -1.4663210, upper bound: 1.4707615
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.97
Output dim: 2, lower bound: -1.4748108, upper bound: 1.4562731
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.97
Output dim: 2, lower bound: -1.4748082, upper bound: 1.4552641
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.97
Output dim: 2, lower bound: -1.4605474, upper bound: 1.4665965
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.97
Output dim: 2, lower bound: -1.4566991, upper bound: 1.4704412
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.97
Output dim: 2, lower bound: -1.4676804, upper bound: 1.4741361
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.97
Output dim: 2, lower bound: -1.4748706, upper bound: 1.4669248
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.97
Output dim: 2, lower bound: -1.4686814, upper bound: 1.4741390
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.97
Output dim: 2, lower bound: -1.4758721, upper bound: 1.4669275
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.97
Output dim: 2, lower bound: -1.4562502, upper bound: 1.4778909
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.97
Output dim: 2, lower bound: -1.4634502, upper bound: 1.4706825
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.97
Output dim: 2, lower bound: -1.4709349, upper bound: 1.4694114
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.97
Output dim: 2, lower bound: -1.4719364, upper bound: 1.4694141
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.97
Output dim: 2, lower bound: -1.4621833, upper bound: 1.4719259
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.97
Output dim: 2, lower bound: -1.4632096, upper bound: 1.4708990
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.97
Output dim: 2, lower bound: -1.4585345, upper bound: 1.4757751
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.97
Output dim: 2, lower bound: -1.4593682, upper bound: 1.4749295
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.97
Output dim: 2, lower bound: -1.4706827, upper bound: 1.4634504
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.97
Output dim: 2, lower bound: -1.4717090, upper bound: 1.4624234
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.97
Output dim: 2, lower bound: -1.4678681, upper bound: 1.4530873
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.97
Output dim: 2, lower bound: -1.4536446, upper bound: 1.4672996
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.97
Output dim: 2, lower bound: -1.4695916, upper bound: 1.4647533
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.97
Output dim: 2, lower bound: -1.4780745, upper bound: 1.4562512
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.97
Output dim: 2, lower bound: -1.4790116, upper bound: 1.4497783
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.97
Output dim: 2, lower bound: -1.4647911, upper bound: 1.4640084
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.97
Output dim: 2, lower bound: -1.4741387, upper bound: 1.4686813
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.97
Output dim: 2, lower bound: -1.4741361, upper bound: 1.4676803
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.97
Output dim: 2, lower bound: -1.4751658, upper bound: 1.4534322
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.97
Output dim: 2, lower bound: -1.4609451, upper bound: 1.4676554

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.7959127, 2.8080258
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.7369990, 2.7293744
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.6476583, 2.6534548
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.6605895, 2.6605895
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.8384295, 2.8292236
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.3334694, 2.3421402
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -3.2512169, 3.2115817
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.2968779, 2.2638674
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.2519083, 2.2787259
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.4073901, 2.3878021

Time for backsubstitution: 14.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 850

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4590663, upper bound: 1.4608544
time: 4.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4448429, upper bound: 1.4750700
time: 4.28 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.7960825, 2.8076229
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.7194314, 2.7370420
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.6413860, 2.6561947
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.6605895, 2.6605895
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.8419943, 2.8210771
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.3369341, 2.3342218
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -3.2484856, 3.2127743
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.2923870, 2.2658257
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.2351518, 2.2859974
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.4065099, 2.3881872

Time for backsubstitution: 14.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 850

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 65

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4552231, upper bound: 1.4789170
time: 4.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.4552205, upper bound: 1.4779156
time: 4.37 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 23.30 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 23.30
Output dim: 2, lower bound: -1.4590663, upper bound: 1.4608544
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 23.30
Output dim: 2, lower bound: -1.4448429, upper bound: 1.4750700
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 23.30
Output dim: 2, lower bound: -1.4552231, upper bound: 1.4789170
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 23.30
Output dim: 2, lower bound: -1.4552205, upper bound: 1.4779156
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4663236, upper bound: 1.4717622
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4663210, upper bound: 1.4707615
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4748108, upper bound: 1.4562731
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4748082, upper bound: 1.4552641
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4605474, upper bound: 1.4665965
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4566991, upper bound: 1.4704412
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4676804, upper bound: 1.4741361
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4748706, upper bound: 1.4669248
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4686814, upper bound: 1.4741390
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4758721, upper bound: 1.4669275
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4562502, upper bound: 1.4778909
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4634502, upper bound: 1.4706825
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4709349, upper bound: 1.4694114
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4719364, upper bound: 1.4694141
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4621833, upper bound: 1.4719259
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4632096, upper bound: 1.4708990
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4585345, upper bound: 1.4757751
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4593682, upper bound: 1.4749295
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4706827, upper bound: 1.4634504
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4717090, upper bound: 1.4624234
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4678681, upper bound: 1.4530873
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4536446, upper bound: 1.4672996
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4695916, upper bound: 1.4647533
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4780745, upper bound: 1.4562512
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4790116, upper bound: 1.4497783
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4647911, upper bound: 1.4640084
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4741387, upper bound: 1.4686813
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4741361, upper bound: 1.4676803
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4751658, upper bound: 1.4534322
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.30
Output dim: 2, lower bound: -1.4609451, upper bound: 1.4676554
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=2.6658577919006348
rel_dist={2: [-1.4790870249746977, 1.4790871137342352]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 6184
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 470

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 850

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0555918, upper bound: 1.0475435
time: 4.56 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0475436, upper bound: 1.0555917
time: 4.88 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 9.45 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 9.45
Output dim: 2, lower bound: -1.0555918, upper bound: 1.0475435
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 9.45
Output dim: 2, lower bound: -1.0475436, upper bound: 1.0555917

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4572062, 2.4553452
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.4024177, 2.4018788
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4470310, 2.4445534
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5147991, 2.5137916
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.5124760, 2.5121765
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0867786, 2.0910158
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.9192581, 2.9242907
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0905256, 2.0900435
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.1776199, 2.1775386
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.1177745, 2.1179147

Time for backsubstitution: 14.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6184
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 470

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6184

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0539596, upper bound: 1.0475399
time: 4.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0555879, upper bound: 1.0458695
time: 4.44 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4553456, 2.4572062
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.4018779, 2.4024177
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4445534, 2.4470310
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5137920, 2.5147991
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.5121765, 2.5124760
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0910158, 2.0867786
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.9242897, 2.9192586
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0900431, 2.0905261
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.1775389, 2.1776197
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.1179147, 2.1177745

Time for backsubstitution: 14.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6184
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 903

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6184

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0458697, upper bound: 1.0555876
time: 4.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0475399, upper bound: 1.0539596
time: 4.23 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 23.59 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 23.59
Output dim: 2, lower bound: -1.0539596, upper bound: 1.0475399
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 23.59
Output dim: 2, lower bound: -1.0555879, upper bound: 1.0458695
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 23.59
Output dim: 2, lower bound: -1.0458697, upper bound: 1.0555876
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 23.59
Output dim: 2, lower bound: -1.0475399, upper bound: 1.0539596

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4543920, 2.4561524
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.4045944, 2.4027190
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4451609, 2.4458008
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5149431, 2.5198846
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.5003142, 2.4901834
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0781956, 2.0838623
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.8519716, 2.8435540
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0694575, 2.0585504
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0948191, 2.1098919
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.1240435, 2.1220331

Time for backsubstitution: 14.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 903

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0491231, upper bound: 1.0473902
time: 5.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0538107, upper bound: 1.0427037
time: 4.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4580140, 2.4525318
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.4032574, 2.4040546
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4482784, 2.4426832
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5208921, 2.5139358
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.4904828, 2.5000148
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0796251, 2.0824327
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.8385210, 2.8570037
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0590339, 2.0689754
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.1099730, 2.0947380
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.1218929, 2.1241837

Time for backsubstitution: 14.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4616

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0514352, upper bound: 1.0458624
time: 4.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0555797, upper bound: 1.0417132
time: 4.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4525323, 2.4580131
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.4040546, 2.4032578
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4426832, 2.4482784
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5139360, 2.5208921
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.5000148, 2.4904826
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0824327, 2.0796251
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.8570032, 2.8385220
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0689750, 2.0590334
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0947380, 2.1099730
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.1241837, 2.1218932

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 470

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 903

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0410329, upper bound: 1.0554382
time: 4.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0457209, upper bound: 1.0507518
time: 4.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4561524, 2.4543929
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.4027195, 2.4045935
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4458008, 2.4451609
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5198851, 2.5149434
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.4901834, 2.5003140
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0838623, 2.0781960
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.8435545, 2.8519716
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0585504, 2.0694580
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.1098919, 2.0948191
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.1220331, 2.1240437

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 4616

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 65

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0470614, upper bound: 1.0539590
time: 4.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0475395, upper bound: 1.0534849
time: 4.39 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 23.52 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.52
Output dim: 2, lower bound: -1.0491231, upper bound: 1.0473902
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.52
Output dim: 2, lower bound: -1.0538107, upper bound: 1.0427037
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.52
Output dim: 2, lower bound: -1.0514352, upper bound: 1.0458624
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.52
Output dim: 2, lower bound: -1.0555797, upper bound: 1.0417132
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.52
Output dim: 2, lower bound: -1.0410329, upper bound: 1.0554382
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.52
Output dim: 2, lower bound: -1.0457209, upper bound: 1.0507518
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.52
Output dim: 2, lower bound: -1.0470614, upper bound: 1.0539590
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.52
Output dim: 2, lower bound: -1.0475395, upper bound: 1.0534849

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4477916, 2.4506507
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.4034815, 2.4014225
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4469633, 2.4480300
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5144548, 2.5192990
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.5020781, 2.4922850
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0758448, 2.0811210
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.8483047, 2.8406291
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0689011, 2.0579085
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0942130, 2.1091850
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.1273055, 2.1259203

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4616

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0449578, upper bound: 1.0473821
time: 4.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0491165, upper bound: 1.0432365
time: 8.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4488902, 2.4495511
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.4032965, 2.4016075
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4473896, 2.4476037
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5143576, 2.5193965
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.5024157, 2.4919472
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0754538, 2.0815110
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.8490467, 2.8398871
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0688152, 2.0579934
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0941124, 2.1092858
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.1279306, 2.1252949

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 4616

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 470

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0532233, upper bound: 1.0427023
time: 4.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0538092, upper bound: 1.0421163
time: 4.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4566593, 2.4509077
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.3849821, 2.3888159
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4271364, 2.4250636
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5422239, 2.5392995
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.5142026, 2.5185735
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0743804, 2.0780540
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.7945061, 2.8041821
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0810137, 2.0865278
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0485768, 2.0438035
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.1015744, 2.0958033

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 470

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 903

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0465897, upper bound: 1.0457136
time: 4.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0513015, upper bound: 1.0410256
time: 4.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4563894, 2.4511786
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.3880186, 2.3857784
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4306526, 2.4215407
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5462494, 2.5352676
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.5090413, 2.5237212
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0752454, 2.0771880
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.7856998, 2.8129730
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0765858, 2.0909462
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0590053, 2.0333421
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.0935125, 2.1038537

Time for backsubstitution: 14.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 903

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 65

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0551049, upper bound: 1.0417129
time: 4.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0555793, upper bound: 1.0412399
time: 4.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4459300, 2.4525118
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.4029436, 2.4019613
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4444857, 2.4505076
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5134478, 2.5203066
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.5017786, 2.4925842
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0800810, 2.0768838
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.8533363, 2.8355966
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0684185, 2.0583911
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0941319, 2.1092658
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.1274457, 2.1257801

Time for backsubstitution: 14.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 470

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4639

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0409682, upper bound: 1.0503409
time: 4.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0359360, upper bound: 1.0553732
time: 4.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4470305, 2.4514117
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.4027586, 2.4021463
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4449120, 2.4500813
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5133505, 2.5204041
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.5021162, 2.4922464
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0796919, 2.0772738
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.8540783, 2.8348551
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0683327, 2.0584764
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0940313, 2.1093669
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.1280708, 2.1251550

Time for backsubstitution: 14.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 4639

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 470

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0451335, upper bound: 1.0507501
time: 4.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0457194, upper bound: 1.0501643
time: 4.48 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4561448, 2.4543767
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.4027119, 2.4045835
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4457979, 2.4451590
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5198812, 2.5149410
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.4901738, 2.5003033
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0838575, 2.0781941
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.8435431, 2.8519545
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0585475, 2.0694580
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.1098843, 2.0948157
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.1220269, 2.1240294

Time for backsubstitution: 14.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 4639

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4616

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0428976, upper bound: 1.0539519
time: 5.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0470531, upper bound: 1.0498028
time: 4.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4561372, 2.4543858
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.4027081, 2.4045868
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4457989, 2.4451575
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5198812, 2.5149395
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.4901719, 2.5003042
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0838594, 2.0781918
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.8435373, 2.8519607
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0585504, 2.0694556
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.1098886, 2.0948114
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.1220188, 2.1240373

Time for backsubstitution: 14.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 4616

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4639

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0474748, upper bound: 1.0483876
time: 4.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0424422, upper bound: 1.0534202
time: 4.27 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 23.47 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.47
Output dim: 2, lower bound: -1.0449578, upper bound: 1.0473821
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.47
Output dim: 2, lower bound: -1.0491165, upper bound: 1.0432365
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.47
Output dim: 2, lower bound: -1.0532233, upper bound: 1.0427023
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.47
Output dim: 2, lower bound: -1.0538092, upper bound: 1.0421163
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.47
Output dim: 2, lower bound: -1.0465897, upper bound: 1.0457136
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.47
Output dim: 2, lower bound: -1.0513015, upper bound: 1.0410256
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.47
Output dim: 2, lower bound: -1.0551049, upper bound: 1.0417129
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.47
Output dim: 2, lower bound: -1.0555793, upper bound: 1.0412399
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.47
Output dim: 2, lower bound: -1.0409682, upper bound: 1.0503409
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.47
Output dim: 2, lower bound: -1.0359360, upper bound: 1.0553732
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.47
Output dim: 2, lower bound: -1.0451335, upper bound: 1.0507501
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.47
Output dim: 2, lower bound: -1.0457194, upper bound: 1.0501643
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.47
Output dim: 2, lower bound: -1.0428976, upper bound: 1.0539519
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.47
Output dim: 2, lower bound: -1.0470531, upper bound: 1.0498028
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.47
Output dim: 2, lower bound: -1.0474748, upper bound: 1.0483876
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.47
Output dim: 2, lower bound: -1.0424422, upper bound: 1.0534202

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4464364, 2.4490256
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.3852072, 2.3861847
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4258223, 2.4304051
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5357871, 2.5446563
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.5257835, 2.5108433
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0706000, 2.0767422
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.8042736, 2.7878075
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0908723, 2.0754609
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0328164, 2.0582170
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.1069765, 2.0975409

Time for backsubstitution: 14.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 470

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0443706, upper bound: 1.0473803
time: 4.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0449563, upper bound: 1.0467945
time: 4.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4461656, 2.4492960
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.3882437, 2.3831477
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4293451, 2.4268885
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5398192, 2.5406308
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.5206356, 2.5160050
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0714669, 2.0758772
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.7954826, 2.7966127
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0864539, 2.0798883
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0432782, 2.0477886
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.0989261, 2.1056025

Time for backsubstitution: 14.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 4639

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 65

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0486420, upper bound: 1.0432366
time: 4.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0491162, upper bound: 1.0427636
time: 4.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4422812, 2.4451828
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.3910322, 2.3935070
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4470625, 2.4471126
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5130596, 2.5174353
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.4885283, 2.4827852
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0725956, 2.0771780
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.8477745, 2.8390260
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0661654, 2.0539737
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0929708, 2.1075625
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.1257801, 2.1220303

Time for backsubstitution: 14.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 4639

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 65

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0532225, upper bound: 1.0427018
time: 4.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0532211, upper bound: 1.0421256
time: 4.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4445233, 2.4429412
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.3951969, 2.3893423
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4468985, 2.4472766
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5123968, 2.5180981
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.4932547, 2.4780593
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0711212, 2.0786524
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.8481846, 2.8386149
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0647960, 2.0553422
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0923891, 2.1081440
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.1246657, 2.1231444

Time for backsubstitution: 14.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 4639

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4616

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0496642, upper bound: 1.0421076
time: 4.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0538026, upper bound: 1.0379383
time: 4.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4500566, 2.4454050
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.3838720, 2.3875213
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4289408, 2.4272938
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5417361, 2.5387139
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.5159664, 2.5206747
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0720296, 2.0753140
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.7908392, 2.8012571
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0804572, 2.0858860
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0479703, 2.0430961
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.1048374, 2.0996914

Time for backsubstitution: 14.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 4639

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 470

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0460025, upper bound: 1.0457122
time: 4.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0465882, upper bound: 1.0451262
time: 4.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4511571, 2.4443054
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.3836870, 2.3877058
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4293661, 2.4268675
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5416389, 2.5388117
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.5163040, 2.5203369
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0716395, 2.0757041
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.7915802, 2.8005152
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0803714, 2.0859709
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0478697, 2.0431969
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.1054626, 2.0990663

Time for backsubstitution: 14.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 470

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4639

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0512370, upper bound: 1.0359288
time: 4.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0462104, upper bound: 1.0409614
time: 5.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4563818, 2.4511628
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.3880110, 2.3857670
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4306498, 2.4215398
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5462456, 2.5352654
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.5090308, 2.5237107
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0752425, 2.0771861
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.7856894, 2.8129559
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0765820, 2.0909452
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0589976, 2.0333388
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.0935063, 2.1038394

Time for backsubstitution: 14.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 903

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4639

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0550402, upper bound: 1.0366236
time: 5.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0500076, upper bound: 1.0416483
time: 4.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4563732, 2.4511714
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.3880072, 2.3857703
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4306517, 2.4215379
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5462475, 2.5352640
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.5090308, 2.5237117
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0752444, 2.0771842
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.7856827, 2.8129616
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0765848, 2.0909424
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0590019, 2.0333345
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.0934982, 2.1038475

Time for backsubstitution: 14.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 903

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 470

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0549900, upper bound: 1.0411364
time: 4.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0555778, upper bound: 1.0411253
time: 4.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4457603, 2.4525719
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.4053135, 2.3942938
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4453306, 2.4477677
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5134335, 2.5203111
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.4982138, 2.4936750
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0766153, 2.0779428
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.8537040, 2.8344030
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0690250, 2.0564327
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0964355, 2.1019940
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.1275649, 2.1253963

Time for backsubstitution: 14.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 470

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0403807, upper bound: 1.0503392
time: 4.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0409667, upper bound: 1.0497530
time: 4.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4459300, 2.4523416
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.3952751, 2.4019613
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4417458, 2.4505076
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5134478, 2.5202918
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.5017786, 2.4890196
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0800810, 2.0734181
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.8521438, 2.8355966
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0664597, 2.0583911
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0868602, 2.1092658
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.1270614, 2.1257801

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 4616

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 470

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0353486, upper bound: 1.0553717
time: 4.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0359345, upper bound: 1.0547861
time: 4.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4404206, 2.4470439
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.3904934, 2.3940458
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4445848, 2.4495902
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5120525, 2.5184429
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.4882288, 2.4830844
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0768328, 2.0729408
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.8528061, 2.8339939
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0656819, 2.0544562
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0928898, 2.1076436
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.1259198, 2.1218903

Time for backsubstitution: 14.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 4616

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4639

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0450687, upper bound: 1.0456527
time: 4.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0400361, upper bound: 1.0506853
time: 4.27 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4426618, 2.4448023
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.3946581, 2.3898811
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4444208, 2.4497542
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5113897, 2.5191057
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.4929552, 2.4783585
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0753584, 2.0744157
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.8532181, 2.8335829
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0643134, 2.0558252
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0923080, 2.1082251
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.1248059, 2.1230042

Time for backsubstitution: 14.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4639

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0456547, upper bound: 1.0450667
time: 4.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0406221, upper bound: 1.0500994
time: 4.48 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4547911, 2.4527531
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.3844347, 2.3893437
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4246559, 2.4275403
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5412130, 2.5403051
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.5138946, 2.5188620
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0786128, 2.0738158
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.7995272, 2.7991323
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0805273, 2.0870094
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0484881, 2.0438809
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.1017079, 2.0956488

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 903

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 470

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0427830, upper bound: 1.0539503
time: 4.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0427930, upper bound: 1.0533628
time: 4.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4545212, 2.4530239
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.3874731, 2.3863058
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4281721, 2.4240174
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5452385, 2.5362730
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.5087314, 2.5240099
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0794787, 2.0729494
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.7907209, 2.8079233
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0760994, 2.0914278
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0589166, 2.0334196
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.0936465, 2.1036994

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 470

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 903

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0422168, upper bound: 1.0496650
time: 4.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0469042, upper bound: 1.0449573
time: 4.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4559674, 2.4544463
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.4050794, 2.3969183
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4466429, 2.4424167
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5198679, 2.5149441
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.4866071, 2.5013943
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0803938, 2.0792499
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.8439040, 2.8507676
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0591583, 2.0674973
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.1121922, 2.0875399
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.1221371, 2.1236525

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 903

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 470

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0468868, upper bound: 1.0482887
time: 4.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0474733, upper bound: 1.0482741
time: 4.48 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4561372, 2.4542155
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.3950400, 2.4045868
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4430580, 2.4451575
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5198812, 2.5149248
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.4901719, 2.4967391
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0838594, 2.0747256
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.8423438, 2.8519607
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0565929, 2.0694556
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.1026173, 2.0948114
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.1216340, 2.1240373

Time for backsubstitution: 14.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 903

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4616

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0382869, upper bound: 1.0534129
time: 4.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0424339, upper bound: 1.0492694
time: 4.54 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 23.74 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 2, lower bound: -1.0443706, upper bound: 1.0473803
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 2, lower bound: -1.0449563, upper bound: 1.0467945
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 2, lower bound: -1.0486420, upper bound: 1.0432366
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 2, lower bound: -1.0491162, upper bound: 1.0427636
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 2, lower bound: -1.0532225, upper bound: 1.0427018
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 2, lower bound: -1.0532211, upper bound: 1.0421256
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 2, lower bound: -1.0496642, upper bound: 1.0421076
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 2, lower bound: -1.0538026, upper bound: 1.0379383
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 2, lower bound: -1.0460025, upper bound: 1.0457122
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 2, lower bound: -1.0465882, upper bound: 1.0451262
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 2, lower bound: -1.0512370, upper bound: 1.0359288
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 2, lower bound: -1.0462104, upper bound: 1.0409614
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 2, lower bound: -1.0550402, upper bound: 1.0366236
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 2, lower bound: -1.0500076, upper bound: 1.0416483
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 2, lower bound: -1.0549900, upper bound: 1.0411364
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 2, lower bound: -1.0555778, upper bound: 1.0411253
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 2, lower bound: -1.0403807, upper bound: 1.0503392
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 2, lower bound: -1.0409667, upper bound: 1.0497530
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 2, lower bound: -1.0353486, upper bound: 1.0553717
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 2, lower bound: -1.0359345, upper bound: 1.0547861
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 2, lower bound: -1.0450687, upper bound: 1.0456527
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 2, lower bound: -1.0400361, upper bound: 1.0506853
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 2, lower bound: -1.0456547, upper bound: 1.0450667
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 2, lower bound: -1.0406221, upper bound: 1.0500994
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 2, lower bound: -1.0427830, upper bound: 1.0539503
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 2, lower bound: -1.0427930, upper bound: 1.0533628
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 2, lower bound: -1.0422168, upper bound: 1.0496650
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 2, lower bound: -1.0469042, upper bound: 1.0449573
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 2, lower bound: -1.0468868, upper bound: 1.0482887
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 2, lower bound: -1.0474733, upper bound: 1.0482741
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 2, lower bound: -1.0382869, upper bound: 1.0534129
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 2, lower bound: -1.0424339, upper bound: 1.0492694

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.4398284, 2.4446588
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.3729410, 2.3780837
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.4254956, 2.4299140
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.5344906, 2.5426965
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.5118961, 2.5016813
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0677419, 2.0724096
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.8030024, 2.7869463
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0882215, 2.0714402
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0316753, 2.0564935
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.1048260, 2.0942760

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 4639

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 65

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0443699, upper bound: 1.0473801
time: 4.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.0443684, upper bound: 1.0468038
time: 4.95 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 24.02 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 24.02
Output dim: 2, lower bound: -1.0443699, upper bound: 1.0473801
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 24.02
Output dim: 2, lower bound: -1.0443684, upper bound: 1.0468038
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.02
Output dim: 2, lower bound: -1.0449563, upper bound: 1.0467945
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.02
Output dim: 2, lower bound: -1.0486420, upper bound: 1.0432366
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.02
Output dim: 2, lower bound: -1.0491162, upper bound: 1.0427636
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.02
Output dim: 2, lower bound: -1.0532225, upper bound: 1.0427018
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.02
Output dim: 2, lower bound: -1.0532211, upper bound: 1.0421256
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.02
Output dim: 2, lower bound: -1.0496642, upper bound: 1.0421076
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.02
Output dim: 2, lower bound: -1.0538026, upper bound: 1.0379383
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.02
Output dim: 2, lower bound: -1.0460025, upper bound: 1.0457122
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.02
Output dim: 2, lower bound: -1.0465882, upper bound: 1.0451262
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.02
Output dim: 2, lower bound: -1.0512370, upper bound: 1.0359288
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.02
Output dim: 2, lower bound: -1.0462104, upper bound: 1.0409614
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.02
Output dim: 2, lower bound: -1.0550402, upper bound: 1.0366236
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.02
Output dim: 2, lower bound: -1.0500076, upper bound: 1.0416483
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.02
Output dim: 2, lower bound: -1.0549900, upper bound: 1.0411364
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.02
Output dim: 2, lower bound: -1.0555778, upper bound: 1.0411253
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.02
Output dim: 2, lower bound: -1.0403807, upper bound: 1.0503392
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.02
Output dim: 2, lower bound: -1.0409667, upper bound: 1.0497530
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.02
Output dim: 2, lower bound: -1.0353486, upper bound: 1.0553717
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.02
Output dim: 2, lower bound: -1.0359345, upper bound: 1.0547861
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.02
Output dim: 2, lower bound: -1.0450687, upper bound: 1.0456527
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.02
Output dim: 2, lower bound: -1.0400361, upper bound: 1.0506853
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.02
Output dim: 2, lower bound: -1.0456547, upper bound: 1.0450667
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.02
Output dim: 2, lower bound: -1.0406221, upper bound: 1.0500994
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.02
Output dim: 2, lower bound: -1.0427830, upper bound: 1.0539503
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.02
Output dim: 2, lower bound: -1.0427930, upper bound: 1.0533628
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.02
Output dim: 2, lower bound: -1.0422168, upper bound: 1.0496650
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.02
Output dim: 2, lower bound: -1.0469042, upper bound: 1.0449573
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.02
Output dim: 2, lower bound: -1.0468868, upper bound: 1.0482887
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.02
Output dim: 2, lower bound: -1.0474733, upper bound: 1.0482741
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.02
Output dim: 2, lower bound: -1.0382869, upper bound: 1.0534129
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.02
Output dim: 2, lower bound: -1.0424339, upper bound: 1.0492694
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=2.4546236991882324
rel_dist={2: [-1.0555928908856513, 1.0555926528901223]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4639
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 6184
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4639

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9012838, upper bound: 0.8963065
time: 5.82 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8963070, upper bound: 0.9012833
time: 5.57 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 11.40 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 11.40
Output dim: 2, lower bound: -0.9012838, upper bound: 0.8963065
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 11.40
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

Time for backsubstitution: 15.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 6184
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4616

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8980005, upper bound: 0.8962992
time: 5.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9012773, upper bound: 0.8930226
time: 4.91 seconds

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

Time for backsubstitution: 15.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 6184
type: RSZ, layer: 1, pos: 4616

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 65

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8959586, upper bound: 0.9012830
time: 4.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8963068, upper bound: 0.9009348
time: 4.96 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 25.08 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 25.08
Output dim: 2, lower bound: -0.8980005, upper bound: 0.8962992
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 25.08
Output dim: 2, lower bound: -0.9012773, upper bound: 0.8930226
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 25.08
Output dim: 2, lower bound: -0.8959586, upper bound: 0.9012830
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 25.08
Output dim: 2, lower bound: -0.8963068, upper bound: 0.9009348

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.3466878, 2.3466578
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.2613473, 2.2560959
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.3630342, 2.3629880
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.4577074, 2.4607463
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.4227552, 2.4223752
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0226436, 2.0266876
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.7699065, 2.7621322
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0499759, 2.0447302
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0467949, 2.0474594
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -1.9989138, 1.9924903

Time for backsubstitution: 14.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 6184
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 850

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 903

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8943279, upper bound: 0.8961520
time: 4.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8978780, upper bound: 0.8926318
time: 5.24 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.3464847, 2.3468609
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.2636247, 2.2538180
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.3656759, 2.3603454
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.4607325, 2.4577222
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.4188833, 2.4262466
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0232940, 2.0260382
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.7633023, 2.7687364
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0466552, 2.0480509
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0546412, 2.0396135
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -1.9928675, 1.9985366

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 6184
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 903

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8976095, upper bound: 0.8928989
time: 4.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9011301, upper bound: 0.8893505
time: 4.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.3482752, 2.3480968
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.2720981, 2.2796240
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.3814697, 2.3841586
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.4363661, 2.4363527
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.4037676, 2.4002750
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0312748, 2.0278831
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.8149009, 2.8160677
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0271444, 2.0290704
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.1008849, 2.1080697
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.0208297, 2.0212009

Time for backsubstitution: 14.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 6184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 470

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8958683, upper bound: 0.9012817
time: 4.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8958791, upper bound: 0.9008436
time: 5.24 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.3482695, 2.3481030
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.2720952, 2.2796268
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.3814707, 2.3841577
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.4363680, 2.4363518
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.4037666, 2.4002757
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0312767, 2.0278811
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.8148971, 2.8160720
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0271463, 2.0290685
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.1008883, 2.1080663
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.0208235, 2.0212069

Time for backsubstitution: 14.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6184
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 903

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8954374, upper bound: 0.9009322
time: 5.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8963038, upper bound: 0.9000681
time: 4.80 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 25.16 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 25.16
Output dim: 2, lower bound: -0.8943279, upper bound: 0.8961520
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 25.16
Output dim: 2, lower bound: -0.8978780, upper bound: 0.8926318
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 25.16
Output dim: 2, lower bound: -0.8976095, upper bound: 0.8928989
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.16
Output dim: 2, lower bound: -0.9011301, upper bound: 0.8893505
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.16
Output dim: 2, lower bound: -0.8958683, upper bound: 0.9012817
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.16
Output dim: 2, lower bound: -0.8958791, upper bound: 0.9008436
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.16
Output dim: 2, lower bound: -0.8954374, upper bound: 0.9009322
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.16
Output dim: 2, lower bound: -0.8963038, upper bound: 0.9000681

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.3407083, 2.3402596
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.2623754, 2.2527070
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.3679066, 2.3622561
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.4601455, 2.4572089
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.4209013, 2.4280109
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0206499, 2.0236859
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.7601910, 2.7650681
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0460129, 2.0474725
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0539589, 2.0390072
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -1.9966002, 2.0018001

Time for backsubstitution: 14.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 6184
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 470

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9006920, upper bound: 0.8893492
time: 4.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9011288, upper bound: 0.8889128
time: 4.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.3416662, 2.3431692
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.2598338, 2.2704887
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.3811007, 2.3836675
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.4349051, 2.4343925
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.3898792, 2.3899364
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0280466, 2.0235496
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.8136315, 2.8151145
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0241518, 2.0250506
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0995984, 2.1063468
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.0184007, 2.0179365

Time for backsubstitution: 15.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 6184
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 850

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4616

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8925850, upper bound: 0.9012741
time: 5.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8958617, upper bound: 0.8979981
time: 4.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.3433352, 2.3414879
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.2629571, 2.2673597
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.3809776, 2.3837881
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.4344063, 2.4348898
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.3934231, 2.3863871
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0269413, 2.0246525
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.8139396, 2.8147974
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0231247, 2.0260735
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0991621, 2.1067767
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.0175653, 2.0187719

Time for backsubstitution: 14.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6184
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 850

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8950100, upper bound: 0.9008410
time: 4.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8958762, upper bound: 0.8999746
time: 5.21 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.3454566, 2.3480067
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.2739372, 2.2804666
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.3796005, 2.3846264
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.4365120, 2.4409580
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.3891482, 2.3782842
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0226927, 2.0203695
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.7442493, 2.7353373
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0034714, 1.9975753
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0180874, 2.0366311
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.0265555, 2.0253258

Time for backsubstitution: 14.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 470

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 903

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8917663, upper bound: 0.9007845
time: 4.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8952936, upper bound: 0.8972669
time: 5.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.3481727, 2.3452911
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.2729359, 2.2814679
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.3819389, 2.3822880
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.4409733, 2.4364965
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.3817744, 2.3856578
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0237646, 2.0192976
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.7341623, 2.7454243
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -1.9956532, 2.0053940
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0294528, 2.0252657
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.0249424, 2.0269389

Time for backsubstitution: 14.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 4616

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 850

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8962860, upper bound: 0.8940870
time: 5.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8903298, upper bound: 0.9000514
time: 5.07 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 25.08 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.08
Output dim: 2, lower bound: -0.9006920, upper bound: 0.8893492
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.08
Output dim: 2, lower bound: -0.9011288, upper bound: 0.8889128
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.08
Output dim: 2, lower bound: -0.8925850, upper bound: 0.9012741
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.08
Output dim: 2, lower bound: -0.8958617, upper bound: 0.8979981
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.08
Output dim: 2, lower bound: -0.8950100, upper bound: 0.9008410
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.08
Output dim: 2, lower bound: -0.8958762, upper bound: 0.8999746
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.08
Output dim: 2, lower bound: -0.8917663, upper bound: 0.9007845
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 25.08
Output dim: 2, lower bound: -0.8952936, upper bound: 0.8972669
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 25.08
Output dim: 2, lower bound: -0.8962860, upper bound: 0.8940870
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.08
Output dim: 2, lower bound: -0.8903298, upper bound: 0.9000514

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.3340993, 2.3353310
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.2501101, 2.2435656
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.3675380, 2.3617654
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.4586816, 2.4552481
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.4070129, 2.4176667
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0174227, 2.0193529
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.7589197, 2.7641053
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0430179, 2.0434523
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0526719, 2.0372839
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -1.9941702, 1.9985347

Time for backsubstitution: 14.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 6184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 850

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9006742, upper bound: 0.8833737
time: 4.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8947183, upper bound: 0.8893307
time: 4.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.3357806, 2.3336501
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.2532344, 2.2404423
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.3674150, 2.3618879
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.4581847, 2.4557452
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.4105577, 2.4141221
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0163174, 2.0204587
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.7592278, 2.7637968
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0419917, 2.0444789
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0522361, 2.0377202
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -1.9933348, 1.9993701

Time for backsubstitution: 14.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 6184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 850

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9011109, upper bound: 0.8829373
time: 5.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8951550, upper bound: 0.8888949
time: 4.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.3402433, 2.3415422
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.2415452, 2.2544785
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.3599749, 2.3651838
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.4562578, 2.4587693
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.4123487, 2.4085350
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0228062, 2.0189586
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.7674537, 2.7623320
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0450535, 2.0426326
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0383191, 2.0529134
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -1.9960985, 1.9895883

Time for backsubstitution: 14.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 6184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 903

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8889124, upper bound: 0.9011274
time: 4.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8924620, upper bound: 0.8976081
time: 4.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.3400402, 2.3417449
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.2438226, 2.2522006
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.3626165, 2.3625417
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.4592819, 2.4557452
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.4084787, 2.4124062
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0234566, 2.0183086
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.7608485, 2.7689362
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0417328, 2.0459533
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0461650, 2.0450675
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -1.9900522, 1.9956346

Time for backsubstitution: 14.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6184
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 850

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8949936, upper bound: 0.8979949
time: 4.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8958588, upper bound: 0.8971429
time: 4.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.3405232, 2.3413911
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.2647972, 2.2681990
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.3791084, 2.3842568
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.4345512, 2.4394960
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.3788056, 2.3643956
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0183582, 2.0171413
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.7432899, 2.7340603
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -1.9994493, 1.9945798
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0163603, 2.0353401
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.0232959, 2.0228901

Time for backsubstitution: 14.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 903

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 850

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8949925, upper bound: 0.8948678
time: 5.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8890285, upper bound: 0.9008229
time: 4.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.3432384, 2.3386755
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.2637959, 2.2692003
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.3814468, 2.3819184
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.4390125, 2.4350345
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.3714318, 2.3717692
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0194302, 2.0160694
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.7332029, 2.7441478
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -1.9916310, 2.0023985
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0277262, 2.0239749
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.0216832, 2.0245028

Time for backsubstitution: 14.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 850

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4616

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8925927, upper bound: 0.8999690
time: 4.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8958696, upper bound: 0.8967046
time: 5.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.3388553, 2.3422289
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.2728252, 2.2792172
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.3815103, 2.3868551
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.4359989, 2.4403713
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.3909111, 2.3803000
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0203424, 2.0177269
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.7405825, 2.7322264
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0028944, 1.9969339
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0174799, 2.0359478
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.0298176, 2.0290565

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 4616

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 470

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8913272, upper bound: 0.9007052
time: 4.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8917649, upper bound: 0.9006943
time: 5.16 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.3389897, 2.3375044
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.2760911, 2.2850285
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.3718672, 2.3740740
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.4468584, 2.4431365
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.3833022, 2.3874092
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0014915, 1.9938469
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.7078004, 2.7152886
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -1.9927616, 2.0028644
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0289059, 2.0247796
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.0241008, 2.0259919

Time for backsubstitution: 14.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 470
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 903

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 470

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8898912, upper bound: 0.8999712
time: 4.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8903284, upper bound: 0.8999581
time: 4.96 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 24.32 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.32
Output dim: 2, lower bound: -0.9006742, upper bound: 0.8833737
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.32
Output dim: 2, lower bound: -0.8947183, upper bound: 0.8893307
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.32
Output dim: 2, lower bound: -0.9011109, upper bound: 0.8829373
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.32
Output dim: 2, lower bound: -0.8951550, upper bound: 0.8888949
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.32
Output dim: 2, lower bound: -0.8889124, upper bound: 0.9011274
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.32
Output dim: 2, lower bound: -0.8924620, upper bound: 0.8976081
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.32
Output dim: 2, lower bound: -0.8949936, upper bound: 0.8979949
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.32
Output dim: 2, lower bound: -0.8958588, upper bound: 0.8971429
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.32
Output dim: 2, lower bound: -0.8949925, upper bound: 0.8948678
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.32
Output dim: 2, lower bound: -0.8890285, upper bound: 0.9008229
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.32
Output dim: 2, lower bound: -0.8925927, upper bound: 0.8999690
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.32
Output dim: 2, lower bound: -0.8958696, upper bound: 0.8967046
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.32
Output dim: 2, lower bound: -0.8913272, upper bound: 0.9007052
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.32
Output dim: 2, lower bound: -0.8917649, upper bound: 0.9006943
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.32
Output dim: 2, lower bound: -0.8898912, upper bound: 0.8999712
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.32
Output dim: 2, lower bound: -0.8903284, upper bound: 0.8999581

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.3263125, 2.3261490
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.2536707, 2.2467213
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.3593264, 2.3516955
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.4653225, 2.4611332
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.4087658, 2.4191947
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -1.9919715, 1.9970798
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.7287836, 2.7377434
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0404882, 2.0405593
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0521870, 2.0367382
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -1.9932227, 1.9976921

Time for backsubstitution: 14.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 6184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 65

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9006734, upper bound: 0.8833732
time: 5.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9006723, upper bound: 0.8829439
time: 4.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.3279929, 2.3244677
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.2567940, 2.2435980
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.3592033, 2.3518186
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.4648256, 2.4616303
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.4123106, 2.4156504
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -1.9908662, 1.9981856
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.7290907, 2.7374349
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0394611, 2.0415859
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0517507, 2.0371745
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -1.9923868, 1.9985275

Time for backsubstitution: 14.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6184
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9002459, upper bound: 0.8829345
time: 4.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9011080, upper bound: 0.8820817
time: 4.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.3336396, 2.3357644
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.2404351, 2.2532296
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.3618855, 2.3674140
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.4557447, 2.4581831
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.4141121, 2.4105515
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0204544, 2.0163140
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.7637863, 2.7592206
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0444760, 2.0419912
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0377126, 2.0522311
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -1.9993610, 1.9933195

Time for backsubstitution: 14.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6184
type: RSZ, layer: 1, pos: 850

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8880568, upper bound: 0.9011250
time: 5.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8889094, upper bound: 0.9002629
time: 5.23 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.3313413, 2.3336043
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.2679524, 2.2717586
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.3690386, 2.3760448
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.4404354, 2.4461358
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.3803334, 2.3661475
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -1.9960842, 1.9916897
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.7169280, 2.7039242
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -1.9965572, 1.9920497
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0158143, 2.0348547
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.0224547, 2.0219440

Time for backsubstitution: 14.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 4616

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 903

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8853568, upper bound: 0.9006752
time: 4.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8888868, upper bound: 0.8971555
time: 4.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.3418169, 2.3370514
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.2455211, 2.2532043
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.3603053, 2.3634186
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.4603448, 2.4593911
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.3938613, 2.3903267
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0141869, 2.0114751
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.6869850, 2.6913261
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0125046, 2.0199513
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -1.9663301, 1.9704251
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -1.9993501, 1.9961233

Time for backsubstitution: 14.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 903

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 850

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8925748, upper bound: 0.8939916
time: 5.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8866141, upper bound: 0.8999510
time: 5.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.3322473, 2.3372893
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.2605610, 2.2700753
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.3811398, 2.3863640
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.4345360, 2.4384117
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.3770227, 2.3699565
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0171118, 2.0133934
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.7393103, 2.7312622
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -1.9998965, 1.9929132
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0161862, 2.0342245
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.0273876, 2.0257916

Time for backsubstitution: 14.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 4616

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 850

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8913096, upper bound: 0.8947281
time: 4.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8853491, upper bound: 0.9006881
time: 5.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.3339276, 2.3356204
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.2636890, 2.2669520
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.3810186, 2.3864870
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.4340391, 2.4389110
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.3805723, 2.3664122
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -2.0160084, 2.0144992
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.7396278, 2.7309542
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -1.9988742, 1.9939399
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0157566, 2.0346606
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.0265522, 2.0266271

Time for backsubstitution: 14.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 4616

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 850

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8917474, upper bound: 0.8947205
time: 4.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8857860, upper bound: 0.9006767
time: 4.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.3323808, 2.3325639
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.2638249, 2.2758861
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.3714981, 2.3735847
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.4453945, 2.4411762
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.3694139, 2.3770664
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -1.9982605, 1.9895134
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.7065282, 2.7143245
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -1.9897633, 1.9988437
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0276127, 2.0230563
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.0216713, 2.0227270

Time for backsubstitution: 14.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 903

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4616

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8866065, upper bound: 0.8999630
time: 5.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8898846, upper bound: 0.8966986
time: 5.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.3340631, 2.3308954
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.2669549, 2.2727628
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.3713779, 2.3737073
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.4448977, 2.4416752
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.3729634, 2.3735218
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -1.9971581, 1.9906192
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.7068458, 2.7140164
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -1.9887409, 1.9998703
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0271831, 2.0234926
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -2.0208359, 2.0235629

Time for backsubstitution: 14.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4616
type: RSZ, layer: 1, pos: 903

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4616

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8870433, upper bound: 0.8999523
time: 4.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8903219, upper bound: 0.8966879
time: 4.73 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 24.23 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 24.23
Output dim: 2, lower bound: -0.9006734, upper bound: 0.8833732
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 24.23
Output dim: 2, lower bound: -0.9006723, upper bound: 0.8829439
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 24.23
Output dim: 2, lower bound: -0.9002459, upper bound: 0.8829345
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 24.23
Output dim: 2, lower bound: -0.9011080, upper bound: 0.8820817
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 24.23
Output dim: 2, lower bound: -0.8880568, upper bound: 0.9011250
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 24.23
Output dim: 2, lower bound: -0.8889094, upper bound: 0.9002629
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 24.23
Output dim: 2, lower bound: -0.8853568, upper bound: 0.9006752
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 24.23
Output dim: 2, lower bound: -0.8888868, upper bound: 0.8971555
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 24.23
Output dim: 2, lower bound: -0.8925748, upper bound: 0.8939916
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 24.23
Output dim: 2, lower bound: -0.8866141, upper bound: 0.8999510
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 24.23
Output dim: 2, lower bound: -0.8913096, upper bound: 0.8947281
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 24.23
Output dim: 2, lower bound: -0.8853491, upper bound: 0.9006881
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 24.23
Output dim: 2, lower bound: -0.8917474, upper bound: 0.8947205
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 24.23
Output dim: 2, lower bound: -0.8857860, upper bound: 0.9006767
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 24.23
Output dim: 2, lower bound: -0.8866065, upper bound: 0.8999630
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 24.23
Output dim: 2, lower bound: -0.8898846, upper bound: 0.8966986
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 24.23
Output dim: 2, lower bound: -0.8870433, upper bound: 0.8999523
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 24.23
Output dim: 2, lower bound: -0.8903219, upper bound: 0.8966879

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.3263030, 2.3261328
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.2536631, 2.2467170
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.3593225, 2.3516932
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.4653220, 2.4611316
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.4087553, 2.4191885
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -1.9919682, 1.9970779
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.7287722, 2.7377372
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0404835, 2.0405574
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0521789, 2.0367332
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -1.9932141, 1.9976778

Time for backsubstitution: 14.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8998084, upper bound: 0.8833700
time: 4.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9006705, upper bound: 0.8825176
time: 4.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0963173, -5.8753343, -9.0963173, -5.8753343, -2.3262973, 2.3261266
1: -14.3965416, -11.0441065, -14.3965416, -11.0441065, -2.2536602, 2.2467141
2: 6.4048781, 9.3080320, 6.4048781, 9.3080320, -2.3593216, 2.3516922
3: -5.2136140, -2.5530245, -5.2136140, -2.5530245, -2.4653211, 2.4611306
4: -11.1295519, -7.9597182, -11.1295519, -7.9597182, -2.4087543, 2.4191847
5: -10.7184896, -7.9950051, -10.7184896, -7.9950051, -1.9919662, 1.9970760
6: -13.6051750, -9.5625849, -13.6051750, -9.5625849, -2.7287683, 2.7377324
7: -4.3419952, -1.8644050, -4.3419952, -1.8644050, -2.0404816, 2.0405550
8: -2.1592803, 0.2029767, -2.1592803, 0.2029767, -2.0521755, 2.0367298
9: -9.3684855, -6.3403826, -9.3684855, -6.3403826, -1.9932084, 1.9976835

Time for backsubstitution: 14.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8998072, upper bound: 0.8829412
time: 4.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9006694, upper bound: 0.8820890
time: 4.87 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 23.70 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 23.70
Output dim: 2, lower bound: -0.8998084, upper bound: 0.8833700
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 23.70
Output dim: 2, lower bound: -0.9006705, upper bound: 0.8825176
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 23.70
Output dim: 2, lower bound: -0.8998072, upper bound: 0.8829412
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 23.70
Output dim: 2, lower bound: -0.9006694, upper bound: 0.8820890
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 23.70
Output dim: 2, lower bound: -0.9002459, upper bound: 0.8829345
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 23.70
Output dim: 2, lower bound: -0.9011080, upper bound: 0.8820817
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 23.70
Output dim: 2, lower bound: -0.8880568, upper bound: 0.9011250
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 23.70
Output dim: 2, lower bound: -0.8889094, upper bound: 0.9002629
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 23.70
Output dim: 2, lower bound: -0.8853568, upper bound: 0.9006752
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 23.70
Output dim: 2, lower bound: -0.8866141, upper bound: 0.8999510
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 23.70
Output dim: 2, lower bound: -0.8853491, upper bound: 0.9006881
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 23.70
Output dim: 2, lower bound: -0.8857860, upper bound: 0.9006767
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 23.70
Output dim: 2, lower bound: -0.8866065, upper bound: 0.8999630
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 23.70
Output dim: 2, lower bound: -0.8870433, upper bound: 0.8999523
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=2.384212017059326
rel_dist={2: [-0.9012986819155433, 0.9012981480281343]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2408.33 seconds
