## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 1.0301740019999999
Search space: {k/256 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-6.1949067, -1.9940324, -6.1949067, -1.9940324, -4.2008743, 4.2008743)
1: (-12.2345352, -8.9912519, -12.2345352, -8.9912519, -3.2432833, 3.2432833)
2: (-5.6206846, -2.2354484, -5.6206846, -2.2354484, -3.3852363, 3.3852363)
3: (-5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.8666837, 3.8666837)
4: (-11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.9559875, 3.9559875)
5: (-6.2900958, -3.0817809, -6.2900958, -3.0817809, -3.2083149, 3.2083149)
6: (-12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.7321682, 3.7321682)
7: (-8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888)
8: (7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.3213153, 2.3213153)
9: (-6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.5451798, 3.5451798)

## BASE Result
execution time: IAR + LP analysis = 15.48 + 40.31 = 55.79 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -1.7692351, upper bound: 1.7692330


# Binary Search by BASE starts (time budget: 3544.21 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.321315288543701
rel_dist={8: [-1.3149698422491287, 1.314969004521421]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=2.225489616394043
rel_dist={8: [-1.040580051898944, 1.0405791313826693]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=2.099544048309326
rel_dist={8: [-0.8121649022104638, 0.8121633074202457]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=2.1625168323516846
rel_dist={8: [-0.9337397585648102, 0.9337386492559308]}

## Binary Search Result
Binary search time: 205.06 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Relational Split (RS_dual_Z) starts
Time budget: 3339.15 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4598
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 4598

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3998403, upper bound: 1.3953427
time: 5.87 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3953405, upper bound: 1.3998405
time: 4.48 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 10.58 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 10.58
Output dim: 8, lower bound: -1.3998403, upper bound: 1.3953427
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 10.58
Output dim: 8, lower bound: -1.3953405, upper bound: 1.3998405

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.6921177, 3.6841936
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -3.2432833, 3.2432833
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -3.0473437, 3.0648112
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.8666837, 3.8666837
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.6763020, 3.6402912
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.9538918, 2.9482689
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.3603315, 3.3646135
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.3213153, 2.3213153
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.2207294, 3.2012091

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 5805

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3998367, upper bound: 1.3931595
time: 6.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3976566, upper bound: 1.3953391
time: 6.36 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.6841927, 3.6921186
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -3.2432833, 3.2432833
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -3.0648112, 3.0473442
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.8666837, 3.8666837
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.6402912, 3.6763020
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.9482689, 2.9538913
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.3646135, 3.3603315
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.3213153, 2.3213153
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.2012086, 3.2207298

Time for backsubstitution: 14.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 5805

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3953369, upper bound: 1.3976569
time: 4.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3931592, upper bound: 1.3998369
time: 4.96 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 24.45 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.45
Output dim: 8, lower bound: -1.3998367, upper bound: 1.3931595
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.45
Output dim: 8, lower bound: -1.3976566, upper bound: 1.3953391
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.45
Output dim: 8, lower bound: -1.3953369, upper bound: 1.3976569
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.45
Output dim: 8, lower bound: -1.3931592, upper bound: 1.3998369

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.6801653, 3.6828184
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -3.2432833, 3.2432833
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -3.0331426, 3.0631948
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.8666837, 3.8666837
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.6759844, 3.6374683
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.9531755, 2.9420505
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.3596563, 3.3586950
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.3213153, 2.3213153
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.2204323, 3.1985726

Time for backsubstitution: 14.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 540

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3977480, upper bound: 1.3931584
time: 6.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3998337, upper bound: 1.3910694
time: 6.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.6907434, 3.6722393
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -3.2432833, 3.2432833
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -3.0457273, 3.0506101
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.8666837, 3.8666837
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.6734791, 3.6399736
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.9476728, 2.9475541
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.3544130, 3.3639393
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.3213153, 2.3213153
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.2180929, 3.2009115

Time for backsubstitution: 14.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 540

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3955676, upper bound: 1.3953336
time: 6.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3976535, upper bound: 1.3932492
time: 6.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.6722393, 3.6907434
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -3.2432833, 3.2432833
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -3.0506101, 3.0457277
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.8666837, 3.8666837
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.6399736, 3.6734791
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.9475546, 2.9476728
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.3639393, 3.3544126
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.3213153, 2.3213153
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.2009115, 3.2180939

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 540

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3932484, upper bound: 1.3976541
time: 5.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3953339, upper bound: 1.3955682
time: 5.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.6828184, 3.6801648
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -3.2432833, 3.2432833
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -3.0631948, 3.0331435
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.8666837, 3.8666837
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.6374683, 3.6759844
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.9420500, 2.9531765
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.3586950, 3.3596568
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.3213153, 2.3213153
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.1985722, 3.2204323

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 540

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3910697, upper bound: 1.3998359
time: 6.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3931561, upper bound: 1.3977478
time: 8.21 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 29.05 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 29.05
Output dim: 8, lower bound: -1.3977480, upper bound: 1.3931584
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 29.05
Output dim: 8, lower bound: -1.3998337, upper bound: 1.3910694
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 29.05
Output dim: 8, lower bound: -1.3955676, upper bound: 1.3953336
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 29.05
Output dim: 8, lower bound: -1.3976535, upper bound: 1.3932492
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 29.05
Output dim: 8, lower bound: -1.3932484, upper bound: 1.3976541
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 29.05
Output dim: 8, lower bound: -1.3953339, upper bound: 1.3955682
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 29.05
Output dim: 8, lower bound: -1.3910697, upper bound: 1.3998359
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 29.05
Output dim: 8, lower bound: -1.3931561, upper bound: 1.3977478

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.6667204, 3.6620674
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -3.2432833, 3.2432833
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -3.0310345, 3.0602179
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.8666837, 3.8666837
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.6733160, 3.6355748
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.9408321, 2.9246225
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.3599648, 3.3591118
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.3213153, 2.3213153
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.2171788, 3.1939754

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 891

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3958617, upper bound: 1.3925569
time: 6.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3949451, upper bound: 1.3925542
time: 6.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.6594143, 3.6693740
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -3.2432833, 3.2432833
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -3.0301657, 3.0610862
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.8666837, 3.8666837
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.6740904, 3.6347990
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.9357491, 2.9297066
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.3600745, 3.3590021
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.3213153, 2.3213153
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.2158351, 3.1953187

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 891

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3979793, upper bound: 1.3904460
time: 5.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3970561, upper bound: 1.3904409
time: 6.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.6772995, 3.6514888
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -3.2432833, 3.2432833
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -3.0436192, 3.0476332
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.8666837, 3.8666837
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.6708097, 3.6380801
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.9353294, 2.9301262
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.3547196, 3.3643565
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.3213153, 2.3213153
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.2148395, 3.1963143

Time for backsubstitution: 14.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 891

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3949348, upper bound: 1.3925668
time: 8.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3949375, upper bound: 1.3934885
time: 5.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.6699924, 3.6587954
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -3.2432833, 3.2432833
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -3.0427504, 3.0485015
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.8666837, 3.8666837
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.6715860, 3.6373043
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.9302444, 2.9352102
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.3548303, 3.3642464
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.3213153, 2.3213153
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.2134957, 3.1976576

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 891

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3970460, upper bound: 1.3904514
time: 7.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3970487, upper bound: 1.3913690
time: 12.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.6587954, 3.6699924
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -3.2432833, 3.2432833
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -3.0485010, 3.0427508
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.8666837, 3.8666837
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.6373053, 3.6715856
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.9352112, 2.9302444
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.3642468, 3.3548298
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.3213153, 2.3213153
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.1976571, 3.2134962

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 891

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3913666, upper bound: 1.3970477
time: 9.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3904513, upper bound: 1.3970463
time: 5.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.6514883, 3.6772990
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -3.2432833, 3.2432833
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -3.0476332, 3.0436187
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.8666837, 3.8666837
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.6380796, 3.6708097
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.9301262, 2.9353294
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.3643565, 3.3547201
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.3213153, 2.3213153
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.1963143, 3.2148399

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 891

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3934861, upper bound: 1.3949398
time: 7.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3925644, upper bound: 1.3949358
time: 6.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.6693735, 3.6594143
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -3.2432833, 3.2432833
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -3.0610857, 3.0301661
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.8666837, 3.8666837
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.6347990, 3.6740909
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.9297066, 2.9357481
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.3590016, 3.3600740
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.3213153, 2.3213153
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.1953187, 3.2158356

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 891

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3904410, upper bound: 1.3970556
time: 14.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3904436, upper bound: 1.3979816
time: 6.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.6620674, 3.6667204
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -3.2432833, 3.2432833
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -3.0602179, 3.0310345
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.8666837, 3.8666837
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.6355753, 3.6733150
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.9246216, 2.9408326
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.3591123, 3.3599644
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.3213153, 2.3213153
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.1939750, 3.2171783

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 891

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3925542, upper bound: 1.3949451
time: 7.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3925569, upper bound: 1.3958616
time: 6.87 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 28.79 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.79
Output dim: 8, lower bound: -1.3958617, upper bound: 1.3925569
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.79
Output dim: 8, lower bound: -1.3949451, upper bound: 1.3925542
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.79
Output dim: 8, lower bound: -1.3979793, upper bound: 1.3904460
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.79
Output dim: 8, lower bound: -1.3970561, upper bound: 1.3904409
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.79
Output dim: 8, lower bound: -1.3949348, upper bound: 1.3925668
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.79
Output dim: 8, lower bound: -1.3949375, upper bound: 1.3934885
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.79
Output dim: 8, lower bound: -1.3970460, upper bound: 1.3904514
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.79
Output dim: 8, lower bound: -1.3970487, upper bound: 1.3913690
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.79
Output dim: 8, lower bound: -1.3913666, upper bound: 1.3970477
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.79
Output dim: 8, lower bound: -1.3904513, upper bound: 1.3970463
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.79
Output dim: 8, lower bound: -1.3934861, upper bound: 1.3949398
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.79
Output dim: 8, lower bound: -1.3925644, upper bound: 1.3949358
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.79
Output dim: 8, lower bound: -1.3904410, upper bound: 1.3970556
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.79
Output dim: 8, lower bound: -1.3904436, upper bound: 1.3979816
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.79
Output dim: 8, lower bound: -1.3925542, upper bound: 1.3949451
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.79
Output dim: 8, lower bound: -1.3925569, upper bound: 1.3958616

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.6656647, 3.6640973
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -3.2432833, 3.2432833
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -3.0296659, 3.0628548
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.8666837, 3.8666837
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.6741228, 3.6351538
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.9417028, 2.9241686
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.3612528, 3.3584437
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.3213153, 2.3213153
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.2172503, 3.1939383

Time for backsubstitution: 14.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 6182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3946058, upper bound: 1.3925496
time: 5.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3958539, upper bound: 1.3912853
time: 14.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.6667204, 3.6610112
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -3.2432833, 3.2432833
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -3.0310345, 3.0588489
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.8666837, 3.8666837
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.6728935, 3.6355748
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.9403791, 2.9246225
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.3592958, 3.3591118
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.3213153, 2.3213153
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.2171416, 3.1939754

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3936777, upper bound: 1.3925489
time: 6.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3949370, upper bound: 1.3912827
time: 7.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.6583576, 3.6714034
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -3.2432833, 3.2432833
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -3.0287981, 3.0637231
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.8666837, 3.8666837
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.6748981, 3.6343780
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.9366179, 2.9292536
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.3613625, 3.3583336
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.3213153, 2.3213153
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.2159076, 3.1952820

Time for backsubstitution: 14.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3966968, upper bound: 1.3904356
time: 5.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3979713, upper bound: 1.3891958
time: 6.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.6594143, 3.6683178
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -3.2432833, 3.2432833
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -3.0301657, 3.0597172
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.8666837, 3.8666837
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.6736698, 3.6347990
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.9352951, 2.9297066
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.3594055, 3.3590021
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.3213153, 2.3213153
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.2157979, 3.1953187

Time for backsubstitution: 14.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3957660, upper bound: 1.3904331
time: 9.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3970482, upper bound: 1.3891916
time: 8.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.6762428, 3.6535125
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -3.2432833, 3.2432833
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -3.0422506, 3.0502696
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.8666837, 3.8666837
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.6716175, 3.6376591
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.9361916, 2.9296722
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.3560038, 3.3636880
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.3213153, 2.3213153
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.2149119, 3.1962771

Time for backsubstitution: 14.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3936685, upper bound: 1.3925561
time: 7.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3904356, upper bound: 1.3912927
time: 6.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.6772995, 3.6504331
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -3.2432833, 3.2432833
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -3.0436192, 3.0462646
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.8666837, 3.8666837
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.6703892, 3.6380801
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.9348755, 2.9301262
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.3540516, 3.3643565
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.3213153, 2.3213153
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.2148032, 3.1963143

Time for backsubstitution: 14.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 6182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3936708, upper bound: 1.3934780
time: 8.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3949298, upper bound: 1.3922237
time: 7.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.6689377, 3.6608186
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -3.2432833, 3.2432833
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -3.0413818, 3.0511379
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.8666837, 3.8666837
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.6723938, 3.6368833
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.9311075, 2.9347568
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.3561134, 3.3635778
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.3213153, 2.3213153
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.2135682, 3.1976204

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3957568, upper bound: 1.3904439
time: 6.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3970385, upper bound: 1.3892002
time: 13.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.6699924, 3.6577392
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -3.2432833, 3.2432833
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -3.0427504, 3.0471330
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.8666837, 3.8666837
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.6711645, 3.6373043
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.9297915, 2.9352102
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.3541622, 3.3642464
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.3213153, 2.3213153
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.2134595, 3.1976576

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3957591, upper bound: 1.3913592
time: 7.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3970411, upper bound: 1.3901217
time: 6.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.6577396, 3.6720223
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -3.2432833, 3.2432833
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -3.0471325, 3.0453877
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.8666837, 3.8666837
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.6381121, 3.6711645
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.9360800, 2.9297915
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.3655348, 3.3541617
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.3213153, 2.3213153
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.1977296, 3.2134595

Time for backsubstitution: 14.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6182

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3901212, upper bound: 1.3970415
time: 6.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3913588, upper bound: 1.3957579
time: 11.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.6587954, 3.6689367
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -3.2432833, 3.2432833
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -3.0485010, 3.0413823
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.8666837, 3.8666837
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.6368828, 3.6715856
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.9347563, 2.9302444
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.3635778, 3.3548298
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.3213153, 2.3213153
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.1976199, 3.2134962

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 6182

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3892005, upper bound: 1.3970384
time: 15.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3904431, upper bound: 1.3957566
time: 11.27 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.6504326, 3.6793289
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -3.2432833, 3.2432833
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -3.0462646, 3.0462561
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.8666837, 3.8666837
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.6388874, 3.6703887
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.9309959, 2.9348755
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.3656445, 3.3540516
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.3213153, 2.3213153
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.1963859, 3.2148027

Time for backsubstitution: 14.98 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=2.321315288543701
rel_dist={8: [-1.3998489967280836, 1.3998484636799233]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4598
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 4598

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1366212, upper bound: 1.1327071
time: 14.00 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1327074, upper bound: 1.1366213
time: 11.36 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 25.61 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 25.61
Output dim: 8, lower bound: -1.1366212, upper bound: 1.1327071
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 25.61
Output dim: 8, lower bound: -1.1327074, upper bound: 1.1366213

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.3299580, 3.3254299
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.9428620, 2.9520216
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8256421, 2.8356228
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.6243029, 3.6281815
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.3383865, 3.3178091
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.7335548, 2.7303424
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.0369797, 3.0394268
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2893772, 2.2845201
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.9626756, 2.9515204

Time for backsubstitution: 14.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 5805

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1366194, upper bound: 1.1315111
time: 8.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1354297, upper bound: 1.1327056
time: 11.47 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.3254299, 3.3299580
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.9520211, 2.9428616
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8356233, 2.8256416
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.6281815, 3.6243029
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.3178091, 3.3383865
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.7303429, 2.7335553
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.0394268, 3.0369797
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2845201, 2.2893775
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.9515204, 2.9626751

Time for backsubstitution: 14.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 5805

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1327057, upper bound: 1.1354320
time: 6.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1315118, upper bound: 1.1366191
time: 8.71 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 29.56 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 29.56
Output dim: 8, lower bound: -1.1366194, upper bound: 1.1315111
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 29.56
Output dim: 8, lower bound: -1.1354297, upper bound: 1.1327056
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 29.56
Output dim: 8, lower bound: -1.1327057, upper bound: 1.1354320
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 29.56
Output dim: 8, lower bound: -1.1315118, upper bound: 1.1366191

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.3180046, 3.3195210
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.9411359, 2.9511676
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8114400, 2.8286133
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.6220474, 3.6236234
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.3369961, 3.3149862
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.7304811, 2.7241240
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.0340576, 3.0335078
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2854681, 2.2825918
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.9613752, 2.9488840

Time for backsubstitution: 14.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 540

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1354562, upper bound: 1.1315094
time: 10.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1366177, upper bound: 1.1303481
time: 10.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.3240499, 3.3134756
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.9420075, 2.9502959
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8186326, 2.8214221
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.6197443, 3.6259265
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.3355646, 3.3164182
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.7273359, 2.7272692
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.0310602, 3.0365047
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2874489, 2.2806110
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.9600391, 2.9502211

Time for backsubstitution: 14.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 540

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1342665, upper bound: 1.1327042
time: 7.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1354281, upper bound: 1.1315448
time: 7.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.3134756, 3.3240495
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.9502959, 2.9420071
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8214211, 2.8186321
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.6259270, 3.6197443
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.3164186, 3.3355641
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.7272692, 2.7273369
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.0365047, 3.0310607
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2806106, 2.2874489
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.9502211, 2.9600391

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 540

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1315424, upper bound: 1.1354276
time: 6.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1327040, upper bound: 1.1342660
time: 5.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.3195210, 3.3180046
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.9511676, 2.9411359
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8286138, 2.8114409
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.6236229, 3.6220474
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.3149862, 3.3369961
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.7241240, 2.7304816
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.0335073, 3.0340576
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2825913, 2.2854681
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.9488840, 2.9613757

Time for backsubstitution: 14.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 540

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1303484, upper bound: 1.1366177
time: 8.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1315102, upper bound: 1.1354559
time: 7.91 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 30.69 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 30.69
Output dim: 8, lower bound: -1.1354562, upper bound: 1.1315094
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 30.69
Output dim: 8, lower bound: -1.1366177, upper bound: 1.1303481
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 30.69
Output dim: 8, lower bound: -1.1342665, upper bound: 1.1327042
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 30.69
Output dim: 8, lower bound: -1.1354281, upper bound: 1.1315448
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 30.69
Output dim: 8, lower bound: -1.1315424, upper bound: 1.1354276
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 30.69
Output dim: 8, lower bound: -1.1327040, upper bound: 1.1342660
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 30.69
Output dim: 8, lower bound: -1.1303484, upper bound: 1.1366177
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 30.69
Output dim: 8, lower bound: -1.1315102, upper bound: 1.1354559

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.3014288, 3.2987700
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.9444880, 2.9539533
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8089600, 2.8256364
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.6131115, 3.6161747
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.3343267, 3.3127599
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.7159595, 2.7066956
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.0343652, 3.0338778
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2887855, 2.2853479
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.9575458, 2.9442873

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 891

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1346481, upper bound: 1.1309833
time: 9.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1337480, upper bound: 1.1309814
time: 8.22 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.2972536, 3.3029451
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.9439216, 2.9545197
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8084641, 2.8261323
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.6145992, 3.6146865
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.3347692, 3.3123169
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.7130527, 2.7096014
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.0344281, 3.0338154
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2882242, 2.2859092
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.9567780, 2.9450550

Time for backsubstitution: 14.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 891

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1358120, upper bound: 1.1298186
time: 12.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1349127, upper bound: 1.1298166
time: 10.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.3074741, 3.2927251
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.9453597, 2.9530816
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8161516, 2.8184452
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.6108074, 3.6184783
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.3328943, 3.3141918
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.7128143, 2.7098408
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.0313687, 3.0368748
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2907662, 2.2833672
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.9562097, 2.9456234

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 891

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1337414, upper bound: 1.1309878
time: 12.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1337435, upper bound: 1.1318861
time: 10.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.3032990, 3.2969003
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.9447932, 2.9536481
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8156548, 2.8189411
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.6122952, 3.6169901
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.3333387, 3.3137488
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.7099094, 2.7127461
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.0314307, 3.0368118
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2902050, 2.2839284
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.9554420, 2.9463911

Time for backsubstitution: 14.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 891

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1349061, upper bound: 1.1298231
time: 9.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1349082, upper bound: 1.1307216
time: 8.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.2968998, 3.3032990
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.9536481, 2.9447932
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8189411, 2.8156552
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.6169891, 3.6122961
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.3137484, 3.3333378
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.7127457, 2.7099085
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.0368123, 3.0314312
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2839284, 2.2902052
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.9463906, 2.9554420

Time for backsubstitution: 14.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 891

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1307219, upper bound: 1.1349082
time: 9.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1298234, upper bound: 1.1349056
time: 5.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.2927246, 3.3074737
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.9530816, 2.9453597
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8184452, 2.8161511
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.6184788, 3.6108079
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.3141928, 3.3328948
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.7098408, 2.7128143
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.0368752, 3.0313683
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2833672, 2.2907662
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.9456239, 2.9562097

Time for backsubstitution: 14.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 891

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1318865, upper bound: 1.1337433
time: 9.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1309880, upper bound: 1.1337415
time: 8.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.3029451, 3.2972536
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.9545197, 2.9439216
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8261328, 2.8084641
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.6146870, 3.6145992
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.3123178, 3.3347697
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.7096004, 2.7130537
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.0338149, 3.0344276
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2859092, 2.2882245
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.9450545, 2.9567785

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 891

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1298168, upper bound: 1.1349117
time: 5.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1298189, upper bound: 1.1358112
time: 5.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.2987700, 3.3014288
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.9539533, 2.9444880
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8256359, 2.8089600
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.6161747, 3.6131110
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.3127604, 3.3343263
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.7066956, 2.7159586
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.0338778, 3.0343647
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2853479, 2.2887855
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.9442868, 2.9575462

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 891

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1309814, upper bound: 1.1337503
time: 6.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1309835, upper bound: 1.1346504
time: 10.50 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 31.20 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 31.20
Output dim: 8, lower bound: -1.1346481, upper bound: 1.1309833
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 31.20
Output dim: 8, lower bound: -1.1337480, upper bound: 1.1309814
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 31.20
Output dim: 8, lower bound: -1.1358120, upper bound: 1.1298186
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 31.20
Output dim: 8, lower bound: -1.1349127, upper bound: 1.1298166
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 31.20
Output dim: 8, lower bound: -1.1337414, upper bound: 1.1309878
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 31.20
Output dim: 8, lower bound: -1.1337435, upper bound: 1.1318861
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 31.20
Output dim: 8, lower bound: -1.1349061, upper bound: 1.1298231
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 31.20
Output dim: 8, lower bound: -1.1349082, upper bound: 1.1307216
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 31.20
Output dim: 8, lower bound: -1.1307219, upper bound: 1.1349082
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 31.20
Output dim: 8, lower bound: -1.1298234, upper bound: 1.1349056
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 31.20
Output dim: 8, lower bound: -1.1318865, upper bound: 1.1337433
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 31.20
Output dim: 8, lower bound: -1.1309880, upper bound: 1.1337415
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 31.20
Output dim: 8, lower bound: -1.1298168, upper bound: 1.1349117
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 31.20
Output dim: 8, lower bound: -1.1298189, upper bound: 1.1358112
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 31.20
Output dim: 8, lower bound: -1.1309814, upper bound: 1.1337503
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 31.20
Output dim: 8, lower bound: -1.1309835, upper bound: 1.1346504

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.3003731, 3.2994771
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.9444051, 2.9540086
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8075914, 2.8265567
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.6132984, 3.6158981
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.3346071, 3.3123388
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.7162619, 2.7062426
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.0348148, 3.0332098
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2884688, 2.2855608
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.9575715, 2.9442501

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 6182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1338651, upper bound: 1.1309843
time: 6.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1346466, upper bound: 1.1301622
time: 5.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.3014288, 3.2977138
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.9444880, 2.9538693
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8089600, 2.8242679
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.6128330, 3.6161747
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.3339052, 3.3127599
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.7155056, 2.7066956
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.0336971, 3.0338778
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2887855, 2.2850313
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.9575095, 2.9442873

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1329586, upper bound: 1.1309794
time: 7.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1337465, upper bound: 1.1301584
time: 6.46 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.2961979, 3.3036523
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.9438376, 2.9545751
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8070955, 2.8270526
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.6147861, 3.6144099
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.3350506, 3.3118958
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.7133560, 2.7091475
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.0348778, 3.0331469
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2879076, 2.2861218
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.9568038, 2.9450178

Time for backsubstitution: 14.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 6182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1350214, upper bound: 1.1298169
time: 8.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1358104, upper bound: 1.1290049
time: 9.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.2972536, 3.3018889
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.9439216, 2.9544358
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8084641, 2.8247638
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.6143227, 3.6146865
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.3343487, 3.3123169
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.7125998, 2.7096014
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.0337601, 3.0338154
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2882242, 2.2855923
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.9567409, 2.9450550

Time for backsubstitution: 14.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 6182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1341146, upper bound: 1.1298145
time: 5.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1349112, upper bound: 1.1290044
time: 6.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.3064175, 3.2934289
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.9452758, 2.9531350
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8147821, 2.8193650
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.6109924, 3.6182017
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.3331757, 3.3137708
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.7131128, 2.7093873
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.0318155, 3.0362062
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2904496, 2.2835801
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.9562345, 2.9455867

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 6182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1329532, upper bound: 1.1309861
time: 6.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1337398, upper bound: 1.1301633
time: 8.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.3074741, 3.2916694
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.9453597, 2.9529977
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8161516, 2.8170762
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.6105309, 3.6184783
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.3324738, 3.3141918
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.7123604, 2.7098408
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.0306997, 3.0368748
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2907662, 2.2830505
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.9561725, 2.9456234

Time for backsubstitution: 14.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 6182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1290079, upper bound: 1.1318873
time: 8.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1337419, upper bound: 1.1310732
time: 6.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.3022423, 3.2976041
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.9447093, 2.9537015
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8142862, 2.8198609
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.6124802, 3.6167135
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.3336191, 3.3133278
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.7102070, 2.7122927
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.0318785, 3.0361433
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2898884, 2.2841411
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.9554667, 2.9463544

Time for backsubstitution: 14.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1341093, upper bound: 1.1298215
time: 10.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1349044, upper bound: 1.1290092
time: 11.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.3032990, 3.2958441
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.9447932, 2.9535642
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8156548, 2.8175726
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.6120186, 3.6169901
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.3329163, 3.3137488
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.7094555, 2.7127461
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.0307627, 3.0368118
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2902050, 2.2836115
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.9554048, 2.9463911

Time for backsubstitution: 14.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1341107, upper bound: 1.1307201
time: 9.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1349066, upper bound: 1.1299176
time: 8.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.2958450, 3.3040061
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.9535642, 2.9448485
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8175726, 2.8165755
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.6171780, 3.6120195
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.3140297, 3.3329167
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.7130489, 2.7094555
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.0372620, 3.0307627
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2836118, 2.2904181
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.9464164, 2.9554052

Time for backsubstitution: 14.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6182

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1299176, upper bound: 1.1349061
time: 8.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1307203, upper bound: 1.1341131
time: 6.31 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 29.16 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.16
Output dim: 8, lower bound: -1.1338651, upper bound: 1.1309843
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.16
Output dim: 8, lower bound: -1.1346466, upper bound: 1.1301622
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.16
Output dim: 8, lower bound: -1.1329586, upper bound: 1.1309794
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.16
Output dim: 8, lower bound: -1.1337465, upper bound: 1.1301584
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.16
Output dim: 8, lower bound: -1.1350214, upper bound: 1.1298169
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.16
Output dim: 8, lower bound: -1.1358104, upper bound: 1.1290049
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.16
Output dim: 8, lower bound: -1.1341146, upper bound: 1.1298145
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.16
Output dim: 8, lower bound: -1.1349112, upper bound: 1.1290044
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.16
Output dim: 8, lower bound: -1.1329532, upper bound: 1.1309861
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.16
Output dim: 8, lower bound: -1.1337398, upper bound: 1.1301633
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.16
Output dim: 8, lower bound: -1.1290079, upper bound: 1.1318873
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.16
Output dim: 8, lower bound: -1.1337419, upper bound: 1.1310732
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.16
Output dim: 8, lower bound: -1.1341093, upper bound: 1.1298215
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.16
Output dim: 8, lower bound: -1.1349044, upper bound: 1.1290092
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.16
Output dim: 8, lower bound: -1.1341107, upper bound: 1.1307201
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.16
Output dim: 8, lower bound: -1.1349066, upper bound: 1.1299176
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.16
Output dim: 8, lower bound: -1.1299176, upper bound: 1.1349061
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.16
Output dim: 8, lower bound: -1.1307203, upper bound: 1.1341131
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.16
Output dim: 8, lower bound: -1.1298234, upper bound: 1.1349056
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.16
Output dim: 8, lower bound: -1.1318865, upper bound: 1.1337433
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.16
Output dim: 8, lower bound: -1.1309880, upper bound: 1.1337415
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.16
Output dim: 8, lower bound: -1.1298168, upper bound: 1.1349117
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.16
Output dim: 8, lower bound: -1.1298189, upper bound: 1.1358112
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.16
Output dim: 8, lower bound: -1.1309814, upper bound: 1.1337503
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.16
Output dim: 8, lower bound: -1.1309835, upper bound: 1.1346504
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=2.2884626388549805
rel_dist={8: [-1.136630455303056, 1.136628973869513]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4598
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 4598

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0375874, upper bound: 1.0375843
time: 5.45 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0375849, upper bound: 1.0405755
time: 15.90 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 21.58 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 21.58
Output dim: 8, lower bound: -1.0375874, upper bound: 1.0375843
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 21.58
Output dim: 8, lower bound: -1.0375849, upper bound: 1.0405755

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.2092381, 3.2058415
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.8420043, 2.8488741
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.7517409, 2.7592268
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.5368910, 3.5398006
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.2257481, 3.2103152
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.6601105, 2.6577005
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -2.9291954, 2.9310312
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2251902, 2.2215474
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.8766570, 2.8682909

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 5805

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0405717, upper bound: 1.0366686
time: 8.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0396696, upper bound: 1.0375861
time: 5.78 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.2058411, 3.2092381
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.8488746, 2.8420043
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.7592263, 2.7517409
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.5398006, 3.5368910
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.2103148, 3.2257481
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.6577005, 2.6601100
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -2.9310312, 2.9291959
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2215471, 2.2251904
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.8682904, 2.8766575

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 5805

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0375835, upper bound: 1.0396720
time: 5.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0366687, upper bound: 1.0405715
time: 8.99 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 29.39 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 29.39
Output dim: 8, lower bound: -1.0405717, upper bound: 1.0366686
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 29.39
Output dim: 8, lower bound: -1.0396696, upper bound: 1.0375861
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 29.39
Output dim: 8, lower bound: -1.0375835, upper bound: 1.0396720
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 29.39
Output dim: 8, lower bound: -1.0366687, upper bound: 1.0405715

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.1972837, 3.1984215
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.8402791, 2.8478022
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.7375398, 2.7504191
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.5340605, 3.5352421
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.2240000, 3.2074928
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.6562490, 2.6514816
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -2.9255247, 2.9251118
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2212811, 2.2191238
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.8750229, 2.8656549

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 540

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0397023, upper bound: 1.0366674
time: 9.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0405703, upper bound: 1.0358045
time: 7.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.2018175, 3.1938877
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.8409324, 2.8471489
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.7429338, 2.7450261
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.5323334, 3.5369697
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.2229261, 3.2085662
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.6538916, 2.6538405
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -2.9232769, 2.9273596
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2227664, 2.2176380
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.8740206, 2.8666573

Time for backsubstitution: 14.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 540

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0388006, upper bound: 1.0375815
time: 6.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0396681, upper bound: 1.0367188
time: 8.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.1938877, 3.2018180
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.8471484, 2.8409324
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.7450261, 2.7429338
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.5369701, 3.5323329
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.2085657, 3.2229257
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.6538401, 2.6538916
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -2.9273596, 2.9232769
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2176380, 2.2227666
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.8666573, 2.8740211

Time for backsubstitution: 14.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 540

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0367191, upper bound: 1.0396687
time: 6.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0367217, upper bound: 1.0387999
time: 6.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.1984215, 3.1972842
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.8478026, 2.8402786
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.7504201, 2.7375402
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.5352421, 3.5340605
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.2074928, 3.2239995
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.6514826, 2.6562500
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -2.9251118, 2.9255247
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2191234, 2.2212811
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.8656549, 2.8750234

Time for backsubstitution: 14.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 540

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0358044, upper bound: 1.0405728
time: 6.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0366673, upper bound: 1.0397033
time: 6.55 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 27.93 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.93
Output dim: 8, lower bound: -1.0397023, upper bound: 1.0366674
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.93
Output dim: 8, lower bound: -1.0405703, upper bound: 1.0358045
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.93
Output dim: 8, lower bound: -1.0388006, upper bound: 1.0375815
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.93
Output dim: 8, lower bound: -1.0396681, upper bound: 1.0367188
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.93
Output dim: 8, lower bound: -1.0367191, upper bound: 1.0396687
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.93
Output dim: 8, lower bound: -1.0367217, upper bound: 1.0387999
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.93
Output dim: 8, lower bound: -1.0358044, upper bound: 1.0405728
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.93
Output dim: 8, lower bound: -1.0366673, upper bound: 1.0397033

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.1796646, 3.1776705
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.8434892, 2.8505883
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.7349348, 2.7474422
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.5251236, 3.5274215
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.2213297, 3.2051554
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.6410007, 2.6340537
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -2.9258313, 2.9254665
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2244582, 2.2218800
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.8710017, 2.8610578

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 891

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0391774, upper bound: 1.0361492
time: 5.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0382801, upper bound: 1.0361509
time: 7.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.1765327, 3.1808023
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.8430638, 2.8510127
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.7345629, 2.7478147
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.5262394, 3.5263057
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.2216616, 3.2048230
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.6388226, 2.6362324
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -2.9258790, 2.9254193
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2240372, 2.2223008
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.8704257, 2.8616333

Time for backsubstitution: 14.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 891

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0400172, upper bound: 1.0352895
time: 6.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0391481, upper bound: 1.0352879
time: 6.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.1841984, 3.1731372
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.8441424, 2.8499341
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.7403288, 2.7420487
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.5233955, 3.5291495
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.2202559, 3.2062292
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.6386433, 2.6364126
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -2.9235845, 2.9277143
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2259436, 2.2203944
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.8699994, 2.8620601

Time for backsubstitution: 14.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 891

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0382751, upper bound: 1.0361559
time: 5.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0382767, upper bound: 1.0370382
time: 8.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.1810665, 3.1762686
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.8437181, 2.8503594
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.7399569, 2.7424212
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.5245132, 3.5280328
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.2205896, 3.2058969
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.6364632, 2.6385913
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -2.9236312, 2.9276667
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2255225, 2.2208152
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.8694234, 2.8626356

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 891

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0391431, upper bound: 1.0352903
time: 7.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0391447, upper bound: 1.0361923
time: 7.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.1762676, 3.1810675
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.8503594, 2.8437181
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.7424212, 2.7399564
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.5280323, 3.5245128
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.2058973, 3.2205887
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.6385918, 2.6364632
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -2.9276671, 2.9236317
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2208152, 2.2255230
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.8626351, 2.8694239

Time for backsubstitution: 14.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 891

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0361896, upper bound: 1.0391446
time: 8.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0352906, upper bound: 1.0391456
time: 6.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.1731367, 3.1841984
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.8499341, 2.8441429
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.7420492, 2.7403283
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.5291500, 3.5233960
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.2062292, 3.2202563
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.6364117, 2.6386423
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -2.9277139, 2.9235840
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2203941, 2.2259438
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.8620601, 2.8699994

Time for backsubstitution: 14.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 891

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0370377, upper bound: 1.0382758
time: 4.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0361535, upper bound: 1.0382743
time: 7.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.1808014, 3.1765337
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.8510127, 2.8430643
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.7478142, 2.7345634
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.5263062, 3.5262399
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.2048235, 3.2216620
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.6362324, 2.6388221
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -2.9254193, 2.9258790
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2223005, 2.2240374
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.8616338, 2.8704262

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 891

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0352855, upper bound: 1.0391484
time: 6.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0352871, upper bound: 1.0400176
time: 5.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.1776705, 3.1796646
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.8505883, 2.8434892
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.7474422, 2.7349353
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.5274220, 3.5251241
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.2051554, 3.2213302
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.6340542, 2.6410007
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -2.9254670, 2.9258318
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2218800, 2.2244582
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.8610578, 2.8710022

Time for backsubstitution: 14.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 891

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0352879, upper bound: 1.0382793
time: 6.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0361500, upper bound: 1.0391768
time: 5.43 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 26.11 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.11
Output dim: 8, lower bound: -1.0391774, upper bound: 1.0361492
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.11
Output dim: 8, lower bound: -1.0382801, upper bound: 1.0361509
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.11
Output dim: 8, lower bound: -1.0400172, upper bound: 1.0352895
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.11
Output dim: 8, lower bound: -1.0391481, upper bound: 1.0352879
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.11
Output dim: 8, lower bound: -1.0382751, upper bound: 1.0361559
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.11
Output dim: 8, lower bound: -1.0382767, upper bound: 1.0370382
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.11
Output dim: 8, lower bound: -1.0391431, upper bound: 1.0352903
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.11
Output dim: 8, lower bound: -1.0391447, upper bound: 1.0361923
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.11
Output dim: 8, lower bound: -1.0361896, upper bound: 1.0391446
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.11
Output dim: 8, lower bound: -1.0352906, upper bound: 1.0391456
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.11
Output dim: 8, lower bound: -1.0370377, upper bound: 1.0382758
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.11
Output dim: 8, lower bound: -1.0361535, upper bound: 1.0382743
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.11
Output dim: 8, lower bound: -1.0352855, upper bound: 1.0391484
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.11
Output dim: 8, lower bound: -1.0352871, upper bound: 1.0400176
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.11
Output dim: 8, lower bound: -1.0352879, upper bound: 1.0382793
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.11
Output dim: 8, lower bound: -1.0361500, upper bound: 1.0391768

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.1786098, 3.1779370
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.8434052, 2.8506083
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.7335663, 2.7477903
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.5251961, 3.5271449
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.2214355, 3.2047343
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.6411142, 2.6336002
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -2.9260025, 2.9247980
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2241416, 2.2219603
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.8710113, 2.8610206

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 6182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0384135, upper bound: 1.0361508
time: 6.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0391759, upper bound: 1.0353804
time: 7.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.1796646, 3.1766148
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.8434892, 2.8505044
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.7349348, 2.7460737
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.5248470, 3.5274215
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.2209091, 3.2051554
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.6405478, 2.6340537
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -2.9251633, 2.9254665
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2244582, 2.2215633
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.8709645, 2.8610578

Time for backsubstitution: 14.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 6182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0375497, upper bound: 1.0361491
time: 5.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0382786, upper bound: 1.0353792
time: 19.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.1754780, 3.1810684
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.8429799, 2.8510337
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.7331944, 2.7481627
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.5263119, 3.5260291
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.2217674, 3.2044020
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.6389360, 2.6357794
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -2.9260492, 2.9247513
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2237206, 2.2223811
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.8704352, 2.8615961

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0392359, upper bound: 1.0352853
time: 8.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0400155, upper bound: 1.0345148
time: 6.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.1765327, 3.1797462
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.8430638, 2.8509293
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.7345629, 2.7464457
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.5259628, 3.5263057
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.2212410, 3.2048230
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.6383686, 2.6362324
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -2.9252110, 2.9254193
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2240372, 2.2219839
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.8703895, 2.8616333

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0384126, upper bound: 1.0352862
time: 5.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0391466, upper bound: 1.0345136
time: 10.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.1831436, 3.1734009
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.8440595, 2.8499537
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.7389603, 2.7423968
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.5234661, 3.5288730
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.2203617, 3.2058082
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.6387529, 2.6359587
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -2.9237528, 2.9270458
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2256269, 2.2204747
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.8700089, 2.8620234

Time for backsubstitution: 14.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0375452, upper bound: 1.0361518
time: 18.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0382735, upper bound: 1.0353809
time: 8.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.1841984, 3.1720810
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.8441424, 2.8498507
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.7403288, 2.7406802
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.5231190, 3.5291495
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.2198353, 3.2062292
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.6381884, 2.6364126
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -2.9229164, 2.9277143
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2259436, 2.2200775
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.8699622, 2.8620601

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 6182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0375464, upper bound: 1.0370358
time: 8.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0382752, upper bound: 1.0362371
time: 9.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.1800117, 3.1765323
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.8436341, 2.8503785
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.7385874, 2.7427688
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.5245819, 3.5277562
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.2206936, 3.2054758
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.6365738, 2.6381378
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -2.9237995, 2.9269986
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2252059, 2.2208955
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.8694339, 2.8625989

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0384081, upper bound: 1.0352888
time: 9.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0391415, upper bound: 1.0345207
time: 9.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.1810665, 3.1752124
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.8437181, 2.8502755
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.7399569, 2.7410522
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.5242348, 3.5280328
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.2201672, 3.2058969
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.6360102, 2.6385913
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -2.9229631, 2.9276667
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2255225, 2.2204983
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.8693862, 2.8626356

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0384094, upper bound: 1.0361908
time: 5.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0391432, upper bound: 1.0354166
time: 8.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.1752129, 3.1813340
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.8502755, 2.8437386
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.7410526, 2.7403045
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.5281048, 3.5242362
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.2060022, 3.2201676
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.6387053, 2.6360097
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -2.9278374, 2.9229631
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2204986, 2.2256033
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.8626456, 2.8693867

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6182

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0354166, upper bound: 1.0391424
time: 7.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0361881, upper bound: 1.0384084
time: 5.18 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.1762676, 3.1800113
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.8503594, 2.8436341
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.7424212, 2.7385879
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.5277557, 3.5245128
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.2054758, 3.2205887
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.6381378, 2.6364632
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -2.9269981, 2.9236317
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2208152, 2.2252061
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.8625989, 2.8694239

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 6182

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0345183, upper bound: 1.0391418
time: 7.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0352889, upper bound: 1.0384106
time: 5.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.1720810, 3.1844649
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.8498502, 2.8441634
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.7406797, 2.7406764
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.5292206, 3.5231194
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.2063351, 3.2198353
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.6365261, 2.6381888
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -2.9278841, 2.9229159
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2200775, 2.2260242
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.8620696, 2.8699627

Time for backsubstitution: 14.26 seconds
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=2.225489616394043
rel_dist={8: [-1.0405825564714881, 1.040579234884591]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2412.57 seconds
