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
execution time: IAR + LP analysis = 14.32 + 40.15 = 54.46 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -1.7692351, upper bound: 1.7692330


# Binary Search by BASE starts (time budget: 3545.54 seconds, max iter: 100)

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
rel_dist={8: [-0.9337424670244374, 0.9337387388635623]}

## Binary Search Result
Binary search time: 202.47 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Relational Split (RS_random_Z) starts
Time budget: 3343.06 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 4598
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 835

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3977010, upper bound: 1.3989970
time: 6.95 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3989950, upper bound: 1.3976983
time: 4.53 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 11.49 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 11.49
Output dim: 8, lower bound: -1.3977010, upper bound: 1.3989970
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 11.49
Output dim: 8, lower bound: -1.3989950, upper bound: 1.3976983

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.7113295, 3.7109189
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -3.2304363, 3.2338533
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -3.1012440, 3.1013436
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.8666837, 3.8666837
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.7636623, 3.7622371
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.9646869, 2.9674006
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.3652778, 3.3670282
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.3213153, 2.3213153
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.2682214, 3.2669721

Time for backsubstitution: 14.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 4598
type: RSZ, layer: 1, pos: 891

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 135

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3963218, upper bound: 1.3989968
time: 6.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3976985, upper bound: 1.3975987
time: 6.46 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.7109184, 3.7113299
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -3.2338533, 3.2304363
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -3.1013432, 3.1012435
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.8666837, 3.8666837
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.7622375, 3.7636619
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.9674010, 2.9646864
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.3670287, 3.3652768
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.3213153, 2.3213153
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.2669721, 3.2682214

Time for backsubstitution: 14.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 4598
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 135

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3975965, upper bound: 1.3976979
time: 5.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3989947, upper bound: 1.3963192
time: 5.09 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 24.97 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.97
Output dim: 8, lower bound: -1.3963218, upper bound: 1.3989968
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.97
Output dim: 8, lower bound: -1.3976985, upper bound: 1.3975987
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.97
Output dim: 8, lower bound: -1.3975965, upper bound: 1.3976979
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.97
Output dim: 8, lower bound: -1.3989947, upper bound: 1.3963192

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.7113295, 3.7109189
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -3.2304339, 3.2338514
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -3.1012421, 3.1013432
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.8666837, 3.8666837
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.7636604, 3.7622361
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.9646821, 2.9673944
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.3652697, 3.3670192
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.3213153, 2.3213153
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.2682214, 3.2669716

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4598
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 6182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5749

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3952909, upper bound: 1.3989912
time: 7.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3952917, upper bound: 1.3966350
time: 6.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.7113295, 3.7109189
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -3.2304339, 3.2338505
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -3.1012421, 3.1013432
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.8666837, 3.8666837
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.7636604, 3.7622361
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.9646802, 2.9673963
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.3652678, 3.3670211
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.3213153, 2.3213153
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.2682214, 3.2669716

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4598
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 5749

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4598

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3976898, upper bound: 1.3930785
time: 6.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3931901, upper bound: 1.3975883
time: 5.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.7109194, 3.7113299
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -3.2338500, 3.2304344
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -3.1013432, 3.1012430
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.8666837, 3.8666837
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.7622356, 3.7636609
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.9673963, 2.9646802
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.3670206, 3.3652678
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.3213153, 2.3213153
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.2669721, 3.2682209

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 4598

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3963690, upper bound: 1.3976975
time: 7.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3963711, upper bound: 1.3964727
time: 6.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.7109194, 3.7113299
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -3.2338510, 3.2304335
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -3.1013432, 3.1012430
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.8666837, 3.8666837
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.7622356, 3.7636609
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.9673944, 2.9646826
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.3670187, 3.3652697
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.3213153, 2.3213153
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.2669721, 3.2682209

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 4598

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6182

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3977416, upper bound: 1.3963122
time: 5.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3989869, upper bound: 1.3950528
time: 5.39 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 24.97 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.97
Output dim: 8, lower bound: -1.3952909, upper bound: 1.3989912
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.97
Output dim: 8, lower bound: -1.3952917, upper bound: 1.3966350
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.97
Output dim: 8, lower bound: -1.3976898, upper bound: 1.3930785
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.97
Output dim: 8, lower bound: -1.3931901, upper bound: 1.3975883
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.97
Output dim: 8, lower bound: -1.3963690, upper bound: 1.3976975
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.97
Output dim: 8, lower bound: -1.3963711, upper bound: 1.3964727
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.97
Output dim: 8, lower bound: -1.3977416, upper bound: 1.3963122
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.97
Output dim: 8, lower bound: -1.3989869, upper bound: 1.3950528

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.7036171, 3.7000318
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -3.2085381, 3.2194443
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -3.1064882, 3.1094728
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.8666837, 3.8666837
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.7554550, 3.7446723
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.9343710, 2.9459224
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.3504782, 3.3565340
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.3213153, 2.3213153
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.2563944, 3.2502770

Time for backsubstitution: 14.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4598
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 5805

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3939639, upper bound: 1.3989837
time: 16.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3952831, upper bound: 1.3977388
time: 8.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.7004423, 3.7032037
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -3.2160263, 3.2119560
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -3.1093702, 3.1065888
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.8666837, 3.8666837
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.7460976, 3.7540302
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.9432116, 2.9370828
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.3547802, 3.3522272
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.3213153, 2.3213153
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.2515259, 3.2551441

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4598
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 6182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4598

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3952830, upper bound: 1.3921417
time: 6.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3907951, upper bound: 1.3966264
time: 5.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.6920958, 3.6837606
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -3.2418575, 3.2432833
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -3.0473437, 3.0649109
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.8666837, 3.8666837
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.6762438, 3.6388078
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.9510479, 2.9481411
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.3584857, 3.3645215
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.3213153, 2.3213153
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.2206726, 3.1999025

Time for backsubstitution: 14.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5749

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3966258, upper bound: 1.3930741
time: 7.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3966267, upper bound: 1.3907974
time: 6.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.6841707, 3.6916857
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -3.2432833, 3.2432833
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -3.0648112, 3.0474439
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.8666837, 3.8666837
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.6402330, 3.6748185
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.9454250, 2.9537630
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.3627677, 3.3602395
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.3213153, 2.3213153
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.2011518, 3.2194233

Time for backsubstitution: 14.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 540

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3919630, upper bound: 1.3975870
time: 10.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3931898, upper bound: 1.3963604
time: 4.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.7110620, 3.7114367
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -3.2336044, 3.2300816
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -3.1014214, 3.1011848
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.8666837, 3.8666837
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.7638445, 3.7658057
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.9674101, 2.9646983
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.3662643, 3.3646812
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.3213153, 2.3213153
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.2683654, 3.2691746

Time for backsubstitution: 14.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 4598
type: RSZ, layer: 1, pos: 891

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5749

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3940557, upper bound: 1.3966352
time: 6.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3963660, upper bound: 1.3966341
time: 4.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.7110257, 3.7114720
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -3.2334976, 3.2301879
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -3.1012859, 3.1013212
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.8666837, 3.8666837
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.7643814, 3.7652693
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.9674139, 2.9646940
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.3664341, 3.3645105
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.3213153, 2.3213153
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.2679248, 3.2696147

Time for backsubstitution: 14.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 4598

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 891

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3957575, upper bound: 1.3941868
time: 4.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3953015, upper bound: 1.3946424
time: 4.47 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.7066345, 3.7130246
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -3.2316461, 3.2312984
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -3.1021414, 3.0992250
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.8666837, 3.8666837
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.7630758, 3.7615700
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.9651484, 2.9655805
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.3667746, 3.3653679
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.3213153, 2.3213153
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.2686391, 3.2640715

Time for backsubstitution: 14.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4598
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 5749

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 891

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3958772, upper bound: 1.3940422
time: 6.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3954243, upper bound: 1.3944997
time: 7.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.7109194, 3.7070446
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -3.2338510, 3.2282281
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -3.0993252, 3.1012430
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.8666837, 3.8666837
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.7601452, 3.7636609
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.9673944, 2.9624362
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.3670187, 3.3650246
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.3213153, 2.3213153
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.2628217, 3.2682209

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 4598
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 5749

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3977686, upper bound: 1.3950525
time: 6.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3989866, upper bound: 1.3938054
time: 4.90 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 25.42 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.42
Output dim: 8, lower bound: -1.3939639, upper bound: 1.3989837
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.42
Output dim: 8, lower bound: -1.3952831, upper bound: 1.3977388
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.42
Output dim: 8, lower bound: -1.3952830, upper bound: 1.3921417
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.42
Output dim: 8, lower bound: -1.3907951, upper bound: 1.3966264
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.42
Output dim: 8, lower bound: -1.3966258, upper bound: 1.3930741
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.42
Output dim: 8, lower bound: -1.3966267, upper bound: 1.3907974
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.42
Output dim: 8, lower bound: -1.3919630, upper bound: 1.3975870
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.42
Output dim: 8, lower bound: -1.3931898, upper bound: 1.3963604
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.42
Output dim: 8, lower bound: -1.3940557, upper bound: 1.3966352
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.42
Output dim: 8, lower bound: -1.3963660, upper bound: 1.3966341
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.42
Output dim: 8, lower bound: -1.3957575, upper bound: 1.3941868
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.42
Output dim: 8, lower bound: -1.3953015, upper bound: 1.3946424
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.42
Output dim: 8, lower bound: -1.3958772, upper bound: 1.3940422
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.42
Output dim: 8, lower bound: -1.3954243, upper bound: 1.3944997
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.42
Output dim: 8, lower bound: -1.3977686, upper bound: 1.3950525
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.42
Output dim: 8, lower bound: -1.3989866, upper bound: 1.3938054

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.6993322, 3.7017264
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -3.2063332, 3.2203088
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -3.1072865, 3.1074548
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.8666837, 3.8666837
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.7562952, 3.7425823
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.9321241, 2.9468193
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.3502321, 3.3566313
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.3213153, 2.3213153
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.2580624, 3.2461271

Time for backsubstitution: 14.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 4598
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 891

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5805

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3939589, upper bound: 1.3968110
time: 5.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3917514, upper bound: 1.3989801
time: 19.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.7036171, 3.6957469
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -3.2085381, 3.2172384
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -3.1044712, 3.1094728
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.8666837, 3.8666837
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.7533655, 3.7446723
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.9343710, 2.9436755
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.3504782, 3.3562880
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.3213153, 2.3213153
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.2522449, 3.2502770

Time for backsubstitution: 14.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4598
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 891

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4598

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3952744, upper bound: 1.3932250
time: 6.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3907862, upper bound: 1.3977301
time: 5.22 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.6812096, 3.6760454
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -3.2274494, 3.2394090
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -3.0554705, 3.0701551
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.8666837, 3.8666837
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.6586790, 3.6306009
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.9295769, 2.9178271
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.3479996, 3.3497276
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.3213153, 2.3213153
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.2039762, 3.1880732

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 540

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 891

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3934799, upper bound: 1.3898720
time: 7.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3930209, upper bound: 1.3903298
time: 7.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.6732845, 3.6839709
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -3.2432833, 3.2233791
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -3.0729370, 3.0526886
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.8666837, 3.8666837
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.6226683, 3.6666117
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.9239550, 2.9234495
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.3522816, 3.3454456
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.3213153, 2.3213153
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.1844554, 3.2075939

Time for backsubstitution: 14.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 540

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 891

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3889917, upper bound: 1.3943537
time: 5.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3885329, upper bound: 1.3948120
time: 5.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.6843853, 3.6728740
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -3.2199621, 3.2432833
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -3.0525885, 3.0730391
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.8666837, 3.8666837
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.6680365, 3.6212430
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.9207354, 2.9266686
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.3436947, 3.3540368
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.3213153, 2.3213153
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.2088447, 3.1832056

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 540

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3953943, upper bound: 1.3930764
time: 5.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3966255, upper bound: 1.3918555
time: 7.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.6812096, 3.6760454
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -3.2274504, 3.2394085
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -3.0554705, 3.0701551
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.8666837, 3.8666837
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.6586790, 3.6306009
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.9295750, 2.9178290
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.3479977, 3.3497295
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.3213153, 2.3213153
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.2039762, 3.1880732

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 540

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3945452, upper bound: 1.3900793
time: 4.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3966237, upper bound: 1.3900556
time: 5.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.6843138, 3.6917930
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -3.2432833, 3.2432833
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -3.0648899, 3.0473862
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.8666837, 3.8666837
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.6418381, 3.6769605
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.9454398, 2.9537826
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.3620105, 3.3596525
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.3213153, 2.3213153
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.2025442, 3.2203760

Time for backsubstitution: 14.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 540

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5805

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3919594, upper bound: 1.3953983
time: 7.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3897624, upper bound: 1.3975846
time: 6.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.6842775, 3.6918287
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -3.2432833, 3.2432833
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -3.0647526, 3.0475221
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.8666837, 3.8666837
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.6423740, 3.6764235
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.9454436, 2.9537783
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.3621821, 3.3594818
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.3213153, 2.3213153
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.2021036, 3.2208161

Time for backsubstitution: 14.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 6182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 891

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3913642, upper bound: 1.3940723
time: 5.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3909093, upper bound: 1.3945271
time: 7.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.7033467, 3.7005501
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -3.2117100, 3.2156744
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -3.1066675, 3.1093135
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.8666837, 3.8666837
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.7556372, 3.7482405
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.9370975, 2.9432259
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.3514724, 3.3541932
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.3213153, 2.3213153
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.2565384, 3.2524805

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4598
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 540

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 891

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3922510, upper bound: 1.3943616
time: 6.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3917904, upper bound: 1.3948190
time: 5.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.7001748, 3.7037249
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -3.2191982, 3.2081866
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -3.1095514, 3.1064320
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.8666837, 3.8666837
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.7462797, 3.7575984
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.9459372, 2.9343863
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.3557801, 3.3498902
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.3213153, 2.3213153
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.2516718, 3.2573490

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4598
type: RSZ, layer: 1, pos: 6182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 540

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3956791, upper bound: 1.3966310
time: 4.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3957019, upper bound: 1.3945534
time: 4.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.7099700, 3.7134957
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -3.2334146, 3.2303452
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -3.0999165, 3.1039577
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.8666837, 3.8666837
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.7651892, 3.7648478
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.9682770, 2.9642401
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.3677158, 3.3638391
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.3213153, 2.3213153
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.2679963, 3.2695780

Time for backsubstitution: 14.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 4598
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 540

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5749

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3934883, upper bound: 1.3931337
time: 7.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3957546, upper bound: 1.3931338
time: 4.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.7110257, 3.7104158
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -3.2334976, 3.2301049
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -3.1012859, 3.0999522
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.8666837, 3.8666837
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.7639599, 3.7652693
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.9669609, 2.9646940
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.3657627, 3.3645105
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.3213153, 2.3213153
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.2678876, 3.2696147

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 4598

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5749

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3930293, upper bound: 1.3935912
time: 5.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3952985, upper bound: 1.3935923
time: 5.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.7055779, 3.7150502
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -3.2315631, 3.2314558
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -3.1007729, 3.1018615
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.8666837, 3.8666837
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.7638855, 3.7611499
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.9660120, 2.9651275
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.3680553, 3.3646970
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.3213153, 2.3213153
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.2687097, 3.2640333

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 4598
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 5749

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5805

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3958700, upper bound: 1.3935520
time: 8.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3949335, upper bound: 1.3935563
time: 6.76 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 29.51 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.51
Output dim: 8, lower bound: -1.3939589, upper bound: 1.3968110
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.51
Output dim: 8, lower bound: -1.3917514, upper bound: 1.3989801
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.51
Output dim: 8, lower bound: -1.3952744, upper bound: 1.3932250
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.51
Output dim: 8, lower bound: -1.3907862, upper bound: 1.3977301
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.51
Output dim: 8, lower bound: -1.3934799, upper bound: 1.3898720
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.51
Output dim: 8, lower bound: -1.3930209, upper bound: 1.3903298
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.51
Output dim: 8, lower bound: -1.3889917, upper bound: 1.3943537
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.51
Output dim: 8, lower bound: -1.3885329, upper bound: 1.3948120
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.51
Output dim: 8, lower bound: -1.3953943, upper bound: 1.3930764
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.51
Output dim: 8, lower bound: -1.3966255, upper bound: 1.3918555
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.51
Output dim: 8, lower bound: -1.3945452, upper bound: 1.3900793
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.51
Output dim: 8, lower bound: -1.3966237, upper bound: 1.3900556
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.51
Output dim: 8, lower bound: -1.3919594, upper bound: 1.3953983
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.51
Output dim: 8, lower bound: -1.3897624, upper bound: 1.3975846
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.51
Output dim: 8, lower bound: -1.3913642, upper bound: 1.3940723
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.51
Output dim: 8, lower bound: -1.3909093, upper bound: 1.3945271
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.51
Output dim: 8, lower bound: -1.3922510, upper bound: 1.3943616
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.51
Output dim: 8, lower bound: -1.3917904, upper bound: 1.3948190
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.51
Output dim: 8, lower bound: -1.3956791, upper bound: 1.3966310
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.51
Output dim: 8, lower bound: -1.3957019, upper bound: 1.3945534
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.51
Output dim: 8, lower bound: -1.3934883, upper bound: 1.3931337
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.51
Output dim: 8, lower bound: -1.3957546, upper bound: 1.3931338
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.51
Output dim: 8, lower bound: -1.3930293, upper bound: 1.3935912
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.51
Output dim: 8, lower bound: -1.3952985, upper bound: 1.3935923
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.51
Output dim: 8, lower bound: -1.3958700, upper bound: 1.3935520
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.51
Output dim: 8, lower bound: -1.3949335, upper bound: 1.3935563
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.51
Output dim: 8, lower bound: -1.3954243, upper bound: 1.3944997
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.51
Output dim: 8, lower bound: -1.3977686, upper bound: 1.3950525
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.51
Output dim: 8, lower bound: -1.3989866, upper bound: 1.3938054
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=2.321315288543701
rel_dist={8: [-1.3998489967280836, 1.3998484636799233]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 4598
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 835

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 540

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1354672, upper bound: 1.1366280
time: 4.91 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1366288, upper bound: 1.1354664
time: 4.79 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 9.72 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 9.72
Output dim: 8, lower bound: -1.1354672, upper bound: 1.1366280
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 9.72
Output dim: 8, lower bound: -1.1366288, upper bound: 1.1354664

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.3360138, 3.3318396
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.9347906, 2.9342246
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8770609, 2.8765645
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.6386976, 3.6401868
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.4385700, 3.4390135
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.7350731, 2.7321682
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.0440707, 3.0441332
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2917786, 2.2912178
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.0147619, 3.0139942

Time for backsubstitution: 14.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 4598
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 135

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1354626, upper bound: 1.1366276
time: 4.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1354672, upper bound: 1.1354717
time: 5.23 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.3318386, 3.3360143
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.9342241, 2.9347911
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8765650, 2.8770604
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.6401873, 3.6386981
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.4390135, 3.4385700
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.7321682, 2.7350731
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.0441337, 3.0440702
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2912178, 2.2917790
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.0139933, 3.0147619

Time for backsubstitution: 14.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 4598
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 6182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1359902, upper bound: 1.1354659
time: 5.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1348338, upper bound: 1.1348312
time: 5.44 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 25.10 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 25.10
Output dim: 8, lower bound: -1.1354626, upper bound: 1.1366276
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 25.10
Output dim: 8, lower bound: -1.1354672, upper bound: 1.1354717
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 25.10
Output dim: 8, lower bound: -1.1359902, upper bound: 1.1354659
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 25.10
Output dim: 8, lower bound: -1.1348338, upper bound: 1.1348312

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.3360138, 3.3318396
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.9347882, 2.9342217
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8770609, 2.8765640
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.6386890, 3.6401796
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.4385691, 3.4390125
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.7350693, 2.7321625
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.0440617, 3.0441236
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2917771, 2.2912169
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.0147610, 3.0139937

Time for backsubstitution: 14.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 4598
type: RSZ, layer: 1, pos: 6182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5749

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1337070, upper bound: 1.1366253
time: 5.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1354602, upper bound: 1.1348708
time: 5.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.3360138, 3.3318396
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.9347873, 2.9342213
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8770609, 2.8765640
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.6386909, 3.6401782
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.4385691, 3.4390125
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.7350674, 2.7321634
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.0440607, 3.0441227
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2917771, 2.2912164
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.0147610, 3.0139937

Time for backsubstitution: 14.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 4598
type: RSZ, layer: 1, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 835

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1338823, upper bound: 1.1349249
time: 4.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1349193, upper bound: 1.1338869
time: 4.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.3319669, 3.3361220
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.9339323, 2.9344378
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8765860, 2.8770041
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.6415176, 3.6398315
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.4406185, 3.4404821
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.7321830, 2.7350912
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.0434494, 3.0434833
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2909341, 2.2915421
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.0153856, 3.0159016

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 4598
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6182

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1340459, upper bound: 1.1354649
time: 5.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1359887, upper bound: 1.1346888
time: 5.17 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.3319459, 3.3361421
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.9338713, 2.9344988
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8765078, 2.8770819
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.6413193, 3.6400299
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.4409256, 3.4401751
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.7321868, 2.7350883
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.0435467, 3.0433855
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2909808, 2.2914956
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.0151339, 3.0161533

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 4598
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 835

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6182

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1340459, upper bound: 1.1348292
time: 7.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1348322, upper bound: 1.1340430
time: 5.98 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 27.97 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.97
Output dim: 8, lower bound: -1.1337070, upper bound: 1.1366253
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.97
Output dim: 8, lower bound: -1.1354602, upper bound: 1.1348708
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.97
Output dim: 8, lower bound: -1.1338823, upper bound: 1.1349249
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.97
Output dim: 8, lower bound: -1.1349193, upper bound: 1.1338869
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.97
Output dim: 8, lower bound: -1.1340459, upper bound: 1.1354649
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.97
Output dim: 8, lower bound: -1.1359887, upper bound: 1.1346888
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.97
Output dim: 8, lower bound: -1.1340459, upper bound: 1.1348292
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.97
Output dim: 8, lower bound: -1.1348322, upper bound: 1.1340430

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.3269396, 3.3209515
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.9093246, 2.9130373
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8823061, 2.8834567
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.6255360, 3.6216245
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.4248714, 3.4199677
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.7047572, 2.7069016
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.0292706, 3.0317912
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2877784, 2.2886360
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.0008502, 2.9973011

Time for backsubstitution: 14.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 4598
type: RSZ, layer: 1, pos: 891

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1330699, upper bound: 1.1366250
time: 5.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1337065, upper bound: 1.1359873
time: 6.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.3251257, 3.3227634
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.9136038, 2.9087586
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8839531, 2.8818097
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.6201344, 3.6270261
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.4195242, 3.4253149
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.7098079, 2.7018509
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.0317292, 3.0293326
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2891965, 2.2872176
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.9980683, 3.0000820

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4598
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 835

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4598

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1354509, upper bound: 1.1308867
time: 8.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1315372, upper bound: 1.1348617
time: 4.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.3358154, 3.3314066
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.9312139, 2.9325995
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8770590, 2.8766208
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.6374149, 3.6373701
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.4378996, 3.4375286
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.7322311, 2.7308779
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.0422249, 3.0432873
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2917762, 2.2914805
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.0141673, 3.0126858

Time for backsubstitution: 14.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4598

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5805

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1338805, upper bound: 1.1337279
time: 5.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1326883, upper bound: 1.1349225
time: 8.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.3355818, 3.3316412
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.9331660, 2.9306474
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8771162, 2.8765635
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.6358833, 3.6389012
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.4370852, 3.4383426
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.7337818, 2.7293267
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.0432253, 3.0422864
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2920413, 2.2912154
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.0134530, 3.0133996

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4598
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 6182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4598

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1349100, upper bound: 1.1299591
time: 7.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1309926, upper bound: 1.1338788
time: 5.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.3276811, 3.3352532
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.9317274, 2.9339871
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8761778, 2.8749862
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.6402264, 3.6395750
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.4402037, 3.4383926
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.7299380, 2.7346420
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.0432024, 3.0434337
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2908554, 2.2911558
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.0145607, 3.0117517

Time for backsubstitution: 14.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 4598
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 5749

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5805

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1340441, upper bound: 1.1342761
time: 6.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1328415, upper bound: 1.1354642
time: 5.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.3310990, 3.3318367
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.9334812, 2.9322324
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8745680, 2.8765955
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.6412611, 3.6385403
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.4385290, 3.4400673
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.7317348, 2.7328458
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.0433989, 3.0432377
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2905483, 2.2914631
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.0112362, 3.0150757

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 4598

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 835

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1344034, upper bound: 1.1341240
time: 5.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1354444, upper bound: 1.1330922
time: 5.48 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.3276620, 3.3352737
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.9316664, 2.9340482
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8760996, 2.8750639
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.6400280, 3.6397729
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.4405107, 3.4380860
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.7299399, 2.7346396
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.0433006, 3.0433359
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2909021, 2.2911093
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.0143089, 3.0120039

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 4598
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 835

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5805

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1358445, upper bound: 1.1336386
time: 7.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1346404, upper bound: 1.1348277
time: 9.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.3310781, 3.3318572
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.9334202, 2.9322934
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8744898, 2.8766732
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.6410637, 3.6387386
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.4388361, 3.4397607
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.7317367, 2.7328434
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.0434961, 3.0431399
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2905946, 2.2914166
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.0109844, 3.0153279

Time for backsubstitution: 14.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4598
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 5805

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4598

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1300876, upper bound: 1.1300847
time: 6.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1327038, upper bound: 1.1340340
time: 4.91 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 25.45 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.45
Output dim: 8, lower bound: -1.1330699, upper bound: 1.1366250
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.45
Output dim: 8, lower bound: -1.1337065, upper bound: 1.1359873
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.45
Output dim: 8, lower bound: -1.1354509, upper bound: 1.1308867
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.45
Output dim: 8, lower bound: -1.1315372, upper bound: 1.1348617
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.45
Output dim: 8, lower bound: -1.1338805, upper bound: 1.1337279
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.45
Output dim: 8, lower bound: -1.1326883, upper bound: 1.1349225
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.45
Output dim: 8, lower bound: -1.1349100, upper bound: 1.1299591
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.45
Output dim: 8, lower bound: -1.1309926, upper bound: 1.1338788
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.45
Output dim: 8, lower bound: -1.1340441, upper bound: 1.1342761
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.45
Output dim: 8, lower bound: -1.1328415, upper bound: 1.1354642
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.45
Output dim: 8, lower bound: -1.1344034, upper bound: 1.1341240
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.45
Output dim: 8, lower bound: -1.1354444, upper bound: 1.1330922
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.45
Output dim: 8, lower bound: -1.1358445, upper bound: 1.1336386
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.45
Output dim: 8, lower bound: -1.1346404, upper bound: 1.1348277
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.45
Output dim: 8, lower bound: -1.1300876, upper bound: 1.1300847
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.45
Output dim: 8, lower bound: -1.1327038, upper bound: 1.1340340

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.3270664, 3.3210583
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.9090343, 2.9126859
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8823261, 2.8833995
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.6268692, 3.6227593
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.4264774, 3.4218798
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.7047696, 2.7069173
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.0285869, 3.0312057
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2874951, 2.2883992
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.0022416, 2.9984412

Time for backsubstitution: 14.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 4598
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 891

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 835

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1324486, upper bound: 1.1360804
time: 5.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1324492, upper bound: 1.1342497
time: 5.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.3270454, 3.3210788
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.9089732, 2.9127469
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8822498, 2.8834777
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.6266718, 3.6229577
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.4267845, 3.4215732
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.7047734, 2.7069149
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.0286851, 3.0311079
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2875414, 2.2883527
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.0019898, 2.9986925

Time for backsubstitution: 14.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 4598
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1329598, upper bound: 1.1359884
time: 5.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1323209, upper bound: 1.1351946
time: 8.19 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.3024960, 3.2956052
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.9250259, 2.9293408
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8300529, 2.8378916
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.5968027, 3.6075730
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.3166714, 3.3018842
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.6937656, 2.6825943
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.0249462, 3.0249963
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2901115, 2.2832756
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.9421511, 2.9330101

Time for backsubstitution: 14.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 5805

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1309035, upper bound: 1.1308870
time: 8.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1354505, upper bound: 1.1302462
time: 6.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.2979679, 3.3001337
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.9341860, 2.9201808
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8400340, 2.8279104
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.6006813, 3.6036940
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.2960930, 3.3224616
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.6905518, 2.6858072
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.0273933, 3.0225496
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2852545, 2.2881327
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.9309959, 2.9441648

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1309011, upper bound: 1.1348611
time: 5.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1290876, upper bound: 1.1342238
time: 6.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.3238611, 3.3254972
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.9294877, 2.9317451
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8628578, 2.8696108
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.6351595, 3.6328111
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.4365082, 3.4347057
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.7291574, 2.7246590
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.0393038, 3.0373697
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2878671, 2.2895520
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.0128660, 3.0100489

Time for backsubstitution: 14.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 4598
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5749

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1330861, upper bound: 1.1337256
time: 8.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1330873, upper bound: 1.1319032
time: 7.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.3299074, 3.3194523
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.9303594, 2.9308739
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8700495, 2.8624191
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.6328554, 3.6351147
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.4350767, 3.4361372
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.7260122, 2.7278037
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.0363073, 3.0403662
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2898479, 2.2875712
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.0115299, 3.0113850

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4598

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5749

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1318942, upper bound: 1.1349190
time: 5.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1318976, upper bound: 1.1330914
time: 10.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.3129511, 3.3044820
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.9445887, 2.9512300
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8232174, 2.8326454
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.6125517, 3.6194482
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.3342342, 3.3149137
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.7177396, 2.7100730
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.0364447, 3.0379524
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2929564, 2.2872739
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.9575405, 2.9463320

Time for backsubstitution: 14.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5749

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1330792, upper bound: 1.1291707
time: 6.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1349077, upper bound: 1.1291695
time: 5.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.3084221, 3.3090105
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.9537487, 2.9420700
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8331985, 2.8226643
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.6164303, 3.6155691
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.3136559, 3.3354912
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.7145276, 2.7132859
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.0388918, 3.0355058
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2880993, 2.2921309
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.9463863, 2.9574871

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 5805

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5749

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1291647, upper bound: 1.1330847
time: 5.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1291608, upper bound: 1.1330863
time: 6.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.3157277, 3.3293467
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.9300013, 2.9331336
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8619757, 2.8679767
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.6379719, 3.6350164
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.4388123, 3.4355693
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.7268639, 2.7284231
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.0402808, 3.0375152
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2869463, 2.2892277
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.0132589, 3.0091143

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4598

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5749

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1334714, upper bound: 1.1342710
time: 6.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1328391, upper bound: 1.1325178
time: 7.25 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.3217731, 3.3232999
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.9308729, 2.9322615
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8691645, 2.8607855
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.6356688, 3.6373196
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.4373817, 3.4370012
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.7237186, 2.7315674
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.0372844, 3.0405116
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2889271, 2.2872467
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.0119228, 3.0104508

Time for backsubstitution: 14.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 4598

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 891

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1323197, upper bound: 1.1337576
time: 7.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1334714, upper bound: 1.1346551
time: 8.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.3308997, 3.3314037
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.9299078, 2.9306111
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8745680, 2.8766527
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.6399860, 3.6357346
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.4378586, 3.4385824
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.7288976, 2.7315602
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.0415611, 3.0424013
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2905464, 2.2917264
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.0106425, 3.0137691

Time for backsubstitution: 14.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 4598
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 891

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5749

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1316653, upper bound: 1.1341215
time: 7.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1324529, upper bound: 1.1323082
time: 6.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.3306651, 3.3316379
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.9318590, 2.9286585
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8746252, 2.8765955
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.6384554, 3.6372652
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.4370441, 3.4393969
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.7304492, 2.7300096
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.0425625, 3.0414004
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2908115, 2.2914615
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.0099292, 3.0144830

Time for backsubstitution: 14.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 4598
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5749

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1336100, upper bound: 1.1323088
time: 5.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1354421, upper bound: 1.1323079
time: 5.23 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.3157077, 3.3293676
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.9299402, 2.9331946
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8618975, 2.8680544
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.6377735, 3.6352148
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.4391193, 3.4352627
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.7268658, 2.7284203
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.0403790, 3.0374169
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2869930, 2.2891812
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.0130072, 3.0093660

Time for backsubstitution: 14.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 4598
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 891

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5749

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1341209, upper bound: 1.1336358
time: 5.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1358421, upper bound: 1.1318803
time: 5.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.3217521, 3.3233204
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.9308119, 2.9323220
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8690882, 2.8608632
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.6354704, 3.6375184
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.4376869, 3.4366946
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.7237206, 2.7315650
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.0373816, 3.0404139
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2889733, 2.2872005
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.0116711, 3.0107026

Time for backsubstitution: 14.60 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=2.2884626388549805
rel_dist={8: [-1.136630455303056, 1.136628973869513]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 4598
type: RSZ, layer: 1, pos: 5749

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 133

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0401012, upper bound: 1.0405789
time: 5.91 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0401012, upper bound: 1.0400980
time: 6.23 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 12.15 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 12.15
Output dim: 8, lower bound: -1.0401012, upper bound: 1.0405789
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 12.15
Output dim: 8, lower bound: -1.0401012, upper bound: 1.0400980

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.2331233, 3.2331080
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.8302746, 2.8302288
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8056402, 2.8055820
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.5615101, 3.5613608
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.3353519, 3.3355823
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.6769686, 2.6769705
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -2.9353199, 2.9353929
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2252045, 2.2252398
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.9367571, 2.9365683

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4598
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 6182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4598

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0400918, upper bound: 1.0375858
time: 6.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 135

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0395005, upper bound: 1.0405789
time: 5.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0400986, upper bound: 1.0399796
time: 5.36 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.2331080, 3.2331233
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.8302288, 2.8302746
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8055820, 2.8056407
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.5613613, 3.5615096
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.3355827, 3.3353519
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.6769705, 2.6769686
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -2.9353924, 2.9353194
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2252393, 2.2252047
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.9365683, 2.9367571

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 4598

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 835

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0392583, upper bound: 1.0395876
time: 5.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0400697, upper bound: 1.0387759
time: 5.04 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 25.03 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 25.03
Output dim: 8, lower bound: -1.0395005, upper bound: 1.0405789
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 25.03
Output dim: 8, lower bound: -1.0400986, upper bound: 1.0399796
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 25.03
Output dim: 8, lower bound: -1.0392583, upper bound: 1.0395876
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 25.03
Output dim: 8, lower bound: -1.0400697, upper bound: 1.0387759

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.2331223, 3.2331071
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.8302717, 2.8302264
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8056412, 2.8055825
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.5615005, 3.5613523
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.3353519, 3.3355808
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.6769629, 2.6769638
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -2.9353104, 2.9353833
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2252026, 2.2252376
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.9367552, 2.9365668

Time for backsubstitution: 14.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 4598
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 835

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 891

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0388958, upper bound: 1.0395886
time: 4.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0384911, upper bound: 1.0400308
time: 5.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.2331223, 3.2331071
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.8302717, 2.8302259
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8056412, 2.8055825
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.5615015, 3.5613513
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.3353519, 3.3355813
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.6769629, 2.6769648
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -2.9353104, 2.9353838
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2252026, 2.2252374
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.9367552, 2.9365668

Time for backsubstitution: 14.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 4598
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 5749

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 891

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0395386, upper bound: 1.0389792
time: 4.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0391081, upper bound: 1.0393863
time: 4.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.2328496, 3.2326894
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.8266544, 2.8281646
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8055820, 2.8056836
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.5597000, 3.5587010
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.3347092, 3.3338680
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.6741343, 2.6752949
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -2.9335546, 2.9342322
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2252383, 2.2254021
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.9357953, 2.9354486

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 4598
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 6182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 540

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0383889, upper bound: 1.0395858
time: 5.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0392569, upper bound: 1.0387182
time: 5.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.2326741, 3.2328653
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.8281183, 2.8267002
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8056259, 2.8056407
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.5585527, 3.5598493
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.3340979, 3.3344789
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.6752958, 2.6741319
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -2.9343052, 2.9334817
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2254367, 2.2252033
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.9352603, 2.9359846

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 4598

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6182

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0393410, upper bound: 1.0380388
time: 25.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0400681, upper bound: 1.0380383
time: 5.27 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 44.99 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 44.99
Output dim: 8, lower bound: -1.0388958, upper bound: 1.0395886
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 44.99
Output dim: 8, lower bound: -1.0384911, upper bound: 1.0400308
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 44.99
Output dim: 8, lower bound: -1.0395386, upper bound: 1.0389792
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 44.99
Output dim: 8, lower bound: -1.0391081, upper bound: 1.0393863
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 44.99
Output dim: 8, lower bound: -1.0383889, upper bound: 1.0395858
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 44.99
Output dim: 8, lower bound: -1.0392569, upper bound: 1.0387182
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 44.99
Output dim: 8, lower bound: -1.0393410, upper bound: 1.0380388
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 44.99
Output dim: 8, lower bound: -1.0400681, upper bound: 1.0380383

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.2320657, 3.2333708
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.8301888, 2.8302464
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8042717, 2.8059297
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.5615683, 3.5610738
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.3354578, 3.3351603
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.6770735, 2.6765103
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -2.9354763, 2.9347115
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2248855, 2.2253180
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.9367647, 2.9365292

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 4598
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5805

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0388889, upper bound: 1.0391516
time: 5.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0380744, upper bound: 1.0391585
time: 5.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.2331223, 3.2320509
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.8302717, 2.8301435
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8056412, 2.8042130
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.5612221, 3.5613523
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.3349314, 3.3355808
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.6765099, 2.6769638
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -2.9346390, 2.9353833
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2252026, 2.2249207
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.9367180, 2.9365668

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4598
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4598

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0384816, upper bound: 1.0370474
time: 10.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0354948, upper bound: 1.0400248
time: 8.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.2320657, 3.2333708
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.8301897, 2.8302460
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8042717, 2.8059297
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.5615692, 3.5610728
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.3354578, 3.3351607
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.6770725, 2.6765113
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -2.9354753, 2.9347124
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2248859, 2.2253180
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.9367647, 2.9365292

Time for backsubstitution: 14.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4598
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 5749

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4598

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0395316, upper bound: 1.0359851
time: 8.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0365531, upper bound: 1.0389720
time: 4.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.2331223, 3.2320509
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.8302717, 2.8301430
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8056412, 2.8042130
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.5612221, 3.5613513
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.3349314, 3.3355813
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.6765079, 2.6769648
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -2.9346390, 2.9353838
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2252026, 2.2249207
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.9367180, 2.9365668

Time for backsubstitution: 14.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 4598
type: RSZ, layer: 1, pos: 5805

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 540

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0382390, upper bound: 1.0391936
time: 4.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0391066, upper bound: 1.0391864
time: 4.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.2152324, 3.2119412
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.8298655, 2.8309503
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8029785, 2.8027081
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.5507631, 3.5508795
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.3320389, 3.3315310
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.6588864, 2.6578684
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -2.9338608, 2.9345856
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2284155, 2.2281587
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.9317708, 2.9308481

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4598
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 891

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4598

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0383819, upper bound: 1.0365935
time: 5.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0353984, upper bound: 1.0395826
time: 7.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.2121024, 3.2150722
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.8294401, 2.8313751
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8026066, 2.8030806
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.5518789, 3.5497632
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.3323727, 3.3311987
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.6567073, 2.6600475
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -2.9339085, 2.9345388
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2279944, 2.2285795
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.9311948, 2.9314241

Time for backsubstitution: 14.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 4598
type: RSZ, layer: 1, pos: 5805

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 891

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0387094, upper bound: 1.0377271
time: 6.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0382666, upper bound: 1.0381982
time: 5.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.2283893, 3.2311435
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.8259134, 2.8258100
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8048162, 2.8036246
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.5572615, 3.5593338
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.3332634, 3.3323884
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.6730518, 2.6732340
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -2.9340601, 2.9333830
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2252812, 2.2248168
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.9336019, 2.9318337

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 4598
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 5749

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5805

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0393397, upper bound: 1.0378695
time: 5.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0384316, upper bound: 1.0387735
time: 6.45 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.2309518, 3.2285805
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.8272295, 2.8244948
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8036089, 2.8048315
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.5580368, 3.5585580
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.3320084, 3.3336439
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.6743994, 2.6718864
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -2.9342070, 2.9332361
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2250504, 2.2250473
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.9311090, 2.9343266

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 4598
type: RSZ, layer: 1, pos: 891

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 540

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0391987, upper bound: 1.0380366
time: 5.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0400667, upper bound: 1.0371735
time: 5.64 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 25.88 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.88
Output dim: 8, lower bound: -1.0388889, upper bound: 1.0391516
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.88
Output dim: 8, lower bound: -1.0380744, upper bound: 1.0391585
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.88
Output dim: 8, lower bound: -1.0384816, upper bound: 1.0370474
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.88
Output dim: 8, lower bound: -1.0354948, upper bound: 1.0400248
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.88
Output dim: 8, lower bound: -1.0395316, upper bound: 1.0359851
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.88
Output dim: 8, lower bound: -1.0365531, upper bound: 1.0389720
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.88
Output dim: 8, lower bound: -1.0382390, upper bound: 1.0391936
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.88
Output dim: 8, lower bound: -1.0391066, upper bound: 1.0391864
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.88
Output dim: 8, lower bound: -1.0383819, upper bound: 1.0365935
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.88
Output dim: 8, lower bound: -1.0353984, upper bound: 1.0395826
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.88
Output dim: 8, lower bound: -1.0387094, upper bound: 1.0377271
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.88
Output dim: 8, lower bound: -1.0382666, upper bound: 1.0381982
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.88
Output dim: 8, lower bound: -1.0393397, upper bound: 1.0378695
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.88
Output dim: 8, lower bound: -1.0384316, upper bound: 1.0387735
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.88
Output dim: 8, lower bound: -1.0391987, upper bound: 1.0380366
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.88
Output dim: 8, lower bound: -1.0400667, upper bound: 1.0371735

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.2201123, 3.2259526
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.8284626, 2.8291755
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.7900715, 2.7971234
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.5587406, 3.5565171
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.3337097, 3.3323393
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.6732168, 2.6702914
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -2.9318109, 2.9287968
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2209773, 2.2228951
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.9351301, 2.9338923

Time for backsubstitution: 14.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 4598

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 835

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0375680, upper bound: 1.0386407
time: 6.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0383800, upper bound: 1.0378304
time: 4.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.2246461, 3.2214165
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.8291159, 2.8285203
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.7954645, 2.7917295
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.5570107, 3.5582442
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.3326359, 3.3334131
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.6708546, 2.6726503
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -2.9295611, 2.9310446
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2224627, 2.2214096
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.9341278, 2.9348946

Time for backsubstitution: 14.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 4598
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 540

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0378000, upper bound: 1.0391533
time: 5.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0378089, upper bound: 1.0382854
time: 5.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.2093596, 3.2048912
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.8416963, 2.8484373
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.7517414, 2.7578001
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.5378895, 3.5409279
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.2269344, 3.2121515
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.6596651, 2.6577091
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -2.9278574, 2.9304361
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2249050, 2.2209802
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.8780136, 2.8694973

Time for backsubstitution: 14.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 835

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 540

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0382295, upper bound: 1.0370431
time: 8.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0382374, upper bound: 1.0361968
time: 14.46 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.2059627, 3.2082877
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.8485656, 2.8415675
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.7592278, 2.7503147
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.5407982, 3.5380187
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.2115021, 3.2275844
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.6572552, 2.6601186
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -2.9296932, 2.9286013
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2212620, 2.2246232
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.8696480, 2.8778629

Time for backsubstitution: 14.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 6182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 835

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0341779, upper bound: 1.0395157
time: 8.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0349897, upper bound: 1.0387037
time: 9.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.2083030, 3.2062111
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.8416133, 2.8485403
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.7503729, 2.7595167
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.5382366, 3.5406504
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.2274609, 3.2117319
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.6602278, 2.6572571
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -2.9286938, 2.9297662
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2245884, 2.2213774
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.8780603, 2.8694592

Time for backsubstitution: 14.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5805

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0395249, upper bound: 1.0355474
time: 5.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0386642, upper bound: 1.0355492
time: 5.26 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.2049060, 3.2096076
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.8484836, 2.8416700
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.7578592, 2.7520308
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.5411472, 3.5377412
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.2120285, 3.2271647
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.6578178, 2.6596665
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -2.9305286, 2.9279308
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2209454, 2.2250202
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.8696947, 2.8778253

Time for backsubstitution: 14.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 5805

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5749

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0351879, upper bound: 1.0389709
time: 8.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0365510, upper bound: 1.0376411
time: 4.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.2155056, 3.2113013
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.8334818, 2.8329282
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8030367, 2.8012376
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.5522833, 3.5535278
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.3322630, 3.3332443
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.6612601, 2.6595364
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -2.9349470, 2.9357371
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2283807, 2.2276788
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.9326959, 2.9319692

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4598
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 5749

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4598

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0382321, upper bound: 1.0362006
time: 9.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0352421, upper bound: 1.0391876
time: 10.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.2123737, 3.2144327
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.8330584, 2.8333526
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8026648, 2.8016095
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.5533991, 3.5524120
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.3325949, 3.3329124
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.6590810, 2.6617150
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -2.9349937, 2.9356918
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2279601, 2.2280993
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.9321198, 2.9325447

Time for backsubstitution: 14.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 4598

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0383455, upper bound: 1.0391885
time: 5.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0382374, upper bound: 1.0384223
time: 5.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.1914706, 3.1847820
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.8412895, 2.8492446
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.7490778, 2.7562933
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.5274305, 3.5304565
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.2240438, 3.2081017
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.6420383, 2.6386118
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -2.9270797, 2.9296389
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2281175, 2.2242177
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.8730698, 2.8637819

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 5749

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 891

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0378627, upper bound: 1.0356050
time: 8.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0373916, upper bound: 1.0360434
time: 7.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.1880736, 3.1881785
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.8481598, 2.8423748
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.7565641, 2.7488074
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.5303402, 3.5275478
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.2086096, 3.2235346
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.6396294, 2.6410217
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -2.9289145, 2.9278040
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2244744, 2.2278605
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.8647041, 2.8721480

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6182

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0346371, upper bound: 1.0395788
time: 11.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0353968, upper bound: 1.0388494
time: 8.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.2110448, 3.2153358
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.8293571, 2.8313951
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8012371, 2.8034272
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.5519447, 3.5494838
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.3324776, 3.3307786
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.6568174, 2.6595941
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -2.9340734, 2.9338675
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2276778, 2.2286599
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.9312057, 2.9313889

Time for backsubstitution: 14.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 4598
type: RSZ, layer: 1, pos: 6182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 135

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0378736, upper bound: 1.0377269
time: 6.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0387093, upper bound: 1.0377241
time: 5.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.1949067, -1.9940324, -6.1949067, -1.9940324, -3.2121024, 3.2140160
1: -12.2345352, -8.9912519, -12.2345352, -8.9912519, -2.8294401, 2.8312922
2: -5.6206846, -2.2354484, -5.6206846, -2.2354484, -2.8026066, 2.8017106
3: -5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.5515995, 3.5497632
4: -11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.3319511, 3.3311987
5: -6.2900958, -3.0817809, -6.2900958, -3.0817809, -2.6562538, 2.6600475
6: -12.4278812, -8.6957130, -12.4278812, -8.6957130, -2.9332371, 2.9345388
7: -8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888
8: 7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.2279944, 2.2282627
9: -6.3480244, -2.8028445, -6.3480244, -2.8028445, -2.9311600, 2.9314241

Time for backsubstitution: 14.43 seconds
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=2.225489616394043
rel_dist={8: [-1.040580051898944, 1.0405791313826693]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2426.60 seconds
