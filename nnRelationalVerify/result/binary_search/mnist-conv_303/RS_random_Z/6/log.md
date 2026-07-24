## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 0.83234478323
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.5797892, 2.5797892)
1: (-3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.4460940, 2.4460940)
2: (1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.9857671, 1.9857671)
3: (-7.2597189, -5.2326798, -7.2597189, -5.2326798, -2.0270391, 2.0270391)
4: (-2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.9272163, 1.9272163)
5: (-4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.8746789, 1.8746789)
6: (-4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.5114598, 2.5114598)
7: (-8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.8371913, 1.8371913)
8: (-4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.2200575, 2.2200575)
9: (-12.0749950, -9.7447462, -12.0749950, -9.7447462, -2.2258463, 2.2258465)

## BASE Result
execution time: IAR + LP analysis = 15.21 + 32.67 = 47.87 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3552.13 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=1.9464805126190186
rel_dist={2: [-1.1320473731522231, 1.1320474951997723]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=1.8296241760253906
rel_dist={2: [-0.835577596891917, 0.8355774930246449]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=1.7517199516296387
rel_dist={2: [-0.5785716541258044, 0.5785690000616741]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=1.7906718254089355
rel_dist={2: [-0.7104656444989477, 0.7104623396628322]}

## Binary Search Result
Binary search time: 203.04 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Relational Split (RS_random_Z) starts
Time budget: 3349.09 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 5733

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6111

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2085782, upper bound: 1.1945464
time: 5.75 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1945466, upper bound: 1.2085776
time: 4.61 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 10.38 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 10.38
Output dim: 2, lower bound: -1.2085782, upper bound: 1.1945464
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 10.38
Output dim: 2, lower bound: -1.1945466, upper bound: 1.2085776

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.3687191, 2.3698912
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.4327269, 2.4358644
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.9567757, 1.9439743
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -2.0270391, 2.0270391
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.7030026, 1.6991056
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.8746789, 1.8746789
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.3499942, 2.3503611
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.6074684, 1.6203392
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.1614475, 2.1633201
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.7554083, 1.7755215

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5733
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 525

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5733

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2085427, upper bound: 1.1889584
time: 6.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1889566, upper bound: 1.1945134
time: 8.53 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.3698912, 2.3687191
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.4358644, 2.4327269
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.9439745, 1.9567759
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -2.0270391, 2.0270391
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.6991059, 1.7030023
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.8746789, 1.8746789
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.3503613, 2.3499939
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.6203392, 1.6074684
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.1633205, 2.1614470
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.7755218, 1.7554083

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 5733
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 927

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1840473, upper bound: 1.2085711
time: 5.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1840473, upper bound: 1.1980237
time: 4.84 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 24.58 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.58
Output dim: 2, lower bound: -1.2085427, upper bound: 1.1889584
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.58
Output dim: 2, lower bound: -1.1889566, upper bound: 1.1945134
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.58
Output dim: 2, lower bound: -1.1840473, upper bound: 1.2085711
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.58
Output dim: 2, lower bound: -1.1840473, upper bound: 1.1980237

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.3745584, 2.3659077
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.4281626, 2.4425902
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.9651227, 1.9382763
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -2.0270391, 2.0270391
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.6917223, 1.7156310
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.8745756, 1.8746789
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.3418279, 2.3623405
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.6028984, 1.6270535
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.1731715, 2.1553349
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.7739663, 1.7628570

Time for backsubstitution: 14.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 904

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 548

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2082548, upper bound: 1.1889491
time: 5.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2085371, upper bound: 1.1886714
time: 6.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.3647346, 2.3698912
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.4327269, 2.4312997
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.9510784, 1.9439743
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -2.0270391, 2.0270391
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.7030026, 1.6878257
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.8746789, 1.8746789
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.3499942, 2.3421950
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.6074684, 1.6157687
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.1534619, 2.1633201
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.7427440, 1.7755215

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 143

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 927

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1924218, upper bound: 1.1945066
time: 6.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2029767, upper bound: 1.1840154
time: 6.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.3746872, 2.3754134
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.4344420, 2.4317179
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.9340034, 1.9497163
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -2.0270391, 2.0270391
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.6968644, 1.6998427
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.8746789, 1.8746789
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.3404522, 2.3360200
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.6016583, 1.5810950
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.1445594, 2.1349685
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.7746458, 1.7541738

Time for backsubstitution: 14.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 5733
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4625

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 904

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1750762, upper bound: 1.2085675
time: 6.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1750762, upper bound: 1.1996150
time: 6.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.3765850, 2.3735156
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.4348559, 2.4313040
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.9369149, 1.9468052
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -2.0270391, 2.0270391
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.6959457, 1.7007611
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.8746789, 1.8746789
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.3363867, 2.3400855
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.5939660, 1.5887874
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.1368413, 2.1426866
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.7742872, 1.7545323

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5733
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 5818

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 525

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1863701, upper bound: 1.1980244
time: 5.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1863703, upper bound: 1.1899818
time: 6.29 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 26.06 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.06
Output dim: 2, lower bound: -1.2082548, upper bound: 1.1889491
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.06
Output dim: 2, lower bound: -1.2085371, upper bound: 1.1886714
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.06
Output dim: 2, lower bound: -1.1924218, upper bound: 1.1945066
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.06
Output dim: 2, lower bound: -1.2029767, upper bound: 1.1840154
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.06
Output dim: 2, lower bound: -1.1750762, upper bound: 1.2085675
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.06
Output dim: 2, lower bound: -1.1750762, upper bound: 1.1996150
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.06
Output dim: 2, lower bound: -1.1863701, upper bound: 1.1980244
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.06
Output dim: 2, lower bound: -1.1863703, upper bound: 1.1899818

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.3781137, 2.3705082
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.4187059, 2.4292355
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.9766073, 1.9471107
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -2.0270391, 2.0270391
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.6755229, 1.6927499
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.8437681, 1.8667352
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.3373866, 2.3654113
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.5708740, 1.5818242
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.2057552, 2.1985641
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.7613521, 1.7450402

Time for backsubstitution: 14.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5818

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1992675, upper bound: 1.1889490
time: 5.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2082545, upper bound: 1.1799415
time: 4.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.3791599, 2.3694630
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.4148073, 2.4331341
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.9739571, 1.9497612
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -2.0270391, 2.0270391
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.6688415, 1.6994313
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.8527603, 1.8577430
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.3448987, 2.3578997
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.5576684, 1.5950292
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.2164001, 2.1879191
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.7561488, 1.7502432

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4625

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 143

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2077529, upper bound: 1.1881971
time: 5.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2085303, upper bound: 1.1881725
time: 5.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.3695326, 2.3765855
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.4313045, 2.4302907
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.9411077, 1.9369147
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -2.0270391, 2.0270391
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.7007611, 1.6846659
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.8746789, 1.8746789
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.3400850, 2.3282199
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.5887876, 1.5893955
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.1347008, 2.1368415
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.7418680, 1.7742867

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 5841

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 548

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1780994, upper bound: 1.1944994
time: 5.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1780994, upper bound: 1.1942223
time: 5.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.3714304, 2.3746877
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.4317179, 2.4298773
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.9440193, 1.9340036
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -2.0270391, 2.0270391
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.6998425, 1.6855843
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.8746789, 1.8746789
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.3360195, 2.3322859
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.5810952, 1.5970879
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.1269827, 2.1445596
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.7415094, 1.7746458

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 510

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5841

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1922723, upper bound: 1.1837826
time: 6.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2029761, upper bound: 1.1837781
time: 5.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.3789053, 2.3813057
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.4150953, 2.4180145
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.9399109, 1.9539452
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -2.0270391, 2.0270391
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.6975086, 1.7003059
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.8746789, 1.8719459
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.3305330, 2.3220291
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.5995469, 1.5795981
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.1445055, 2.1349301
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.7664285, 1.7483578

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 5733
type: RSZ, layer: 1, pos: 143

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 525

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1668822, upper bound: 1.2085697
time: 7.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1750758, upper bound: 1.2004547
time: 5.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.3805799, 2.3796306
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.4207387, 2.4123712
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.9382324, 1.9556236
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -2.0270391, 2.0270391
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.6973279, 1.7004867
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.8746789, 1.8732121
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.3264618, 2.3261003
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.6001616, 1.5789833
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.1445208, 2.1349144
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.7688303, 1.7459562

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 5733
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 4632

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1742223, upper bound: 1.1996150
time: 5.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1840430, upper bound: 1.1987449
time: 5.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.3902960, 2.3912597
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.3979554, 2.4052153
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.9550786, 1.9608350
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -2.0270391, 2.0270391
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.6615491, 1.6764007
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.8632572, 1.8507233
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.2949543, 2.2815814
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.5399342, 1.5505257
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.1845999, 2.1783979
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.7381797, 1.7388082

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 5733
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 510

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1855809, upper bound: 1.1980241
time: 6.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1855828, upper bound: 1.1971367
time: 6.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.3943300, 2.3872266
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.4087667, 2.3944039
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.9509444, 1.9649692
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -2.0270391, 2.0270391
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.6715851, 1.6663642
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.8560464, 1.8579338
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.2778831, 2.2986526
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.5557041, 1.5347556
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.1725531, 2.1904449
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.7585626, 1.7184253

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 5733
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 143

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1937603, upper bound: 1.1899717
time: 6.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1945269, upper bound: 1.1892281
time: 5.23 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 26.19 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.19
Output dim: 2, lower bound: -1.1992675, upper bound: 1.1889490
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.19
Output dim: 2, lower bound: -1.2082545, upper bound: 1.1799415
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.19
Output dim: 2, lower bound: -1.2077529, upper bound: 1.1881971
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.19
Output dim: 2, lower bound: -1.2085303, upper bound: 1.1881725
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.19
Output dim: 2, lower bound: -1.1780994, upper bound: 1.1944994
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.19
Output dim: 2, lower bound: -1.1780994, upper bound: 1.1942223
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.19
Output dim: 2, lower bound: -1.1922723, upper bound: 1.1837826
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.19
Output dim: 2, lower bound: -1.2029761, upper bound: 1.1837781
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.19
Output dim: 2, lower bound: -1.1668822, upper bound: 1.2085697
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.19
Output dim: 2, lower bound: -1.1750758, upper bound: 1.2004547
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.19
Output dim: 2, lower bound: -1.1742223, upper bound: 1.1996150
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.19
Output dim: 2, lower bound: -1.1840430, upper bound: 1.1987449
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.19
Output dim: 2, lower bound: -1.1855809, upper bound: 1.1980241
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.19
Output dim: 2, lower bound: -1.1855828, upper bound: 1.1971367
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.19
Output dim: 2, lower bound: -1.1937603, upper bound: 1.1899717
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.19
Output dim: 2, lower bound: -1.1945269, upper bound: 1.1892281

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.3820105, 2.3755932
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.3360624, 2.3707204
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.9857671, 1.9643736
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -2.0270391, 2.0270391
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.6519367, 1.6801226
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.8172753, 1.8293345
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.2961478, 2.3071711
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4967420, 1.5293307
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.2200575, 2.2200575
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.7565665, 1.7628381

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 4632

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 510

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1983854, upper bound: 1.1889510
time: 6.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1992675, upper bound: 1.1880694
time: 5.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.3831987, 2.3744054
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.3601909, 2.3465919
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.9857671, 1.9696407
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -2.0270391, 2.0270391
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.6628958, 1.6691635
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.8063676, 1.8402424
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.2791467, 2.3241727
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.5183804, 1.5076927
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.2200575, 2.2200575
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.7791495, 1.7402549

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4632

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2082503, upper bound: 1.1774459
time: 5.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1827433, upper bound: 1.1799393
time: 5.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.3795819, 2.3700576
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.4171681, 2.4344511
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.9705954, 1.9454663
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -2.0270391, 2.0270391
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.6694176, 1.6997545
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.8534474, 1.8589368
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.3359556, 2.3509226
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.5573874, 1.5934330
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.2011242, 2.1760480
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.7555766, 1.7494302

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 5818

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5841

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1986065, upper bound: 1.1881939
time: 5.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2077525, upper bound: 1.1790528
time: 7.15 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.3797584, 2.3698854
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.4182215, 2.4354949
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.9715934, 1.9463997
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -2.0270391, 2.0270391
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.6696551, 1.7000077
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.8539538, 1.8594513
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.3379216, 2.3529730
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.5560727, 1.5947477
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.2045293, 2.1794713
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.7553363, 1.7496703

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5841

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1993864, upper bound: 1.1881693
time: 5.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2085295, upper bound: 1.1790285
time: 5.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.3730879, 2.3811860
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.4218478, 2.4169359
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.9525914, 1.9457479
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -2.0270391, 2.0270391
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.6845613, 1.6617842
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.8446259, 1.8550766
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.3356438, 2.3312902
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.5567636, 1.5441663
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.1672826, 2.1800678
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.7292547, 1.7564707

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 5818

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4625

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1919729, upper bound: 1.1821893
time: 4.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1798042, upper bound: 1.1943926
time: 5.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.3741331, 2.3801403
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.4179492, 2.4208345
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.9499412, 1.9483984
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -2.0270391, 2.0270391
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.6778798, 1.6684656
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.8536181, 1.8460851
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.3431563, 2.3237786
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.5435581, 1.5573714
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.1779256, 2.1694229
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.7240520, 1.7616737

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 143

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 510

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1772093, upper bound: 1.1942223
time: 5.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1924192, upper bound: 1.1933382
time: 6.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.3819351, 2.3883967
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.4223495, 2.4232435
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.9253836, 1.9217372
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -2.0086765, 2.0261192
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.6796281, 1.6570511
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.8564172, 1.8619282
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.3106227, 2.2964182
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.5361354, 1.5305338
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.0721517, 2.0671711
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.7406311, 1.7734063

Time for backsubstitution: 14.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 904

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1913978, upper bound: 1.1837818
time: 5.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1922716, upper bound: 1.1829067
time: 4.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.3851480, 2.3851938
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.4250851, 2.4205079
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.9317513, 1.9153681
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -2.0270391, 2.0066359
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.6713088, 1.6653700
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.8622241, 1.8561206
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.3001528, 2.3068876
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.5145409, 1.5521293
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.0495944, 2.0897415
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.7402697, 1.7737677

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 143

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1950514, upper bound: 1.1837729
time: 4.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2029714, upper bound: 1.1757858
time: 8.12 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.3926163, 2.3990507
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.3781943, 2.3919253
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.9580736, 1.9679735
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -2.0270391, 2.0270391
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.6631129, 1.6759461
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.8592796, 1.8472345
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.2891016, 2.2635264
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.5455141, 1.5413356
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.1922641, 2.1706417
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.7303200, 1.7326312

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 5733
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 548

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5818

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1668493, upper bound: 1.2085675
time: 11.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1668818, upper bound: 1.1995779
time: 5.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.3966503, 2.3950176
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.3890057, 2.3811135
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.9539394, 1.9721076
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -2.0270391, 2.0270391
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.6731472, 1.6659101
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.8520689, 1.8544450
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.2720304, 2.2805972
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.5612841, 1.5255656
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.1802173, 2.1826885
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.7507029, 1.7122493

Time for backsubstitution: 14.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 5733

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4632

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1643875, upper bound: 1.1979584
time: 4.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1643875, upper bound: 1.2004505
time: 7.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.3805923, 2.3796468
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.4217105, 2.4136400
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.9382353, 1.9557528
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -2.0270391, 2.0270391
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.6972430, 1.7004279
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.8746789, 1.8735757
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.3263087, 2.3258922
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.6004949, 1.5792364
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.1443295, 2.1347816
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.7701669, 1.7469814

Time for backsubstitution: 14.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 5733
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 5818

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 510

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1742128, upper bound: 1.1996155
time: 5.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1831643, upper bound: 1.1987312
time: 5.21 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.3805971, 2.3796430
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.4220071, 2.4133430
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.9383616, 1.9556265
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -2.0270391, 2.0270391
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.6972690, 1.7004017
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.8746789, 1.8736901
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.3262534, 2.3259475
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.6004148, 1.5793165
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.1443887, 2.1347225
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.7698550, 1.7472932

Time for backsubstitution: 14.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 5733
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 4625

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1761199, upper bound: 1.1987394
time: 5.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1840378, upper bound: 1.1908911
time: 5.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.4058485, 2.4114189
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.3840170, 2.3855538
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.9599161, 1.9643793
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -2.0270391, 2.0270391
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.6505923, 1.6723993
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.8534639, 1.8437862
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.2890062, 2.2731929
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.5417571, 1.5519310
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.2003517, 2.2020183
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.7114098, 1.7198362

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 5733
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 548

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 904

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1765637, upper bound: 1.1980201
time: 5.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1855773, upper bound: 1.1890834
time: 5.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.4104557, 2.4068117
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.3782940, 2.3912773
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.9586229, 1.9656725
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -2.0270391, 2.0270391
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.6575489, 1.6654437
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.8563206, 1.8409300
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.2865658, 2.2756333
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.5413394, 1.5523484
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.2082200, 2.1941500
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.7192080, 1.7120383

Time for backsubstitution: 14.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5733
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 548

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5733

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1863370, upper bound: 1.1915354
time: 4.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1807739, upper bound: 1.1971020
time: 5.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.3947511, 2.3878202
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.4111280, 2.3978181
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.9508910, 1.9659135
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -2.0270391, 2.0270391
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.6721611, 1.6671782
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.8577542, 1.8591285
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.2845073, 2.3032262
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.5525737, 1.5303104
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.1837502, 2.1982188
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.7579875, 1.7176104

Time for backsubstitution: 14.49 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=1.9854326248168945
rel_dist={2: [-1.2085888365066806, 1.2085886275699598]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 5733
type: RSZ, layer: 1, pos: 927

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 904

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9431274, upper bound: 0.9484667
time: 4.70 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9484677, upper bound: 0.9431251
time: 5.19 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 9.90 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 9.90
Output dim: 2, lower bound: -0.9431274, upper bound: 0.9484667
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 9.90
Output dim: 2, lower bound: -0.9484677, upper bound: 0.9431251

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1507926, 2.1517496
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.2144747, 2.2176991
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8737631, 1.8728039
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.9365320, 1.9374135
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.5490880, 1.5489848
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7784271, 1.7777038
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1573372, 2.1550102
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4939144, 1.4942656
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9691205, 1.9691293
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5751829, 1.5765553

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 5733
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 548

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9389904, upper bound: 0.9484492
time: 5.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9431039, upper bound: 0.9443113
time: 5.06 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1517501, 2.1507926
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.2176995, 2.2144747
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8728037, 1.8737631
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.9374132, 1.9365320
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.5489848, 1.5490882
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7777038, 1.7784274
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1550107, 2.1573372
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4942653, 1.4939141
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9691291, 1.9691205
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5765553, 1.5751829

Time for backsubstitution: 14.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 5733
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 927

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9476580, upper bound: 0.9431239
time: 4.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9484670, upper bound: 0.9423202
time: 5.16 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 24.86 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.86
Output dim: 2, lower bound: -0.9389904, upper bound: 0.9484492
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.86
Output dim: 2, lower bound: -0.9431039, upper bound: 0.9443113
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.86
Output dim: 2, lower bound: -0.9476580, upper bound: 0.9431239
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.86
Output dim: 2, lower bound: -0.9484670, upper bound: 0.9423202

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1508026, 2.1517577
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.2144523, 2.2176776
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8737550, 1.8727968
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.9365520, 1.9374375
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.5490856, 1.5489819
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7784009, 1.7776723
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1573162, 2.1549883
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4939048, 1.4942553
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9691162, 1.9691231
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5751796, 1.5765514

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 5733
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4625

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4632

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9375827, upper bound: 0.9470424
time: 6.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9375808, upper bound: 0.9484468
time: 6.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1508007, 2.1517606
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.2144527, 2.2176771
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8737559, 1.8727956
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.9365563, 1.9374332
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.5490854, 1.5489824
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7783957, 1.7776771
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1573153, 2.1549892
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4939039, 1.4942560
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9691143, 1.9691250
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5751796, 1.5765522

Time for backsubstitution: 14.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 5733
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 143

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9425770, upper bound: 0.9443077
time: 5.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9431000, upper bound: 0.9437898
time: 5.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1517615, 2.1508074
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.2186728, 2.2156172
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8728061, 1.8738375
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.9388375, 1.9377408
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.5488982, 1.5490165
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7781334, 1.7787914
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1548328, 2.1571281
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4945650, 1.4941680
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9689384, 1.9689631
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5777578, 1.5762074

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 5733
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 525

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4625

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9475033, upper bound: 0.9302049
time: 5.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9294016, upper bound: 0.9429706
time: 5.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1517653, 2.1508050
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.2188420, 2.2154474
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8728786, 1.8737652
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.9386225, 1.9379563
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.5489130, 1.5490017
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7780676, 1.7788568
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1548014, 2.1571596
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4945192, 1.4942138
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9689722, 1.9689293
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5775795, 1.5763853

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5733
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 5841

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5733

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9484334, upper bound: 0.9367085
time: 5.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9428606, upper bound: 0.9422867
time: 5.04 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 24.75 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.75
Output dim: 2, lower bound: -0.9375827, upper bound: 0.9470424
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.75
Output dim: 2, lower bound: -0.9375808, upper bound: 0.9484468
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.75
Output dim: 2, lower bound: -0.9425770, upper bound: 0.9443077
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.75
Output dim: 2, lower bound: -0.9431000, upper bound: 0.9437898
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.75
Output dim: 2, lower bound: -0.9475033, upper bound: 0.9302049
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.75
Output dim: 2, lower bound: -0.9294016, upper bound: 0.9429706
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.75
Output dim: 2, lower bound: -0.9484334, upper bound: 0.9367085
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.75
Output dim: 2, lower bound: -0.9428606, upper bound: 0.9422867

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1505995, 2.1515169
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.2125902, 2.2161889
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8728728, 1.8717387
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.9365406, 1.9374285
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.5486259, 1.5485992
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7786522, 1.7779975
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1572080, 2.1549220
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4905348, 1.4914458
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9684367, 1.9683073
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5715513, 1.5735266

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 5733

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4625

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9388305, upper bound: 0.9341133
time: 5.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9246471, upper bound: 0.9468882
time: 5.21 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1505623, 2.1515546
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.2129636, 2.2158155
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8726964, 1.8719149
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.9365425, 1.9374266
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.5487025, 1.5485225
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7787256, 1.7779236
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1572495, 2.1548805
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4910955, 1.4908857
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9683003, 1.9684439
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5721550, 1.5729232

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 5733
type: RSZ, layer: 1, pos: 5818

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 525

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9372610, upper bound: 0.9484470
time: 8.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9374129, upper bound: 0.9440077
time: 8.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1512241, 2.1522822
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.2168140, 2.2206402
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8737035, 1.8733132
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.9268956, 1.9293897
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.5496613, 1.5496944
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7798843, 1.7788715
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1630588, 2.1595609
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4902060, 1.4898067
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9788456, 1.9768999
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5745015, 1.5757377

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 5733
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 6111

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5818

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9425580, upper bound: 0.9443070
time: 5.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9425766, upper bound: 0.9389628
time: 5.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1513214, 2.1521835
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.2174158, 2.2200384
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8742738, 1.8727429
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.9285126, 1.9277730
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.5497969, 1.5495585
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7795901, 1.7791655
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1618872, 2.1607325
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4894545, 1.4905579
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9768896, 1.9788561
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5743651, 1.5758748

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 5733
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 5841

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4625

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9429461, upper bound: 0.9308530
time: 6.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9301722, upper bound: 0.9436338
time: 5.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1519184, 2.1478000
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.2083006, 2.2160845
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8731961, 1.8644371
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.9391227, 1.9312794
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.5490205, 1.5460677
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7707739, 1.7791066
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1456542, 2.1575229
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4925432, 1.4942775
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9679646, 1.9689996
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5778327, 1.5744290

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 5733
type: RSZ, layer: 1, pos: 927

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5841

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9423129, upper bound: 0.9302044
time: 7.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9475023, upper bound: 0.9250144
time: 5.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1487541, 2.1508074
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.2186728, 2.2052450
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8634057, 1.8738375
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.9323764, 1.9377408
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.5459495, 1.5490165
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7781334, 1.7714324
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1548328, 2.1479495
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4945650, 1.4921460
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9689384, 1.9679897
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5759792, 1.5762074

Time for backsubstitution: 14.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 5733
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 927

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 525

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9303424, upper bound: 0.9429679
time: 6.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9250074, upper bound: 0.9386371
time: 5.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1533947, 2.1468215
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.2142725, 2.2173300
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8752069, 1.8680677
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.9367681, 1.9387374
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.5376346, 1.5536116
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7742033, 1.7804356
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1466341, 2.1605043
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4899349, 1.4960749
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9722514, 1.9609463
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5827818, 1.5637462

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 4632

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 510

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9476113, upper bound: 0.9367093
time: 5.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9484334, upper bound: 0.9366950
time: 5.16 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1477814, 2.1508050
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.2188420, 2.2108779
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8671808, 1.8737652
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.9386225, 1.9361019
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.5489130, 1.5377229
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7780676, 1.7749920
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1548014, 2.1489925
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4945192, 1.4896290
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9609890, 1.9689293
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5649405, 1.5763853

Time for backsubstitution: 14.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 510

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 525

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9385191, upper bound: 0.9422846
time: 5.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9428584, upper bound: 0.9379318
time: 5.40 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 25.81 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.81
Output dim: 2, lower bound: -0.9388305, upper bound: 0.9341133
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.81
Output dim: 2, lower bound: -0.9246471, upper bound: 0.9468882
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.81
Output dim: 2, lower bound: -0.9372610, upper bound: 0.9484470
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.81
Output dim: 2, lower bound: -0.9374129, upper bound: 0.9440077
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.81
Output dim: 2, lower bound: -0.9425580, upper bound: 0.9443070
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.81
Output dim: 2, lower bound: -0.9425766, upper bound: 0.9389628
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.81
Output dim: 2, lower bound: -0.9429461, upper bound: 0.9308530
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.81
Output dim: 2, lower bound: -0.9301722, upper bound: 0.9436338
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.81
Output dim: 2, lower bound: -0.9423129, upper bound: 0.9302044
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.81
Output dim: 2, lower bound: -0.9475023, upper bound: 0.9250144
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.81
Output dim: 2, lower bound: -0.9303424, upper bound: 0.9429679
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.81
Output dim: 2, lower bound: -0.9250074, upper bound: 0.9386371
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.81
Output dim: 2, lower bound: -0.9476113, upper bound: 0.9367093
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.81
Output dim: 2, lower bound: -0.9484334, upper bound: 0.9366950
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.81
Output dim: 2, lower bound: -0.9385191, upper bound: 0.9422846
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.81
Output dim: 2, lower bound: -0.9428584, upper bound: 0.9379318

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1507564, 2.1485095
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.2022190, 2.2166572
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8732634, 1.8623385
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.9368248, 1.9309661
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.5487480, 1.5456495
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7712932, 1.7783124
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1480293, 2.1553164
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4885130, 1.4915550
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9674630, 1.9683437
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5716264, 1.5717487

Time for backsubstitution: 14.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 5733
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 143

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 927

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9333179, upper bound: 0.9340918
time: 6.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9388186, upper bound: 0.9285903
time: 5.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1475921, 2.1515169
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.2125902, 2.2058177
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8634729, 1.8717387
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.9300785, 1.9374285
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.5456769, 1.5485992
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7786522, 1.7706385
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1572080, 2.1457429
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4905348, 1.4894235
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9684367, 1.9673338
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5697734, 1.5735266

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 5733
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9260438, upper bound: 0.9385868
time: 5.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9163502, upper bound: 0.9468809
time: 4.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1642728, 2.1675754
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1760778, 2.1851106
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8890886, 1.8859446
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.9414043, 1.9431062
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.5143065, 1.5198622
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7581358, 1.7532129
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1085029, 2.0963764
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4370630, 1.4458672
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.0108852, 2.0041449
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5360513, 1.5484676

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 5733

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 510

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9364111, upper bound: 0.9484429
time: 5.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9364112, upper bound: 0.9476155
time: 5.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1665778, 2.1652651
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1822557, 2.1789298
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8867264, 1.8883042
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.9422221, 1.9422882
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.5200403, 1.5141263
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7540150, 1.7573333
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.0987453, 2.1061316
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4460742, 1.4368535
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.0040016, 2.0110283
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5476985, 1.5368199

Time for backsubstitution: 14.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 5733
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 143

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9368874, upper bound: 0.9440043
time: 5.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9374091, upper bound: 0.9434823
time: 8.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1551204, 2.1568584
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1341686, 2.1517820
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8939743, 1.8905737
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.9145660, 1.9146006
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.5249796, 1.5312744
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7487187, 1.7414706
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1145353, 2.1013212
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4160748, 1.4280393
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.0329351, 2.0229743
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5839939, 1.5981338

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 5733
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 927

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 548

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9425580, upper bound: 0.9443068
time: 5.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9425571, upper bound: 0.9438060
time: 5.06 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1557994, 2.1561794
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1479549, 2.1379948
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8909645, 1.8935835
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.9121070, 1.9152989
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.5310335, 1.5250121
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7424831, 1.7477031
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1048188, 2.1110353
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4284382, 1.4156752
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.0249195, 2.0309711
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5968981, 1.5852292

Time for backsubstitution: 14.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 5733
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 510

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5841

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9373867, upper bound: 0.9389615
time: 5.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9425758, upper bound: 0.9337752
time: 5.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1514783, 2.1491761
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.2070441, 2.2205062
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8746638, 1.8633423
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.9287968, 1.9213104
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.5499194, 1.5466093
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7722311, 1.7794807
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1527081, 2.1611276
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4874327, 1.4906673
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9759154, 1.9788923
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5744398, 1.5740969

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 5733
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 510

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5818

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9429271, upper bound: 0.9308522
time: 6.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9429457, upper bound: 0.9255047
time: 5.27 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1483140, 2.1521835
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.2174158, 2.2096667
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8648734, 1.8727429
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.9220505, 1.9277730
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.5468481, 1.5495585
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7795901, 1.7718067
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1618872, 2.1515541
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4894545, 1.4885356
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9768896, 1.9778824
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5725868, 1.5758748

Time for backsubstitution: 14.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 5733
type: RSZ, layer: 1, pos: 6111

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 525

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9251817, upper bound: 0.9434697
time: 5.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9301705, upper bound: 0.9433257
time: 5.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1624260, 2.1601367
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1989307, 2.2082782
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8513870, 1.8462675
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.8723764, 1.8756666
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.5252414, 1.5175344
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7508795, 1.7625308
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1157684, 2.1216540
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4309120, 1.4203062
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9034657, 1.8916106
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5767994, 1.5731893

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 5733
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 548

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 525

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9379496, upper bound: 0.9302017
time: 5.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9423106, upper bound: 0.9258323
time: 10.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1642551, 2.1583066
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.2004938, 2.2067151
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8550267, 1.8426280
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.8835101, 1.8645332
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.5204875, 1.5222884
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7541983, 1.7592120
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1097851, 2.1276369
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4185715, 1.4326463
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.8905759, 1.9045005
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5765929, 1.5733957

Time for backsubstitution: 14.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 5733
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 548

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 143

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9469786, upper bound: 0.9250112
time: 6.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9474986, upper bound: 0.9244917
time: 6.45 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1624651, 2.1668224
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1817718, 2.1745224
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8797960, 1.8878660
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.9372377, 1.9434202
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.5115522, 1.5203534
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7575426, 1.7467220
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1060839, 2.0894449
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4405322, 1.4471245
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.0115342, 2.0037012
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5398712, 1.5517457

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5733
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 5841

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5733

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9303116, upper bound: 0.9373633
time: 5.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9247303, upper bound: 0.9429341
time: 7.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1647701, 2.1645179
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1879501, 2.1683445
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8774338, 1.8902285
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.9380555, 1.9426022
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.5172871, 1.5146199
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7534223, 1.7508426
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.0963287, 2.0992000
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4495444, 1.4381130
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.0046501, 2.0105853
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5515180, 1.5400984

Time for backsubstitution: 14.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 5733
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5818

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9249855, upper bound: 0.9386359
time: 5.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9347322, upper bound: 0.9386160
time: 5.45 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1689491, 2.1650085
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1975265, 2.1976676
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8789096, 1.8711772
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.9229646, 1.9267933
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.5268459, 1.5467982
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7644100, 1.7721419
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1396413, 2.1521168
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4915781, 1.4974804
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9883761, 1.9815674
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5560169, 1.5410631

Time for backsubstitution: 14.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 143

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4632

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9476089, upper bound: 0.9353026
time: 5.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9462040, upper bound: 0.9367089
time: 5.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1715813, 2.1623759
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1946101, 2.2009382
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8783164, 1.8719161
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.9252625, 1.9249341
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.5308213, 1.5428228
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7660422, 1.7706423
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1382465, 2.1535144
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4913397, 1.4977188
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9928727, 1.9770713
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5604734, 1.5369813

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 5818

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 143

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9479073, upper bound: 0.9366912
time: 5.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9484297, upper bound: 0.9361719
time: 6.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1614914, 2.1668200
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1819415, 2.1801553
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8835721, 1.8877938
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.9434838, 1.9417820
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.5145164, 1.5090597
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7574773, 1.7502804
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1060519, 2.0904875
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4404864, 1.4446077
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.0035853, 2.0046411
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5288327, 1.5519240

Time for backsubstitution: 14.80 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=1.8685765266418457
rel_dist={2: [-0.9484739266423019, 0.9484713204465995]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 5733
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 5841

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8326438, upper bound: 0.8355755
time: 7.99 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8326438, upper bound: 0.8326434
time: 6.85 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 14.85 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 14.85
Output dim: 2, lower bound: -0.8326438, upper bound: 0.8355755
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 14.85
Output dim: 2, lower bound: -0.8326438, upper bound: 0.8326434

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.0738068, 2.0738044
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1708894, 2.1708894
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8296151, 1.8296158
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.8862753, 1.8862786
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4940658, 1.4940658
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7475677, 1.7475643
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1090460, 2.1090455
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4533706, 1.4533699
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9036570, 1.9036558
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5030918, 1.5030913

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 5733
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 143

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8324730, upper bound: 0.8355712
time: 5.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8324730, upper bound: 0.8354158
time: 7.10 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.0738044, 2.0738068
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1708894, 2.1708889
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8296161, 1.8296149
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.8862786, 1.8862753
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4940655, 1.4940658
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7475643, 1.7475679
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1090455, 2.1090460
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4533701, 1.4533703
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9036560, 1.9036572
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5030918, 1.5030918

Time for backsubstitution: 14.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 5733
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 143

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8322234, upper bound: 0.8326404
time: 7.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8355704, upper bound: 0.8322225
time: 7.07 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 29.46 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 29.46
Output dim: 2, lower bound: -0.8324730, upper bound: 0.8355712
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 29.46
Output dim: 2, lower bound: -0.8324730, upper bound: 0.8354158
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 29.46
Output dim: 2, lower bound: -0.8322234, upper bound: 0.8326404
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 29.46
Output dim: 2, lower bound: -0.8355704, upper bound: 0.8322225

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.0738187, 2.0738187
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1718607, 2.1719885
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8296165, 1.8296719
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.8876457, 1.8874876
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4939802, 1.4939911
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7479806, 1.7479281
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1088610, 2.1088367
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4536586, 1.4536235
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9034653, 1.9034896
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5042498, 1.5041158

Time for backsubstitution: 14.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5733
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 6111

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5733

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8324389, upper bound: 0.8299452
time: 8.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8268530, upper bound: 0.8355380
time: 4.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.0738206, 2.0738168
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1719880, 2.1718612
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8296709, 1.8296175
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.8874841, 1.8876491
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4939914, 1.4939802
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7479315, 1.7479773
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1088371, 2.1088605
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4536242, 1.4536579
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9034910, 1.9034641
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5041163, 1.5042496

Time for backsubstitution: 14.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5733
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 5818

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5733

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8268530, upper bound: 0.8297797
time: 11.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8268530, upper bound: 0.8353823
time: 12.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.0742269, 2.0743027
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1732516, 2.1737022
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8295622, 1.8299890
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.8766193, 1.8778286
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4946425, 1.4947444
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7489796, 1.7487626
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1144943, 2.1136158
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4494839, 1.4489212
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9128971, 1.9114311
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5023799, 1.5022779

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 5733
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 525

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 510

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8320504, upper bound: 0.8326393
time: 5.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8320504, upper bound: 0.8324684
time: 6.47 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.0743003, 2.0742288
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1737027, 2.1732512
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8299904, 1.8295612
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.8778319, 1.8766160
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4947441, 1.4946429
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7487588, 1.7489831
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1136150, 2.1144946
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4489212, 1.4494846
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9114294, 1.9128983
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5022774, 1.5023804

Time for backsubstitution: 14.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5733

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5818

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8281770, upper bound: 0.8322225
time: 9.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8355697, upper bound: 0.8281759
time: 7.25 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 31.43 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 31.43
Output dim: 2, lower bound: -0.8324389, upper bound: 0.8299452
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 31.43
Output dim: 2, lower bound: -0.8268530, upper bound: 0.8355380
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 31.43
Output dim: 2, lower bound: -0.8268530, upper bound: 0.8297797
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 31.43
Output dim: 2, lower bound: -0.8268530, upper bound: 0.8353823
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 31.43
Output dim: 2, lower bound: -0.8320504, upper bound: 0.8326393
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 31.43
Output dim: 2, lower bound: -0.8320504, upper bound: 0.8324684
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 31.43
Output dim: 2, lower bound: -0.8281770, upper bound: 0.8322225
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 31.43
Output dim: 2, lower bound: -0.8355697, upper bound: 0.8281759

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.0740452, 2.0698352
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1672916, 2.1722579
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8299384, 1.8239739
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.8857908, 1.8876095
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4827011, 1.4946284
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7441158, 1.7481461
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1006932, 2.1093028
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4490743, 1.4538736
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9039292, 1.8955064
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5049925, 1.4914770

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 4625

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 548

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8266154, upper bound: 0.8299481
time: 5.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8266155, upper bound: 0.8297183
time: 5.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.0698352, 2.0738187
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1718607, 2.1674190
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8239188, 1.8296719
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.8876457, 1.8856328
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4939802, 1.4827118
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7479806, 1.7440634
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1088610, 2.1006691
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4536586, 1.4490390
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.8954825, 1.9034896
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.4916115, 1.5041158

Time for backsubstitution: 14.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 927

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 143

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8264321, upper bound: 0.8355351
time: 5.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8264321, upper bound: 0.8351218
time: 5.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.0698371, 2.0738168
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1719880, 2.1672916
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8239732, 1.8296175
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.8874841, 1.8857942
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4939914, 1.4827008
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7479315, 1.7441125
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1088371, 2.1006930
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4536242, 1.4490733
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.8955073, 1.9034641
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.4914775, 1.5042496

Time for backsubstitution: 14.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 4625

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 548

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8266155, upper bound: 0.8353803
time: 4.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8266155, upper bound: 0.8351476
time: 5.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.0897794, 2.0918298
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1560431, 2.1540413
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8336620, 1.8335347
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.8628159, 1.8657484
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4836876, 1.4867709
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7391853, 1.7401927
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1071525, 2.1052279
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4510674, 1.4503262
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9286489, 1.9305553
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.4756148, 1.4788542

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 5733
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 4625

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4632

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8309703, upper bound: 0.8315584
time: 5.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8309703, upper bound: 0.8326365
time: 5.22 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.0917540, 2.0898557
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1535902, 2.1564941
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8331079, 1.8340888
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.8645391, 1.8640251
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4866688, 1.4837897
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7404099, 1.7389684
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1061063, 2.1062737
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4508891, 1.4505050
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9320211, 1.9271834
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.4789565, 1.4755123

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 5733
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 4632

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4625

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8320254, upper bound: 0.8227727
time: 5.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8223550, upper bound: 0.8324433
time: 6.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.0787053, 2.0781250
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1013975, 2.0906057
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8472528, 1.8490810
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.8630424, 1.8636711
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4758549, 1.4710569
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7113571, 1.7162557
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.0553756, 2.0635405
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.3840623, 1.3753533
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9575052, 1.9649844
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5071702, 1.4975948

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 5733
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 548

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 927

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8315562, upper bound: 0.8281442
time: 8.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8355307, upper bound: 0.8241679
time: 6.67 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 29.97 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 29.97
Output dim: 2, lower bound: -0.8266154, upper bound: 0.8299481
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 29.97
Output dim: 2, lower bound: -0.8266155, upper bound: 0.8297183
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.97
Output dim: 2, lower bound: -0.8264321, upper bound: 0.8355351
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.97
Output dim: 2, lower bound: -0.8264321, upper bound: 0.8351218
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.97
Output dim: 2, lower bound: -0.8266155, upper bound: 0.8353803
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.97
Output dim: 2, lower bound: -0.8266155, upper bound: 0.8351476
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 29.97
Output dim: 2, lower bound: -0.8309703, upper bound: 0.8315584
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.97
Output dim: 2, lower bound: -0.8309703, upper bound: 0.8326365
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 29.97
Output dim: 2, lower bound: -0.8320254, upper bound: 0.8227727
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.97
Output dim: 2, lower bound: -0.8223550, upper bound: 0.8324433
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 29.97
Output dim: 2, lower bound: -0.8315562, upper bound: 0.8281442
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.97
Output dim: 2, lower bound: -0.8355307, upper bound: 0.8241679

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.0702581, 2.0743141
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1742239, 2.1702332
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8238659, 1.8300459
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.8779860, 1.8771861
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4945567, 1.4833900
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7493958, 1.7452583
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1143103, 2.1052401
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4497731, 1.4445899
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9047251, 1.9112632
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.4909000, 1.5033011

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 5841

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 510

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8264276, upper bound: 0.8355362
time: 5.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8264309, upper bound: 0.8353765
time: 5.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.0703316, 2.0742407
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1746750, 2.1697817
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8242936, 1.8296182
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.8791986, 1.8759732
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4946582, 1.4832884
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7491755, 1.7454786
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1134315, 2.1061189
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4492095, 1.4451535
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9032574, 1.9127305
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.4907970, 1.5034039

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 548

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 525

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8262713, upper bound: 0.8351220
time: 7.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8267913, upper bound: 0.8320685
time: 9.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.0733929, 2.0778198
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1603050, 2.1539373
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8339429, 1.8384514
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.9065309, 1.9026990
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4739735, 1.4598202
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7171249, 1.7171590
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1043968, 2.0994725
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4140549, 1.4038441
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9280925, 1.9406106
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.4758911, 1.4864330

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 904

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 510

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8265934, upper bound: 0.8353826
time: 5.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8267823, upper bound: 0.8353805
time: 5.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.0738411, 2.0773716
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1586342, 2.1556082
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8328071, 1.8395872
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.9043889, 1.9048409
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4711101, 1.4626836
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7209787, 1.7133052
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1076164, 2.0962534
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4083948, 1.4095032
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9326544, 1.9360485
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.4736614, 1.4886632

Time for backsubstitution: 14.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 510

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4632

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8255358, upper bound: 0.8340644
time: 4.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8255358, upper bound: 0.8351423
time: 6.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.0895386, 2.0916176
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1544614, 2.1521792
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8326039, 1.8326087
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.8628068, 1.8657377
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4832859, 1.4863120
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7394915, 1.7404435
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1070743, 2.1051188
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4481196, 1.4469573
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9278326, 1.9298418
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.4724395, 1.4752262

Time for backsubstitution: 14.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 5733
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 525

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8308343, upper bound: 0.8326184
time: 5.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8308333, upper bound: 0.8325201
time: 5.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.0887489, 2.0892234
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1513481, 2.1461220
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8237071, 1.8320308
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.8580770, 1.8626232
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4837192, 1.4831436
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7388067, 1.7316101
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1041079, 2.0970953
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4504657, 1.4484830
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9318042, 1.9262097
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.4771781, 1.4751236

Time for backsubstitution: 14.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 5733
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 6111

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8252686, upper bound: 0.8324204
time: 5.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8223346, upper bound: 0.8324396
time: 8.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.0843163, 2.0829215
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1001544, 2.0891857
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8385286, 1.8391106
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.8476715, 1.8461025
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4726945, 1.4682899
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7087259, 1.7132487
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.0414052, 2.0513120
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.3576927, 1.3522822
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9310255, 1.9418180
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5059366, 1.4965153

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 5733
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 4632

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 904

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8312652, upper bound: 0.8239297
time: 5.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8355272, upper bound: 0.8239331
time: 6.55 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 26.27 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.27
Output dim: 2, lower bound: -0.8264276, upper bound: 0.8355362
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.27
Output dim: 2, lower bound: -0.8264309, upper bound: 0.8353765
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.27
Output dim: 2, lower bound: -0.8262713, upper bound: 0.8351220
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 26.27
Output dim: 2, lower bound: -0.8267913, upper bound: 0.8320685
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.27
Output dim: 2, lower bound: -0.8265934, upper bound: 0.8353826
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.27
Output dim: 2, lower bound: -0.8267823, upper bound: 0.8353805
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.27
Output dim: 2, lower bound: -0.8255358, upper bound: 0.8340644
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.27
Output dim: 2, lower bound: -0.8255358, upper bound: 0.8351423
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.27
Output dim: 2, lower bound: -0.8308343, upper bound: 0.8326184
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.27
Output dim: 2, lower bound: -0.8308333, upper bound: 0.8325201
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.27
Output dim: 2, lower bound: -0.8252686, upper bound: 0.8324204
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.27
Output dim: 2, lower bound: -0.8223346, upper bound: 0.8324396
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 26.27
Output dim: 2, lower bound: -0.8312652, upper bound: 0.8239297
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.27
Output dim: 2, lower bound: -0.8355272, upper bound: 0.8239331

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.0858111, 2.0918422
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1570144, 2.1505704
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8275294, 1.8331556
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.8641825, 1.8651063
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4837677, 1.4755833
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7396026, 1.7366886
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1069703, 2.0968513
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4513569, 1.4459949
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9208493, 1.9307611
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.4641347, 1.4798784

Time for backsubstitution: 14.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 4632

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8264220, upper bound: 0.8292323
time: 5.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8201328, upper bound: 0.8355317
time: 5.27 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.0877857, 2.0898676
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1545615, 2.1527576
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8269753, 1.8336010
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.8655772, 1.8633831
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4867492, 1.4726017
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7407269, 1.7354643
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1059217, 2.0978975
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4511776, 1.4461737
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9242206, 1.9273889
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.4671960, 1.4765363

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 904

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 927

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8224327, upper bound: 0.8353638
time: 5.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8264209, upper bound: 0.8313645
time: 6.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.0840430, 2.0896859
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1377892, 2.1375308
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8400955, 1.8436482
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.8840594, 1.8814487
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4602613, 1.4531937
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7275553, 1.7207682
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.0622468, 2.0476153
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.3951786, 1.3978822
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9441228, 1.9484324
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.4546952, 1.4760385

Time for backsubstitution: 14.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 6111

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 927

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8222613, upper bound: 0.8350807
time: 4.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8222613, upper bound: 0.8311013
time: 5.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.0889459, 2.0953479
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1428304, 2.1342759
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8374987, 1.8415611
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.8927264, 1.8902895
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4631846, 1.4520117
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7073309, 1.7084899
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.0970540, 2.0910833
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4156384, 1.4052491
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9442167, 1.9601068
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.4491267, 1.4627299

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 904

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 143

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8264095, upper bound: 0.8349705
time: 4.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8264095, upper bound: 0.8349583
time: 5.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.0909204, 2.0933733
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1406431, 2.1367288
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8370533, 1.8421154
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.8944497, 1.8888950
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4661660, 1.4490305
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7085550, 1.7073655
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.0960083, 2.0921319
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4154596, 1.4054279
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9475884, 1.9567347
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.4524689, 1.4596686

Time for backsubstitution: 14.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 143

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5818

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8227388, upper bound: 0.8353801
time: 5.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8267817, upper bound: 0.8313454
time: 6.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.0736270, 2.0771294
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1567712, 2.1540256
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8318815, 1.8385296
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.9043779, 1.9048314
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4706509, 1.4622815
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7212293, 1.7136116
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1075077, 2.0961752
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4050257, 1.4065535
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9319401, 1.9352319
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.4700336, 1.4854882

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 5841

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4625

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8158385, upper bound: 0.8243495
time: 6.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8158385, upper bound: 0.8340412
time: 5.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.0735984, 2.0771575
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1570511, 2.1537457
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8317490, 1.8386617
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.9043794, 1.9048300
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4707081, 1.4622242
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7212846, 1.7135563
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1075382, 2.0961447
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4054453, 1.4061334
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9318380, 1.9353344
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.4704862, 1.4850357

Time for backsubstitution: 14.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 927

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5841

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8215721, upper bound: 0.8351414
time: 5.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8215721, upper bound: 0.8311653
time: 5.47 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1032495, 2.1070566
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1175752, 2.1199265
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8484020, 1.8466372
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.8676677, 1.8712120
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4488869, 1.4562135
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7178726, 1.7157342
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.0558867, 2.0466151
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.3940878, 1.3996843
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9686980, 1.9655437
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.4363327, 1.4478550

Time for backsubstitution: 14.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 5733
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 548

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 927

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8268268, upper bound: 0.8325898
time: 5.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8308251, upper bound: 0.8286015
time: 5.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1049824, 2.1053281
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1222110, 2.1152930
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8466325, 1.8484089
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.8682809, 1.8705986
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4531887, 1.4519129
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7147822, 1.7188249
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.0485706, 2.0539336
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4008479, 1.3929257
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9635358, 1.9707067
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.4450688, 1.4391196

Time for backsubstitution: 14.65 seconds
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=1.8296241760253906
rel_dist={2: [-0.835577596891917, 0.8355774930246449]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2434.66 seconds
