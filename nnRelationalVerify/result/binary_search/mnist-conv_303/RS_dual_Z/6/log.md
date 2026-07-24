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
execution time: IAR + LP analysis = 15.31 + 32.44 = 47.75 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3552.25 seconds, max iter: 100)

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
Binary search time: 204.80 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Relational Split (RS_dual_Z) starts
Time budget: 3347.46 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 5733
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 510

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2077084, upper bound: 1.2085906
time: 5.73 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2085887, upper bound: 1.2077068
time: 5.62 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 11.56 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 11.56
Output dim: 2, lower bound: -1.2077084, upper bound: 1.2085906
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 11.56
Output dim: 2, lower bound: -1.2085887, upper bound: 1.2077068

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.3804703, 2.3850770
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.4086161, 2.4028926
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.9857671, 1.9857671
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -2.0270391, 2.0270391
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.7009237, 1.7078803
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.8746789, 1.8746789
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.3428574, 2.3404164
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.6257854, 1.6253682
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.1814623, 2.1893299
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.7975526, 1.8053503

Time for backsubstitution: 14.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5733
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 5733

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2076726, upper bound: 1.2029960
time: 5.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2021158, upper bound: 1.2085564
time: 5.11 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.3850765, 2.3804698
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.4028931, 2.4086161
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.9857671, 1.9857671
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -2.0270391, 2.0270391
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.7078803, 1.7009237
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.8746789, 1.8746789
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.3404169, 2.3428569
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.6253686, 1.6257856
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.1893301, 2.1814620
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.8053503, 1.7975526

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5733
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 5733

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2085545, upper bound: 1.2021144
time: 5.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2029957, upper bound: 1.2076745
time: 5.90 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 26.55 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 26.55
Output dim: 2, lower bound: -1.2076726, upper bound: 1.2029960
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 26.55
Output dim: 2, lower bound: -1.2021158, upper bound: 1.2085564
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 26.55
Output dim: 2, lower bound: -1.2085545, upper bound: 1.2021144
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 26.55
Output dim: 2, lower bound: -1.2029957, upper bound: 1.2076745

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.3863091, 2.3810935
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.4040470, 2.4096146
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.9857671, 1.9832797
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -2.0270391, 2.0270391
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.6896439, 1.7244055
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.8718328, 1.8746789
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.3346906, 2.3523955
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.6212020, 1.6320646
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.1931887, 2.1813471
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.8161359, 1.7927120

Time for backsubstitution: 14.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 4625

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2075681, upper bound: 1.1906043
time: 5.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1952570, upper bound: 1.2028920
time: 5.26 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.3764863, 2.3850770
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.4086161, 2.3983235
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.9845729, 1.9857671
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -2.0270391, 2.0270391
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.7009237, 1.6966002
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.8746789, 1.8746789
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.3428574, 2.3322501
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.6257854, 1.6207842
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.1734791, 2.1893299
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.7849135, 1.8053503

Time for backsubstitution: 14.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 4625

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2020100, upper bound: 1.1960619
time: 5.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1897824, upper bound: 1.2084518
time: 5.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.3909173, 2.3764863
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.3983235, 2.4153376
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.9857671, 1.9845729
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -2.0270391, 2.0270391
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.6966004, 1.7174494
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.8746789, 1.8746789
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.3322501, 2.3548360
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.6207843, 1.6324818
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.2010565, 2.1734793
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.8239341, 1.7849140

Time for backsubstitution: 14.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 4625

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2084498, upper bound: 1.1897834
time: 5.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1960617, upper bound: 1.2020103
time: 5.19 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.3810935, 2.3804698
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.4028931, 2.4040465
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.9832797, 1.9857671
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -2.0270391, 2.0270391
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.7078803, 1.6896441
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.8746789, 1.8718328
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.3404169, 2.3346906
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.6253686, 1.6212015
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.1813469, 2.1814620
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.7927117, 1.7975526

Time for backsubstitution: 14.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 4625

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2028932, upper bound: 1.1952586
time: 6.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1906045, upper bound: 1.2075673
time: 6.39 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 27.42 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.42
Output dim: 2, lower bound: -1.2075681, upper bound: 1.1906043
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.42
Output dim: 2, lower bound: -1.1952570, upper bound: 1.2028920
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.42
Output dim: 2, lower bound: -1.2020100, upper bound: 1.1960619
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.42
Output dim: 2, lower bound: -1.1897824, upper bound: 1.2084518
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.42
Output dim: 2, lower bound: -1.2084498, upper bound: 1.1897834
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.42
Output dim: 2, lower bound: -1.1960617, upper bound: 1.2020103
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.42
Output dim: 2, lower bound: -1.2028932, upper bound: 1.1952586
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.42
Output dim: 2, lower bound: -1.1906045, upper bound: 1.2075673

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.3888416, 2.3780885
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.3936753, 2.4182119
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.9857671, 1.9738784
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -2.0270391, 2.0270391
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.6920686, 1.7214558
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.8644738, 1.8746789
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.3255119, 2.3599701
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.6191797, 1.6337729
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.1922154, 2.1821399
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.8176010, 1.7909336

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 525

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1995148, upper bound: 1.1906046
time: 5.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2075684, upper bound: 1.1824580
time: 6.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.3833046, 2.3810935
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.4040470, 2.3992429
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.9857671, 1.9832797
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -2.0270391, 2.0270391
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.6866941, 1.7244055
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.8718328, 1.8746789
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.3346906, 2.3432169
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.6212020, 1.6300426
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.1931887, 2.1803741
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.8143580, 1.7927120

Time for backsubstitution: 14.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 525

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1872657, upper bound: 1.2028940
time: 6.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1952573, upper bound: 1.1947649
time: 5.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.3790188, 2.3820710
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.3982439, 2.4069209
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.9857671, 1.9795771
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -2.0270391, 2.0270391
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.7033484, 1.6936505
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.8683386, 1.8746789
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.3336782, 2.3398247
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.6237636, 1.6224924
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.1725063, 2.1901226
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.7863786, 1.8035722

Time for backsubstitution: 14.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 525

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1939563, upper bound: 1.1960622
time: 6.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2020103, upper bound: 1.1880127
time: 13.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.3734808, 2.3850770
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.4086161, 2.3879519
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.9751716, 1.9857671
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -2.0270391, 2.0270391
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.6979740, 1.6966002
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.8746789, 1.8673303
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.3428574, 2.3230715
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.6257854, 1.6187621
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.1734791, 2.1883569
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.7831357, 1.8053503

Time for backsubstitution: 14.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 525

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1817252, upper bound: 1.2084497
time: 4.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1817272, upper bound: 1.2003319
time: 5.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.3934507, 2.3734813
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.3879519, 2.4239354
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.9857671, 1.9751716
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -2.0270391, 2.0270391
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.6990283, 1.7144997
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.8673306, 1.8746789
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.3230715, 2.3624105
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.6187620, 1.6341902
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.2000837, 2.1742737
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.8254011, 1.7831354

Time for backsubstitution: 14.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 525

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2003309, upper bound: 1.1897822
time: 5.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2084501, upper bound: 1.1817253
time: 6.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.3879118, 2.3764863
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.3983235, 2.4049659
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.9857671, 1.9845729
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -2.0270391, 2.0270391
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.6936507, 1.7174494
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.8746789, 1.8740001
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.3322501, 2.3456573
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.6207843, 1.6304598
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.2010565, 2.1725063
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.8221562, 1.7849140

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 525

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1880128, upper bound: 1.2020101
time: 4.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1960620, upper bound: 1.1939566
time: 5.27 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.3836269, 2.3774638
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.3925209, 2.4126444
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.9857671, 1.9808702
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -2.0270391, 2.0270391
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.7103083, 1.6866944
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.8711948, 1.8746789
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.3312378, 2.3422651
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.6233468, 1.6229098
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.1803741, 2.1822565
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.7941787, 1.7957740

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 525

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1947646, upper bound: 1.1952574
time: 5.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2028920, upper bound: 1.1872657
time: 6.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.3780890, 2.3804698
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.4028931, 2.3936753
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.9738784, 1.9857671
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -2.0270391, 2.0270391
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.7049305, 1.6896441
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.8746789, 1.8644738
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.3404169, 2.3255119
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.6253686, 1.6191795
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.1813469, 2.1804891
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.7909338, 1.7975526

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 525

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5818

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1816618, upper bound: 1.2075675
time: 6.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1906038, upper bound: 1.1985809
time: 6.35 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 42.48 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 42.48
Output dim: 2, lower bound: -1.1995148, upper bound: 1.1906046
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 42.48
Output dim: 2, lower bound: -1.2075684, upper bound: 1.1824580
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 42.48
Output dim: 2, lower bound: -1.1872657, upper bound: 1.2028940
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 42.48
Output dim: 2, lower bound: -1.1952573, upper bound: 1.1947649
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 42.48
Output dim: 2, lower bound: -1.1939563, upper bound: 1.1960622
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 42.48
Output dim: 2, lower bound: -1.2020103, upper bound: 1.1880127
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 42.48
Output dim: 2, lower bound: -1.1817252, upper bound: 1.2084497
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 42.48
Output dim: 2, lower bound: -1.1817272, upper bound: 1.2003319
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 42.48
Output dim: 2, lower bound: -1.2003309, upper bound: 1.1897822
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 42.48
Output dim: 2, lower bound: -1.2084501, upper bound: 1.1817253
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 42.48
Output dim: 2, lower bound: -1.1880128, upper bound: 1.2020101
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 42.48
Output dim: 2, lower bound: -1.1960620, upper bound: 1.1939566
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 42.48
Output dim: 2, lower bound: -1.1947646, upper bound: 1.1952574
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 42.48
Output dim: 2, lower bound: -1.2028920, upper bound: 1.1872657
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 42.48
Output dim: 2, lower bound: -1.1816618, upper bound: 1.2075675
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 42.48
Output dim: 2, lower bound: -1.1906038, upper bound: 1.1985809

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.4025526, 2.3958321
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.3567734, 2.3921218
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.9857671, 1.9857671
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -2.0270391, 2.0270391
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.6576705, 1.6970928
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.8469739, 1.8655753
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.2840786, 2.3014662
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.5651479, 1.5955111
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.2200575, 2.2178519
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.7814889, 1.7752042

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5818

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1905051, upper bound: 1.1906045
time: 5.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1995145, upper bound: 1.1816625
time: 4.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.4065847, 2.3917990
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.3675852, 2.3813100
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.9857671, 1.9857671
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -2.0270391, 2.0270391
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.6677065, 1.6870582
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.8397632, 1.8727860
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.2670074, 2.3185370
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.5809178, 1.5797409
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.2200575, 2.2200575
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.8018713, 1.7548213

Time for backsubstitution: 14.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5818

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1985812, upper bound: 1.1824579
time: 4.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2075681, upper bound: 1.1734148
time: 4.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.3970146, 2.3988380
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.3671446, 2.3731527
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.9857671, 1.9857671
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -2.0270391, 2.0270391
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.6522961, 1.7000420
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.8543324, 1.8521457
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.2932582, 2.2847121
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.5671697, 1.5917807
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.2200575, 2.2160864
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.7782454, 1.7769823

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5818

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1782364, upper bound: 1.2028925
time: 6.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1872654, upper bound: 1.1938977
time: 6.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.4010477, 2.3948050
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.3779564, 2.3623409
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.9857671, 1.9857671
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -2.0270391, 2.0270391
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.6623321, 1.6900070
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.8471217, 1.8593564
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.2761869, 2.3017838
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.5829396, 1.5760105
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.2200575, 2.2200575
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.7986283, 1.7565994

Time for backsubstitution: 14.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5818

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1863316, upper bound: 1.1947643
time: 6.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1952570, upper bound: 1.1857090
time: 5.45 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.3927288, 2.3998160
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.3613424, 2.3808308
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.9857671, 1.9857671
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -2.0270391, 2.0270391
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.6689498, 1.6692874
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.8508387, 1.8560493
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.2922463, 2.2813203
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.5697322, 1.5842307
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.2200575, 2.2200575
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.7502666, 1.7878428

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5818

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1849417, upper bound: 1.1960619
time: 4.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1939559, upper bound: 1.1871189
time: 5.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.3967619, 2.3957825
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.3721538, 2.3700190
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.9857671, 1.9857671
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -2.0270391, 2.0270391
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.6789858, 1.6592529
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.8436279, 1.8632603
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.2751746, 2.2983916
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.5855021, 1.5684605
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.2082186, 2.2200575
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.7706490, 1.7674599

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5818

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1930231, upper bound: 1.1880129
time: 10.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2020100, upper bound: 1.1789728
time: 6.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.3871918, 2.4028206
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.3717151, 2.3618617
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.9857671, 1.9857671
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -2.0270391, 2.0270391
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.6635754, 1.6722367
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.8581972, 1.8426197
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.3014250, 2.2645667
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.5717535, 1.5805002
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.2200575, 2.2200575
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.7470231, 1.7896214

Time for backsubstitution: 14.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5818

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1726822, upper bound: 1.2084496
time: 7.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1817249, upper bound: 1.1994612
time: 5.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.3912249, 2.3987870
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.3825264, 2.3510499
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.9857671, 1.9857671
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -2.0270391, 2.0270391
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.6736114, 1.6622016
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.8509865, 1.8498304
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.2843537, 2.2816384
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.5875239, 1.5647302
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.2091923, 2.2200575
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.7674060, 1.7692389

Time for backsubstitution: 14.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5818

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1808466, upper bound: 1.2003319
time: 7.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1897816, upper bound: 1.1912793
time: 5.27 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.4071608, 2.3912249
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.3510504, 2.3978453
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.9857671, 1.9857671
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -2.0270391, 2.0270391
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.6646304, 1.6901376
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.8498306, 1.8627191
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.2816381, 2.3039067
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.5647302, 1.5959284
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.2200575, 2.2099860
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.7892890, 1.7674060

Time for backsubstitution: 14.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5818

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1912792, upper bound: 1.1897817
time: 5.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2003306, upper bound: 1.1808470
time: 5.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.4111938, 2.3871918
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.3618617, 2.3870335
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.9857671, 1.9857671
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -2.0270391, 2.0270391
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.6746655, 1.6801012
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.8426199, 1.8699298
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.2645669, 2.3209779
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.5805001, 1.5801582
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.2200575, 2.2200575
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.8096714, 1.7470233

Time for backsubstitution: 14.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5818

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1994613, upper bound: 1.1817253
time: 5.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2084498, upper bound: 1.1726804
time: 4.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.4016218, 2.3942308
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.3614216, 2.3788762
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.9857671, 1.9857671
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -2.0270391, 2.0270391
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.6592526, 1.6930864
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.8571892, 1.8492894
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.2908177, 2.2871530
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.5667520, 1.5921980
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.2200575, 2.2082183
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.7860436, 1.7691841

Time for backsubstitution: 14.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5818

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1789708, upper bound: 1.2020102
time: 7.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1880125, upper bound: 1.1930251
time: 5.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.4056559, 2.3901978
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.3722329, 2.3680644
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.9857671, 1.9857671
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -2.0270391, 2.0270391
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.6692877, 1.6830504
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.8499780, 1.8564999
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.2737465, 2.3042243
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.5825229, 1.5764279
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.2200575, 2.2200575
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.8064265, 1.7488017

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5818

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1871185, upper bound: 1.1939557
time: 8.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1960617, upper bound: 1.1849421
time: 5.49 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.3973370, 2.3952088
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.3556190, 2.3865542
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.9857671, 1.9857671
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -2.0270391, 2.0270391
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.6759095, 1.6623323
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.8536949, 1.8531930
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.2898059, 2.2837613
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.5693145, 1.5846480
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.2200575, 2.2179687
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.7580667, 1.7800446

Time for backsubstitution: 14.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5818

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1857068, upper bound: 1.1952568
time: 4.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1947643, upper bound: 1.1863334
time: 5.19 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.4013700, 2.3911753
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.3664308, 2.3757424
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.9857671, 1.9857671
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -2.0270391, 2.0270391
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.6859446, 1.6522958
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.8464842, 1.8604035
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.2727342, 2.3008320
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.5850844, 1.5688779
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.2160864, 2.2200575
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.7784491, 1.7596622

Time for backsubstitution: 14.43 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=1.9854326248168945
rel_dist={2: [-1.2085888365066806, 1.2085886275699598]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 5733
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 510

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9476533, upper bound: 0.9484718
time: 4.83 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9476534, upper bound: 0.9476501
time: 5.98 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 11.01 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 11.01
Output dim: 2, lower bound: -0.9476533, upper bound: 0.9484718
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 11.01
Output dim: 2, lower bound: -0.9476534, upper bound: 0.9476501

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1621284, 2.1647611
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.2174315, 2.2141614
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8728600, 1.8721206
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.9188509, 1.9211485
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.5375671, 1.5415423
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7722759, 1.7739081
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1620064, 2.1606116
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4976697, 1.4974315
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9849253, 1.9894216
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5566354, 1.5610914

Time for backsubstitution: 14.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5733
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5733

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9476199, upper bound: 0.9428648
time: 5.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9420412, upper bound: 0.9484371
time: 5.44 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1647606, 2.1621284
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.2141614, 2.2174315
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8721209, 1.8728595
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.9211483, 1.9188509
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.5415423, 1.5375674
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7739081, 1.7722759
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1606121, 2.1620064
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4974313, 1.4976699
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9894218, 1.9849255
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5610914, 1.5566354

Time for backsubstitution: 14.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5733
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5733

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9484397, upper bound: 0.9420389
time: 5.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9420412, upper bound: 0.9476164
time: 7.80 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 27.92 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 27.92
Output dim: 2, lower bound: -0.9476199, upper bound: 0.9428648
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 27.92
Output dim: 2, lower bound: -0.9420412, upper bound: 0.9484371
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 27.92
Output dim: 2, lower bound: -0.9484397, upper bound: 0.9420389
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 27.92
Output dim: 2, lower bound: -0.9420412, upper bound: 0.9476164

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1637578, 2.1607776
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.2128625, 2.2160439
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8751879, 1.8664231
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.9169970, 1.9219303
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.5262873, 1.5461514
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7684116, 1.7754874
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1538401, 2.1639571
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4930863, 1.4992933
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9882059, 1.9814389
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5618386, 1.5484531

Time for backsubstitution: 14.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 4625

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9474641, upper bound: 0.9299393
time: 5.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9346974, upper bound: 0.9427096
time: 5.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1581445, 2.1647611
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.2174315, 2.2095919
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8671622, 1.8721206
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.9188509, 1.9192946
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.5375671, 1.5302627
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7722759, 1.7700438
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1620064, 2.1524453
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4976697, 1.4928474
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9769430, 1.9894216
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5439973, 1.5610914

Time for backsubstitution: 14.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 4625

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9418865, upper bound: 0.9355168
time: 7.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9291155, upper bound: 0.9482829
time: 5.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1663899, 2.1581450
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.2095919, 2.2193141
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8744493, 1.8671622
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.9192944, 1.9196327
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.5302625, 1.5421765
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7700438, 1.7738552
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1524453, 2.1653519
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4928474, 1.4995317
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9927015, 1.9769428
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5662942, 1.5439970

Time for backsubstitution: 14.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 4625

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9482856, upper bound: 0.9291132
time: 5.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9355180, upper bound: 0.9418841
time: 5.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1607776, 2.1621284
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.2141614, 2.2128620
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8664231, 1.8728595
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.9211483, 1.9169970
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.5415423, 1.5262873
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7739081, 1.7684116
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1606121, 2.1538401
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4974313, 1.4930859
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9814386, 1.9849255
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5484529, 1.5566354

Time for backsubstitution: 14.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 4625

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9418865, upper bound: 0.9346954
time: 7.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9299419, upper bound: 0.9474625
time: 5.06 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 26.77 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.77
Output dim: 2, lower bound: -0.9474641, upper bound: 0.9299393
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.77
Output dim: 2, lower bound: -0.9346974, upper bound: 0.9427096
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.77
Output dim: 2, lower bound: -0.9418865, upper bound: 0.9355168
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.77
Output dim: 2, lower bound: -0.9291155, upper bound: 0.9482829
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.77
Output dim: 2, lower bound: -0.9482856, upper bound: 0.9291132
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.77
Output dim: 2, lower bound: -0.9355180, upper bound: 0.9418841
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.77
Output dim: 2, lower bound: -0.9418865, upper bound: 0.9346954
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.77
Output dim: 2, lower bound: -0.9299419, upper bound: 0.9474625

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1639175, 2.1577725
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.2024908, 2.2165117
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8755770, 1.8570218
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.9172812, 1.9154680
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.5264087, 1.5432017
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7610526, 1.7758026
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1446609, 2.1643515
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4910641, 1.4994030
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9872322, 1.9814749
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5619128, 1.5466743

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 525

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9431048, upper bound: 0.9299366
time: 6.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9474619, upper bound: 0.9255663
time: 5.27 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1607533, 2.1607776
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.2128625, 2.2056723
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8657866, 1.8664231
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.9105349, 1.9219303
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.5233376, 1.5461514
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7684116, 1.7681284
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1538401, 2.1547780
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4930863, 1.4972711
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9882059, 1.9804659
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5600598, 1.5484531

Time for backsubstitution: 14.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 525

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9303043, upper bound: 0.9427082
time: 8.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9346951, upper bound: 0.9383679
time: 5.25 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1583042, 2.1617551
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.2070594, 2.2100596
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8675513, 1.8627205
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.9191351, 1.9128323
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.5376885, 1.5273130
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7649174, 1.7703590
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1528277, 2.1528401
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4956479, 1.4929569
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9759698, 1.9894576
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5440714, 1.5593128

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 525

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9375192, upper bound: 0.9355157
time: 5.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9375215, upper bound: 0.9311442
time: 4.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1551399, 2.1647611
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.2174315, 2.1992202
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8577609, 1.8721206
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.9123888, 1.9192946
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.5346174, 1.5302627
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7722759, 1.7626848
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1620064, 2.1432662
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4976697, 1.4908252
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9769430, 1.9884486
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5422184, 1.5610914

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 525

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9247230, upper bound: 0.9482811
time: 4.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9247230, upper bound: 0.9439488
time: 5.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1665506, 2.1551399
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1992202, 2.2197824
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8748379, 1.8577609
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.9195786, 1.9131703
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.5303857, 1.5392268
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7626848, 1.7741704
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1432667, 2.1657462
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4908252, 1.4996414
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9917283, 1.9769800
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5663702, 1.5422182

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 525

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9439497, upper bound: 0.9291106
time: 5.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9482833, upper bound: 0.9247209
time: 5.24 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1633863, 2.1581450
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.2095919, 2.2089424
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8650475, 1.8671622
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.9128323, 1.9196327
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.5273128, 1.5421765
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7700438, 1.7664962
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1524453, 2.1561728
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4928474, 1.4975097
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9927015, 1.9759698
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5645158, 1.5439970

Time for backsubstitution: 14.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 525

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9311458, upper bound: 0.9418808
time: 5.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9355161, upper bound: 0.9375188
time: 5.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1609373, 2.1591225
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.2037892, 2.2133303
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8668122, 1.8634596
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.9214325, 1.9105346
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.5416656, 1.5233376
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7665496, 1.7687268
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1514330, 2.1542344
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4954095, 1.4931955
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9804659, 1.9849627
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5485289, 1.5548568

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 525

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9383692, upper bound: 0.9346956
time: 5.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9427100, upper bound: 0.9303017
time: 5.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1577730, 2.1621284
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.2141614, 2.2024903
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8570218, 1.8728595
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.9146862, 1.9169970
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.5385926, 1.5262873
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7739081, 1.7610526
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1606121, 2.1446609
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4974313, 1.4910638
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9814386, 1.9839525
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5466745, 1.5566354

Time for backsubstitution: 14.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 525

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9247230, upper bound: 0.9474631
time: 6.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9247230, upper bound: 0.9431022
time: 6.55 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 27.99 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.99
Output dim: 2, lower bound: -0.9431048, upper bound: 0.9299366
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.99
Output dim: 2, lower bound: -0.9474619, upper bound: 0.9255663
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.99
Output dim: 2, lower bound: -0.9303043, upper bound: 0.9427082
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.99
Output dim: 2, lower bound: -0.9346951, upper bound: 0.9383679
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.99
Output dim: 2, lower bound: -0.9375192, upper bound: 0.9355157
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.99
Output dim: 2, lower bound: -0.9375215, upper bound: 0.9311442
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.99
Output dim: 2, lower bound: -0.9247230, upper bound: 0.9482811
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.99
Output dim: 2, lower bound: -0.9247230, upper bound: 0.9439488
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.99
Output dim: 2, lower bound: -0.9439497, upper bound: 0.9291106
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.99
Output dim: 2, lower bound: -0.9482833, upper bound: 0.9247209
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.99
Output dim: 2, lower bound: -0.9311458, upper bound: 0.9418808
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.99
Output dim: 2, lower bound: -0.9355161, upper bound: 0.9375188
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.99
Output dim: 2, lower bound: -0.9383692, upper bound: 0.9346956
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.99
Output dim: 2, lower bound: -0.9427100, upper bound: 0.9303017
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.99
Output dim: 2, lower bound: -0.9247230, upper bound: 0.9474631
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.99
Output dim: 2, lower bound: -0.9247230, upper bound: 0.9431022

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1776276, 2.1737876
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1655889, 2.1857877
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8919687, 1.8710508
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.9221425, 1.9211476
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4920106, 1.5145376
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7404623, 1.7510920
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.0959115, 2.1058476
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4370322, 1.4543824
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.0298285, 2.0171871
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5258007, 1.5222096

Time for backsubstitution: 14.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 5818

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9377470, upper bound: 0.9299369
time: 5.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9431020, upper bound: 0.9245771
time: 5.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1799326, 2.1714830
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1717672, 2.1796098
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8896060, 1.8734133
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.9229608, 1.9203296
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4977455, 1.5088031
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7363420, 1.7552123
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.0861564, 2.1156023
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4460435, 1.4453709
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.0229445, 2.0240707
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5374479, 1.5105624

Time for backsubstitution: 14.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5818

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9421013, upper bound: 0.9255659
time: 5.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9474615, upper bound: 0.9202050
time: 5.15 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1744633, 2.1767936
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1759601, 2.1749487
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8821778, 1.8804519
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.9153962, 1.9276099
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4889395, 1.5174868
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7478209, 1.7434177
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1050916, 2.0962737
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4390540, 1.4522507
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.0308018, 2.0161781
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5239472, 1.5239878

Time for backsubstitution: 14.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5818

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9249441, upper bound: 0.9427082
time: 5.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9303015, upper bound: 0.9373478
time: 5.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1767683, 2.1744890
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1821384, 2.1687703
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8798156, 1.8828144
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.9162145, 1.9267919
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4946744, 1.5117524
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7437005, 1.7475381
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.0953364, 2.1060288
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4480653, 1.4432392
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.0239182, 2.0230620
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5355945, 1.5123405

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 5818

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9293360, upper bound: 0.9383668
time: 5.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9346947, upper bound: 0.9330106
time: 5.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1720142, 2.1777716
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1701579, 2.1793361
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8839431, 1.8767493
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.9239969, 1.9185119
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.5032899, 1.4986489
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7443271, 1.7456484
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1040792, 2.0943358
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4416165, 1.4479365
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.0185661, 2.0251698
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5079594, 1.5348482

Time for backsubstitution: 14.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 5818

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9321634, upper bound: 0.9355138
time: 5.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9375188, upper bound: 0.9301548
time: 5.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1743193, 2.1754665
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1763358, 2.1731577
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8815804, 1.8791118
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.9248147, 1.9176939
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.5090249, 1.4929149
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7402067, 1.7497687
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.0943241, 2.1040905
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4506278, 1.4389249
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.0116820, 2.0320535
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5196066, 1.5232010

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5818

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9365235, upper bound: 0.9311447
time: 5.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9321635, upper bound: 0.9257879
time: 4.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1688499, 2.1807756
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1805305, 2.1684966
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8741522, 1.8861496
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.9172506, 1.9249744
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.5002189, 1.5015981
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7516856, 1.7379742
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1132579, 2.0847619
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4436378, 1.4458048
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.0195398, 2.0241609
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5061059, 1.5366268

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 5818

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9193655, upper bound: 0.9482805
time: 5.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9193655, upper bound: 0.9429202
time: 5.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1711550, 2.1784711
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1867085, 2.1623187
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8717899, 1.8885121
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.9180684, 1.9241562
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.5059538, 1.4958637
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7475653, 1.7420948
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1035028, 2.0945170
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4526496, 1.4367933
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.0126553, 2.0310447
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5177531, 1.5249796

Time for backsubstitution: 14.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 5818

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9193655, upper bound: 0.9439482
time: 6.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9291114, upper bound: 0.9385897
time: 5.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1802607, 2.1711550
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1623182, 2.1890588
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8912296, 1.8717899
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.9244404, 1.9188499
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4959877, 1.5105631
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7420950, 1.7494597
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.0945172, 2.1072419
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4367938, 1.4546208
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.0343246, 2.0126920
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5302577, 1.5177536

Time for backsubstitution: 14.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 5818

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9385908, upper bound: 0.9291104
time: 5.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9439483, upper bound: 0.9237534
time: 6.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1825657, 2.1688504
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1684966, 2.1828804
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8888669, 1.8741522
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.9252582, 1.9180319
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.5017219, 1.5048282
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7379742, 1.7535801
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.0847621, 2.1169970
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4458051, 1.4456093
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.0274405, 2.0195761
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5419049, 1.5061064

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 5818

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9429198, upper bound: 0.9247197
time: 6.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9482830, upper bound: 0.9193634
time: 5.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1770964, 2.1741610
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1726894, 2.1782188
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8814392, 1.8811910
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.9176936, 1.9253123
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4929147, 1.5135124
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7494531, 1.7417855
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1036963, 2.0976684
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4388156, 1.4524893
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.0352983, 2.0116820
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5284038, 1.5195317

Time for backsubstitution: 14.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 5818

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9257893, upper bound: 0.9418810
time: 5.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9311453, upper bound: 0.9365223
time: 5.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1794004, 2.1718564
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1788678, 2.1720409
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8790765, 1.8835533
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.9185119, 1.9244943
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4986491, 1.5077775
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7453327, 1.7459059
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.0939417, 2.1074235
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4478269, 1.4434777
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.0284138, 2.0185661
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5400510, 1.5078845

Time for backsubstitution: 14.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5818

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9301552, upper bound: 0.9375186
time: 5.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9355153, upper bound: 0.9321607
time: 5.26 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1746473, 2.1751385
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1668873, 2.1826062
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8832040, 1.8774884
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.9262943, 1.9162145
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.5072670, 1.4946744
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7459593, 1.7440162
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1026845, 2.0957301
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4413776, 1.4481750
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.0230622, 2.0206747
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5124164, 1.5303922

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5818

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9330101, upper bound: 0.9346936
time: 5.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9383676, upper bound: 0.9293356
time: 5.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1769524, 2.1728339
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1730657, 2.1764283
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8808413, 1.8798506
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.9271126, 1.9153962
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.5130014, 1.4889395
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7418389, 1.7481365
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.0929294, 2.1054852
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4503894, 1.4391634
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.0161781, 2.0275588
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5240636, 1.5187449

Time for backsubstitution: 14.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5818

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9373484, upper bound: 0.9303013
time: 6.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.9321635, upper bound: 0.9249429
time: 5.14 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1714830, 2.1781430
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1772599, 2.1717668
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8734136, 1.8868887
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.9195480, 1.9226766
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.5041940, 1.4976232
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7533178, 1.7363420
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1118636, 2.0861566
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4433994, 1.4460434
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.0240355, 2.0196648
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.5105624, 1.5321708

Time for backsubstitution: 14.45 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=1.8685765266418457
rel_dist={2: [-0.9484739266423019, 0.9484713204465995]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 510
type: RSZ, layer: 1, pos: 5733
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 510

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8354207, upper bound: 0.8355764
time: 5.64 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8354207, upper bound: 0.8354209
time: 6.39 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 12.23 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 12.23
Output dim: 2, lower bound: -0.8354207, upper bound: 0.8355764
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 12.23
Output dim: 2, lower bound: -0.8354207, upper bound: 0.8354209

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.0893474, 2.0913224
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1537037, 2.1512504
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8337231, 1.8331685
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.8724542, 1.8741775
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4831150, 1.4860966
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7378020, 1.7390263
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1017227, 2.1006765
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4549651, 1.4547858
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9194136, 1.9227855
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.4763293, 1.4796715

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5733
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5733

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8297869, upper bound: 0.8299522
time: 5.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8297869, upper bound: 0.8355433
time: 6.69 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.0913224, 2.0893478
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1512508, 2.1537032
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8331685, 1.8337226
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.8741775, 1.8724542
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4860964, 1.4831150
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7390261, 1.7378020
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1006770, 2.1017227
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4547858, 1.4549646
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9227858, 1.9194136
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.4796715, 1.4763298

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5733
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5733

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8297869, upper bound: 0.8297851
time: 5.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8297869, upper bound: 0.8353879
time: 8.44 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 28.97 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 28.97
Output dim: 2, lower bound: -0.8297869, upper bound: 0.8299522
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 28.97
Output dim: 2, lower bound: -0.8297869, upper bound: 0.8355433
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 28.97
Output dim: 2, lower bound: -0.8297869, upper bound: 0.8297851
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 28.97
Output dim: 2, lower bound: -0.8297869, upper bound: 0.8353879

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.0853643, 2.0913224
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1537037, 2.1466813
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8280253, 1.8331685
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.8724542, 1.8723235
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4831150, 1.4748166
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7378020, 1.7351620
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1017227, 2.0925102
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4549651, 1.4502017
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9114313, 1.9227855
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.4636912, 1.4796715

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 4625

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8200709, upper bound: 0.8258622
time: 7.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8200709, upper bound: 0.8355187
time: 5.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.0873389, 2.0893478
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1512508, 2.1491342
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8274713, 1.8337226
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.8741775, 1.8706002
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4860964, 1.4718354
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7390261, 1.7339377
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.1006770, 2.0935564
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4547858, 1.4503806
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9148026, 1.9194136
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.4670329, 1.4763298

Time for backsubstitution: 14.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4625
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 4625

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8297617, upper bound: 0.8256729
time: 7.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8200709, upper bound: 0.8353591
time: 6.61 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 28.45 seconds
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 28.45
Output dim: 2, lower bound: -0.8200709, upper bound: 0.8258622
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 28.45
Output dim: 2, lower bound: -0.8200709, upper bound: 0.8355187
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 28.45
Output dim: 2, lower bound: -0.8297617, upper bound: 0.8256729
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 28.45
Output dim: 2, lower bound: -0.8200709, upper bound: 0.8353591

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.0823593, 2.0906897
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1514611, 2.1363096
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8186240, 1.8311112
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.8659916, 1.8709209
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4801652, 1.4741719
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7361994, 1.7278032
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.0997243, 2.0833316
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4545417, 1.4481797
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9112153, 1.9218125
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.4619129, 1.4792838

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 525

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8170558, upper bound: 0.8355181
time: 7.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8170558, upper bound: 0.8325164
time: 8.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.0843339, 2.0887146
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1490083, 2.1387625
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8180699, 1.8316655
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.8677149, 1.8691976
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4831467, 1.4711888
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7374234, 1.7265790
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.0986786, 2.0843773
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4543624, 1.4483585
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9145865, 1.9184406
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.4652545, 1.4759412

Time for backsubstitution: 14.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 525

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8170558, upper bound: 0.8353603
time: 11.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8170558, upper bound: 0.8323481
time: 5.56 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 31.26 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 31.26
Output dim: 2, lower bound: -0.8170558, upper bound: 0.8355181
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 31.26
Output dim: 2, lower bound: -0.8170558, upper bound: 0.8325164
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 31.26
Output dim: 2, lower bound: -0.8170558, upper bound: 0.8353603
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 31.26
Output dim: 2, lower bound: -0.8170558, upper bound: 0.8323481

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.0960698, 2.1061301
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1145592, 2.1040416
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8344245, 1.8451402
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.8708534, 1.8763962
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4457667, 1.4440744
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7145786, 1.7030926
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.0485368, 2.0248272
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4005098, 1.4009063
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9520903, 1.9575248
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.4258003, 1.4519076

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5818

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8130263, upper bound: 0.8355185
time: 5.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8130263, upper bound: 0.8314623
time: 4.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.0977983, 2.1044016
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1191931, 2.0994077
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8326530, 1.8469117
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.8714671, 1.8757825
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4500680, 1.4397733
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7114887, 1.7061830
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.0412207, 2.0321434
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4072685, 1.3941478
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9469275, 1.9626877
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.4345360, 1.4431720

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 5818

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8130263, upper bound: 0.8325160
time: 5.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8130263, upper bound: 0.8284624
time: 6.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.0980444, 2.1041551
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1121063, 2.1064944
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8338704, 1.8456943
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.8725767, 1.8746729
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4487481, 1.4410918
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7158031, 1.7018683
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.0474911, 2.0258729
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4003310, 1.4010853
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9554615, 1.9541526
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.4291425, 1.4485645

Time for backsubstitution: 14.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5818

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8130261, upper bound: 0.8353591
time: 5.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8130263, upper bound: 0.8313275
time: 7.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.0997729, 2.1024265
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.1167402, 2.1018605
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8320990, 1.8474662
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.8731904, 1.8740592
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4530487, 1.4367907
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.7127128, 1.7049584
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -2.0401750, 2.0331895
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.4070897, 1.3943266
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9502988, 1.9593158
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.4378781, 1.4398289

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5818
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 5818

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8130263, upper bound: 0.8323447
time: 4.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8130263, upper bound: 0.8283114
time: 5.66 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 25.41 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.41
Output dim: 2, lower bound: -0.8130263, upper bound: 0.8355185
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 25.41
Output dim: 2, lower bound: -0.8130263, upper bound: 0.8314623
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.41
Output dim: 2, lower bound: -0.8130263, upper bound: 0.8325160
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 25.41
Output dim: 2, lower bound: -0.8130263, upper bound: 0.8284624
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.41
Output dim: 2, lower bound: -0.8130261, upper bound: 0.8353591
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 25.41
Output dim: 2, lower bound: -0.8130263, upper bound: 0.8313275
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 25.41
Output dim: 2, lower bound: -0.8130263, upper bound: 0.8323447
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 25.41
Output dim: 2, lower bound: -0.8130263, upper bound: 0.8283114

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.0999660, 2.1105356
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.0319109, 2.0317326
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8539448, 1.8624012
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.8579092, 1.8616071
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4221909, 1.4251952
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.6818538, 1.6656923
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -1.9975801, 1.9665852
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.3263781, 1.3360462
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.0041766, 2.0035992
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.4210558, 1.4568396

Time for backsubstitution: 14.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 4632

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8119460, upper bound: 0.8344393
time: 5.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8119460, upper bound: 0.8355170
time: 5.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1016946, 2.1088066
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.0365443, 2.0271001
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8521729, 1.8641729
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.8585224, 1.8609934
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4264920, 1.4208951
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.6787634, 1.6687827
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -1.9902654, 1.9739013
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.3331368, 1.3292892
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9990129, 2.0087619
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.4297910, 1.4481061

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 4632

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8119460, upper bound: 0.8314374
time: 5.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8119460, upper bound: 0.8325159
time: 7.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1019406, 2.1085606
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.0294580, 2.0341854
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8533907, 1.8629556
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.8596325, 1.8598838
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4251723, 1.4222131
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.6830783, 1.6644681
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -1.9965339, 1.9676309
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.3261993, 1.3362252
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.0075479, 2.0002275
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.4243975, 1.4534967

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4632
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 4632

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8131967, upper bound: 0.8342783
time: 5.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8119460, upper bound: 0.8353594
time: 6.30 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 26.65 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 26.65
Output dim: 2, lower bound: -0.8119460, upper bound: 0.8344393
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.65
Output dim: 2, lower bound: -0.8119460, upper bound: 0.8355170
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 26.65
Output dim: 2, lower bound: -0.8119460, upper bound: 0.8314374
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.65
Output dim: 2, lower bound: -0.8119460, upper bound: 0.8325159
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 26.65
Output dim: 2, lower bound: -0.8131967, upper bound: 0.8342783
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.65
Output dim: 2, lower bound: -0.8119460, upper bound: 0.8353594

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.0997529, 2.1102943
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.0300503, 2.0301518
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8530188, 1.8613434
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.8578978, 1.8615971
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4217322, 1.4247931
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.6821041, 1.6659985
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -1.9974723, 1.9665077
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.3230078, 1.3330964
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.0034618, 2.0027826
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.4174283, 1.4536641

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 6111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8056516, upper bound: 0.8281369
time: 5.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8056516, upper bound: 0.8344363
time: 6.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.0997248, 2.1103225
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.0303302, 2.0298719
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8528867, 1.8614755
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.8578992, 1.8615956
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4217896, 1.4247359
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.6821594, 1.6659431
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -1.9975028, 1.9664767
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.3234279, 1.3326763
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.0033598, 2.0028851
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.4178803, 1.4532118

Time for backsubstitution: 14.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 6111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8056516, upper bound: 0.8292141
time: 5.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8056516, upper bound: 0.8355142
time: 8.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1014533, 2.1085935
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.0349641, 2.0252390
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8511147, 1.8632469
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.8585124, 1.8609822
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4260910, 1.4204353
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.6790686, 1.6690335
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -1.9901876, 1.9737933
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.3301866, 1.3259193
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9981966, 2.0080478
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.4266160, 1.4444780

Time for backsubstitution: 14.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 6111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8056516, upper bound: 0.8262133
time: 6.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8056516, upper bound: 0.8325113
time: 6.20 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1017275, 2.1083193
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.0275974, 2.0326047
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8524647, 1.8618975
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.8596210, 1.8598738
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4247136, 1.4218105
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.6833282, 1.6647742
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -1.9964261, 1.9675539
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.3228290, 1.3332753
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.0068331, 1.9994104
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.4207699, 1.4503214

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 6111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8056516, upper bound: 0.8279774
time: 6.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8068931, upper bound: 0.8342738
time: 6.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1016994, 2.1083474
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.0278773, 2.0323248
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8523326, 1.8620296
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.8596225, 1.8598726
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4247711, 1.4217533
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.6833835, 1.6647189
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -1.9964566, 1.9675229
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.3232491, 1.3328552
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.0067310, 1.9995129
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.4212229, 1.4498689

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6111
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 6111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8056514, upper bound: 0.8290536
time: 6.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8058175, upper bound: 0.8353510
time: 5.02 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 25.63 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 25.63
Output dim: 2, lower bound: -0.8056516, upper bound: 0.8281369
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 25.63
Output dim: 2, lower bound: -0.8056516, upper bound: 0.8344363
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 25.63
Output dim: 2, lower bound: -0.8056516, upper bound: 0.8292141
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 25.63
Output dim: 2, lower bound: -0.8056516, upper bound: 0.8355142
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 25.63
Output dim: 2, lower bound: -0.8056516, upper bound: 0.8262133
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 25.63
Output dim: 2, lower bound: -0.8056516, upper bound: 0.8325113
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 25.63
Output dim: 2, lower bound: -0.8056516, upper bound: 0.8279774
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 25.63
Output dim: 2, lower bound: -0.8068931, upper bound: 0.8342738
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 25.63
Output dim: 2, lower bound: -0.8056514, upper bound: 0.8290536
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 25.63
Output dim: 2, lower bound: -0.8058175, upper bound: 0.8353510

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1040583, 2.1140971
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.0415678, 2.0403299
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8115606, 1.8253713
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.8760338, 1.8776243
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4089606, 1.4136907
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.6769595, 1.6589477
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -1.9988189, 1.9676969
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.3120298, 1.3166163
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9999995, 1.9985199
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.3571124, 1.3847542

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 904

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8064760, upper bound: 0.8344299
time: 5.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8054018, upper bound: 0.8301702
time: 5.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1040297, 2.1141253
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.0418482, 2.0400496
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8114285, 1.8255033
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.8760357, 1.8776231
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4090178, 1.4136335
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.6770148, 1.6588924
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -1.9988494, 1.9676659
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.3124499, 1.3161962
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9998975, 1.9986219
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.3575649, 1.3843017

Time for backsubstitution: 14.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 904

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8054021, upper bound: 0.8355084
time: 7.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8054018, upper bound: 0.8312439
time: 6.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1057587, 2.1123962
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.0464816, 2.0354171
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8096566, 1.8272750
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.8766489, 1.8770094
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4133191, 1.4093329
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.6739244, 1.6619828
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -1.9915347, 1.9749820
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.3192086, 1.3094392
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9947338, 2.0037851
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.3663006, 1.3755682

Time for backsubstitution: 14.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 904

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8084016, upper bound: 0.8325045
time: 9.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8054018, upper bound: 0.8282415
time: 5.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1060324, 2.1121221
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.0391150, 2.0427828
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8110065, 1.8259256
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.8777571, 1.8759012
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4119420, 1.4107085
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.6781840, 1.6577234
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -1.9977727, 1.9687426
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.3118510, 1.3167952
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.0033708, 1.9951477
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.3604546, 1.3814113

Time for backsubstitution: 14.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 904

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8066576, upper bound: 0.8342736
time: 5.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8054018, upper bound: 0.8300155
time: 8.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1060042, 2.1121502
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.0393949, 2.0425024
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8108745, 1.8260577
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.8777585, 1.8758998
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4119995, 1.4106508
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.6782393, 1.6576681
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -1.9978037, 1.9687116
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.3122711, 1.3163751
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.0032682, 1.9952497
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.3609071, 1.3809588

Time for backsubstitution: 15.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 904
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 904

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8055820, upper bound: 0.8353484
time: 6.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8055818, upper bound: 0.8310887
time: 6.76 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 28.76 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 28.76
Output dim: 2, lower bound: -0.8064760, upper bound: 0.8344299
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 28.76
Output dim: 2, lower bound: -0.8054018, upper bound: 0.8301702
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 28.76
Output dim: 2, lower bound: -0.8054021, upper bound: 0.8355084
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 28.76
Output dim: 2, lower bound: -0.8054018, upper bound: 0.8312439
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 28.76
Output dim: 2, lower bound: -0.8084016, upper bound: 0.8325045
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 28.76
Output dim: 2, lower bound: -0.8054018, upper bound: 0.8282415
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 28.76
Output dim: 2, lower bound: -0.8066576, upper bound: 0.8342736
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 28.76
Output dim: 2, lower bound: -0.8054018, upper bound: 0.8300155
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 28.76
Output dim: 2, lower bound: -0.8055820, upper bound: 0.8353484
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 28.76
Output dim: 2, lower bound: -0.8055818, upper bound: 0.8310887

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1082754, 2.1190329
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.0222192, 2.0233994
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8165083, 1.8295999
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.8799129, 1.8821635
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4084053, 1.4130580
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.6731365, 1.6545811
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -1.9865723, 1.9537053
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.3099174, 1.3147670
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9999466, 1.9984727
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.3631711, 1.3918419

Time for backsubstitution: 14.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 5841

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8025009, upper bound: 0.8344302
time: 6.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8064753, upper bound: 0.8304712
time: 6.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1082473, 2.1190610
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.0224991, 2.0231190
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8163762, 1.8297319
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.8799138, 1.8821621
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4084628, 1.4130008
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.6731918, 1.6545258
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -1.9866028, 1.9536743
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.3103375, 1.3143469
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9998441, 1.9985752
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.3636241, 1.3913894

Time for backsubstitution: 15.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 5841

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8014247, upper bound: 0.8355067
time: 6.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8053990, upper bound: 0.8315463
time: 7.46 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1099753, 2.1173320
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.0271330, 2.0184865
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8146043, 1.8315036
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.8805275, 1.8815484
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4127629, 1.4087002
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.6701014, 1.6576161
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -1.9792881, 1.9609904
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.3170962, 1.3075900
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -1.9946804, 2.0037384
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.3723588, 1.3826561

Time for backsubstitution: 14.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5841
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 5841

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.8044263, upper bound: 0.8325038
time: 5.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.8084009, upper bound: 0.8285453
time: 9.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -10.8108559, -8.2310667, -10.8108559, -8.2310667, -2.1102500, 2.1170578
1: -3.3504086, -0.9043145, -3.3504086, -0.9043145, -2.0197663, 2.0258522
2: 1.6547616, 3.6405287, 1.6547616, 3.6405287, -1.8159542, 1.8301542
3: -7.2597189, -5.2326798, -7.2597189, -5.2326798, -1.8816361, 1.8804402
4: -2.3514392, -0.4242229, -2.3514392, -0.4242229, -1.4113867, 1.4100754
5: -4.6279707, -2.7532918, -4.6279707, -2.7532918, -1.6743605, 1.6533568
6: -4.7157536, -2.2042937, -4.7157536, -2.2042937, -1.9855261, 1.9547510
7: -8.7202091, -6.8711286, -8.7202091, -6.8711286, -1.3097386, 1.3149459
8: -4.6486425, -2.4285851, -4.6486425, -2.4285851, -2.0033174, 1.9951010
9: -12.0749950, -9.7447462, -12.0749950, -9.7447462, -1.3665128, 1.3884990

Time for backsubstitution: 15.05 seconds
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=1.8296241760253906
rel_dist={2: [-0.835577596891917, 0.8355774930246449]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2418.81 seconds
