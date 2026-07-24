## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 0.78041487804
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.8020515, 2.8020515)
1: (-9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142)
2: (-9.9503212, -6.9502754, -9.9503212, -6.9502754, -3.0000458, 3.0000458)
3: (-10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.5673351, 2.5673351)
4: (-5.5582318, -3.5118723, -5.5582318, -3.5118723, -2.0463595, 2.0463595)
5: (-8.8875761, -6.1918221, -8.8875761, -6.1918221, -2.4845166, 2.4845166)
6: (-12.9723425, -9.7499943, -12.9723425, -9.7499943, -3.1591969, 3.1591969)
7: (0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.4368451, 2.4368451)
8: (-3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.7339888, 2.7339888)
9: (0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423)

## BASE Result
execution time: IAR + LP analysis = 15.10 + 32.19 = 47.29 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3552.71 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.2286510467529297
rel_dist={7: [-1.3704629636566878, 1.370462874585793]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=2.0185980796813965
rel_dist={7: [-1.0634246550232, 1.0634268200891506]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=1.8785624504089355
rel_dist={7: [-0.7927072900922123, 0.7927031607168487]}

## Binary Search Result
Binary search time: 147.82 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 3404.89 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6181

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6181

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3607291, upper bound: 1.3704543
time: 3.87 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3704541, upper bound: 1.3607290
time: 3.93 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 7.96 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 7.96
Output dim: 7, lower bound: -1.3607291, upper bound: 1.3704543
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 7.96
Output dim: 7, lower bound: -1.3704541, upper bound: 1.3607290

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.3100605, 2.3221960
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6256409, 2.6177151
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1944685, 2.1932862
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8637290, 1.8583035
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8905749, 1.8949389
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4407959, 2.4394846
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.1946836, 2.2031641
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.6133065, 2.6013312
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3382187, upper bound: 1.3433985
time: 3.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3336696, upper bound: 1.3479615
time: 3.93 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.3221960, 2.3100607
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6177149, 2.6256411
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1932864, 2.1944683
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8583035, 1.8637285
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8949389, 1.8905749
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4394846, 2.4407959
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.2031641, 2.1946836
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.6013312, 2.6133060
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3479616, upper bound: 1.3336696
time: 3.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3433977, upper bound: 1.3382194
time: 3.65 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 22.02 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 22.02
Output dim: 7, lower bound: -1.3382187, upper bound: 1.3433985
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 22.02
Output dim: 7, lower bound: -1.3336696, upper bound: 1.3479615
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 22.02
Output dim: 7, lower bound: -1.3479616, upper bound: 1.3336696
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 22.02
Output dim: 7, lower bound: -1.3433977, upper bound: 1.3382194

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.2707825, 2.2837164
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.9653521, 2.9826980
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6175318, 2.6044061
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.0885639, 2.0851314
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8604550, 1.8736410
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8929105, 1.8949633
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4088869, 2.4196892
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.0771494, 2.0958269
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.5377617, 2.5007648
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 14.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 1929

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3282508, upper bound: 1.3330008
time: 3.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3279262, upper bound: 1.3333228
time: 4.27 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.2715812, 2.2829175
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.9916849, 2.9563689
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6123323, 2.6096056
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.0863142, 2.0873811
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8790660, 1.8550301
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8905988, 1.8972747
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4210024, 2.4075756
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.0873461, 2.0856359
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.5127411, 2.5257869
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 1929

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3235836, upper bound: 1.3376591
time: 3.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3232606, upper bound: 1.3379878
time: 4.23 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.2829175, 2.2715812
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.9563684, 2.9916844
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6096058, 2.6123323
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.0873814, 2.0863137
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8550305, 1.8790665
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8972745, 1.8905988
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4075756, 2.4210024
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.0856357, 2.0873466
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.5257874, 2.5127411
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 14.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1929

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3379876, upper bound: 1.3232606
time: 3.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3376588, upper bound: 1.3235838
time: 3.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.2837162, 2.2707825
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.9826975, 2.9653516
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6044064, 2.6175315
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.0851316, 2.0885634
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8736415, 1.8604555
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8949633, 1.8929105
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4196892, 2.4088869
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.0958271, 2.0771494
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.5007648, 2.5377617
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 14.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1929

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3333230, upper bound: 1.3279264
time: 3.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3330009, upper bound: 1.3282510
time: 3.66 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 21.66 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.66
Output dim: 7, lower bound: -1.3282508, upper bound: 1.3330008
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.66
Output dim: 7, lower bound: -1.3279262, upper bound: 1.3333228
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.66
Output dim: 7, lower bound: -1.3235836, upper bound: 1.3376591
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.66
Output dim: 7, lower bound: -1.3232606, upper bound: 1.3379878
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.66
Output dim: 7, lower bound: -1.3379876, upper bound: 1.3232606
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.66
Output dim: 7, lower bound: -1.3376588, upper bound: 1.3235838
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.66
Output dim: 7, lower bound: -1.3333230, upper bound: 1.3279264
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.66
Output dim: 7, lower bound: -1.3330009, upper bound: 1.3282510

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.3057041, 2.3214817
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6252093, 2.6179202
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1887345, 2.1840246
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8612661, 1.8568487
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8866310, 1.8815620
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4400477, 2.4385185
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.1936097, 2.2010899
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.6105018, 2.5896773
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 14.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1206

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2923475, upper bound: 1.2989743
time: 3.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2919724, upper bound: 1.2990541
time: 3.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.3100605, 2.3178396
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6256409, 2.6172833
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1944685, 2.1875520
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8622742, 1.8583035
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8905749, 1.8909953
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4398298, 2.4394846
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.1926093, 2.2031641
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.6133065, 2.5985274
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1206

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2919960, upper bound: 1.2993226
time: 4.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2916220, upper bound: 1.2994018
time: 4.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.3057041, 2.3214817
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6252093, 2.6179202
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1887345, 2.1840246
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8612661, 1.8568487
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8866310, 1.8815620
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4400477, 2.4385185
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.1936097, 2.2010899
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.6105018, 2.5896773
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 14.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1206

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2900269, upper bound: 1.3010822
time: 3.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2899488, upper bound: 1.3014548
time: 4.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.3100605, 2.3178396
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6256409, 2.6172833
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1944685, 2.1875520
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8622742, 1.8583035
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8905749, 1.8909953
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4398298, 2.4394846
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.1926093, 2.2031641
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.6133065, 2.5985274
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1206

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2896592, upper bound: 1.3014273
time: 3.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2895847, upper bound: 1.3018007
time: 4.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.3178396, 2.3093467
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6172833, 2.6258461
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1875520, 2.1852067
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8558407, 1.8622742
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8909950, 1.8771975
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4387364, 2.4398298
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.2020888, 2.1926098
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.5985274, 2.6016512
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 14.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1206

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3018010, upper bound: 1.2895856
time: 3.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3014271, upper bound: 1.2896600
time: 3.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.3221960, 2.3057044
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6177149, 2.6252093
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1932864, 2.1887341
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8568487, 1.8637285
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8949389, 1.8866308
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4385185, 2.4407959
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.2010899, 2.1946836
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.6013312, 2.6105018
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 14.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1206

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3014552, upper bound: 1.2899494
time: 3.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3010821, upper bound: 1.2900277
time: 3.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.3178396, 2.3093467
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6172833, 2.6258461
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1875520, 2.1852067
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8558407, 1.8622742
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8909950, 1.8771975
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4387364, 2.4398298
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.2020888, 2.1926098
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.5985274, 2.6016512
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 14.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1206

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2994012, upper bound: 1.2916228
time: 3.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2993218, upper bound: 1.2919967
time: 3.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.3221960, 2.3057044
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6177149, 2.6252093
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1932864, 2.1887341
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8568487, 1.8637285
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8949389, 1.8866308
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4385185, 2.4407959
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.2010899, 2.1946836
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.6013312, 2.6105018
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1206

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2990534, upper bound: 1.2919727
time: 4.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2989741, upper bound: 1.2923483
time: 3.96 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 22.69 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.69
Output dim: 7, lower bound: -1.2923475, upper bound: 1.2989743
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.69
Output dim: 7, lower bound: -1.2919724, upper bound: 1.2990541
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.69
Output dim: 7, lower bound: -1.2919960, upper bound: 1.2993226
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.69
Output dim: 7, lower bound: -1.2916220, upper bound: 1.2994018
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.69
Output dim: 7, lower bound: -1.2900269, upper bound: 1.3010822
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.69
Output dim: 7, lower bound: -1.2899488, upper bound: 1.3014548
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.69
Output dim: 7, lower bound: -1.2896592, upper bound: 1.3014273
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.69
Output dim: 7, lower bound: -1.2895847, upper bound: 1.3018007
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.69
Output dim: 7, lower bound: -1.3018010, upper bound: 1.2895856
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.69
Output dim: 7, lower bound: -1.3014271, upper bound: 1.2896600
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.69
Output dim: 7, lower bound: -1.3014552, upper bound: 1.2899494
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.69
Output dim: 7, lower bound: -1.3010821, upper bound: 1.2900277
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.69
Output dim: 7, lower bound: -1.2994012, upper bound: 1.2916228
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.69
Output dim: 7, lower bound: -1.2993218, upper bound: 1.2919967
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.69
Output dim: 7, lower bound: -1.2990534, upper bound: 1.2919727
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.69
Output dim: 7, lower bound: -1.2989741, upper bound: 1.2923483

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.3099966, 2.3221469
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6239386, 2.6188059
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1938272, 2.1954479
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8624511, 1.8632402
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8905120, 1.8947849
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4392214, 2.4390068
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.1909728, 2.2017546
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.6100631, 2.6000400
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1929

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2873598, upper bound: 1.2824521
time: 3.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2758277, upper bound: 1.2938749
time: 3.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.3100114, 2.3221960
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6256409, 2.6160123
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1944685, 2.1926448
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8637290, 1.8570256
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8904209, 1.8949389
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4403181, 2.4394846
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.1932740, 2.2031641
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.6120143, 2.6013312
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 14.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1929

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2869902, upper bound: 1.2825293
time: 3.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2754537, upper bound: 1.2938886
time: 4.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.3099966, 2.3221469
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6239386, 2.6188059
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1938272, 2.1954479
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8624511, 1.8632402
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8905120, 1.8947849
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4392214, 2.4390068
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.1909728, 2.2017546
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.6100631, 2.6000400
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1929

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2869866, upper bound: 1.2828000
time: 3.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2754712, upper bound: 1.2942464
time: 4.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.3100114, 2.3221960
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6256409, 2.6160123
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1944685, 2.1926448
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8637290, 1.8570256
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8904209, 1.8949389
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4403181, 2.4394846
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.1932740, 2.2031641
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.6120143, 2.6013312
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1929

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2866165, upper bound: 1.2828793
time: 3.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2750972, upper bound: 1.2942649
time: 4.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.3099966, 2.3221469
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6239386, 2.6188059
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1938272, 2.1954479
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8624511, 1.8632402
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8905120, 1.8947849
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4392214, 2.4390068
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.1909728, 2.2017546
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.6100631, 2.6000400
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 14.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1929

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2849139, upper bound: 1.2845601
time: 3.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2735002, upper bound: 1.2959680
time: 4.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.3100114, 2.3221960
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6256409, 2.6160123
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1944685, 2.1926448
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8637290, 1.8570256
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8904209, 1.8949389
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4403181, 2.4394846
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.1932740, 2.2031641
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.6120143, 2.6013312
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 14.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1929

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2848974, upper bound: 1.2849328
time: 3.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2734234, upper bound: 1.2963380
time: 4.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.3099966, 2.3221469
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6239386, 2.6188059
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1938272, 2.1954479
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8624511, 1.8632402
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8905120, 1.8947849
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4392214, 2.4390068
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.1909728, 2.2017546
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.6100631, 2.6000400
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 14.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1929

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2845372, upper bound: 1.2849061
time: 3.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2731325, upper bound: 1.2963388
time: 4.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.3100114, 2.3221960
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6256409, 2.6160123
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1944685, 2.1926448
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8637290, 1.8570256
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8904209, 1.8949389
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4403181, 2.4394846
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.1932740, 2.2031641
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.6120143, 2.6013312
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 14.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1929

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2845255, upper bound: 1.2852790
time: 4.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2730580, upper bound: 1.2967088
time: 4.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.3221316, 2.3100116
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6160126, 2.6267319
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1926451, 2.1966302
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8570256, 1.8686652
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8948760, 1.8904209
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4379101, 2.4403181
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.1994538, 2.1932740
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.5980897, 2.6120143
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1929

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2967092, upper bound: 1.2730589
time: 3.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2852792, upper bound: 1.2845263
time: 3.47 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.3221469, 2.3100607
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6177149, 2.6239386
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1932864, 2.1938269
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8583035, 1.8624506
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8947849, 1.8905749
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4390068, 2.4407959
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.2017546, 2.1946836
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.6000400, 2.6133060
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 14.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1929

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2963391, upper bound: 1.2731334
time: 3.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2849054, upper bound: 1.2845379
time: 3.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.3221316, 2.3100116
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6160126, 2.6267319
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1926451, 2.1966302
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8570256, 1.8686652
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8948760, 1.8904209
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4379101, 2.4403181
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.1994538, 2.1932740
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.5980897, 2.6120143
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 14.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1929

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2963381, upper bound: 1.2734242
time: 3.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2849325, upper bound: 1.2848982
time: 3.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.3221469, 2.3100607
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6177149, 2.6239386
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1932864, 2.1938269
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8583035, 1.8624506
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8947849, 1.8905749
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4390068, 2.4407959
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.2017546, 2.1946836
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.6000400, 2.6133060
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 14.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1929

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2959678, upper bound: 1.2735011
time: 3.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2845594, upper bound: 1.2849146
time: 3.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.3221316, 2.3100116
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6160126, 2.6267319
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1926451, 2.1966302
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8570256, 1.8686652
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8948760, 1.8904209
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4379101, 2.4403181
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.1994538, 2.1932740
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.5980897, 2.6120143
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1929

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2942642, upper bound: 1.2750979
time: 3.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2828785, upper bound: 1.2866171
time: 3.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.3221469, 2.3100607
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6177149, 2.6239386
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1932864, 2.1938269
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8583035, 1.8624506
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8947849, 1.8905749
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4390068, 2.4407959
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.2017546, 2.1946836
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.6000400, 2.6133060
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 14.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1929

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2942456, upper bound: 1.2754714
time: 3.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2827992, upper bound: 1.2869873
time: 3.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.3221316, 2.3100116
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6160126, 2.6267319
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1926451, 2.1966302
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8570256, 1.8686652
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8948760, 1.8904209
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4379101, 2.4403181
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.1994538, 2.1932740
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.5980897, 2.6120143
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1929

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2938878, upper bound: 1.2754545
time: 3.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2825286, upper bound: 1.2869903
time: 4.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.3221469, 2.3100607
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6177149, 2.6239386
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1932864, 2.1938269
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8583035, 1.8624506
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8947849, 1.8905749
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4390068, 2.4407959
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.2017546, 2.1946836
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.6000400, 2.6133060
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1929

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2938741, upper bound: 1.2758279
time: 4.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2824515, upper bound: 1.2873605
time: 4.18 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 23.06 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.06
Output dim: 7, lower bound: -1.2873598, upper bound: 1.2824521
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.06
Output dim: 7, lower bound: -1.2758277, upper bound: 1.2938749
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.06
Output dim: 7, lower bound: -1.2869902, upper bound: 1.2825293
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.06
Output dim: 7, lower bound: -1.2754537, upper bound: 1.2938886
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.06
Output dim: 7, lower bound: -1.2869866, upper bound: 1.2828000
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.06
Output dim: 7, lower bound: -1.2754712, upper bound: 1.2942464
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.06
Output dim: 7, lower bound: -1.2866165, upper bound: 1.2828793
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.06
Output dim: 7, lower bound: -1.2750972, upper bound: 1.2942649
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.06
Output dim: 7, lower bound: -1.2849139, upper bound: 1.2845601
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.06
Output dim: 7, lower bound: -1.2735002, upper bound: 1.2959680
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.06
Output dim: 7, lower bound: -1.2848974, upper bound: 1.2849328
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.06
Output dim: 7, lower bound: -1.2734234, upper bound: 1.2963380
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.06
Output dim: 7, lower bound: -1.2845372, upper bound: 1.2849061
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.06
Output dim: 7, lower bound: -1.2731325, upper bound: 1.2963388
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.06
Output dim: 7, lower bound: -1.2845255, upper bound: 1.2852790
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.06
Output dim: 7, lower bound: -1.2730580, upper bound: 1.2967088
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.06
Output dim: 7, lower bound: -1.2967092, upper bound: 1.2730589
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.06
Output dim: 7, lower bound: -1.2852792, upper bound: 1.2845263
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.06
Output dim: 7, lower bound: -1.2963391, upper bound: 1.2731334
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.06
Output dim: 7, lower bound: -1.2849054, upper bound: 1.2845379
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.06
Output dim: 7, lower bound: -1.2963381, upper bound: 1.2734242
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.06
Output dim: 7, lower bound: -1.2849325, upper bound: 1.2848982
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.06
Output dim: 7, lower bound: -1.2959678, upper bound: 1.2735011
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.06
Output dim: 7, lower bound: -1.2845594, upper bound: 1.2849146
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.06
Output dim: 7, lower bound: -1.2942642, upper bound: 1.2750979
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.06
Output dim: 7, lower bound: -1.2828785, upper bound: 1.2866171
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.06
Output dim: 7, lower bound: -1.2942456, upper bound: 1.2754714
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.06
Output dim: 7, lower bound: -1.2827992, upper bound: 1.2869873
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.06
Output dim: 7, lower bound: -1.2938878, upper bound: 1.2754545
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.06
Output dim: 7, lower bound: -1.2825286, upper bound: 1.2869903
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.06
Output dim: 7, lower bound: -1.2938741, upper bound: 1.2758279
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.06
Output dim: 7, lower bound: -1.2824515, upper bound: 1.2873605

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.3104386, 2.2949576
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6263108, 2.6106758
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1695313, 2.1649661
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.7652159, 1.7282877
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8703160, 1.8722885
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4411488, 2.4338572
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.1522884, 2.1585865
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.5705829, 2.5518272
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1929

### Candidate
type: RSZ, layer: 3, pos: 1206

### Candidate
type: RSZ, layer: 3, pos: 312

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2822520, upper bound: 1.2814399
time: 3.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2863096, upper bound: 1.2767649
time: 3.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.2828221, 2.3225741
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6186018, 2.6183846
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1661487, 2.1683493
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.7337132, 1.7597907
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8679242, 1.8746803
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4351683, 2.4398372
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.1501060, 2.1607687
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.5638022, 2.5586078
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1929

### Candidate
type: RSZ, layer: 3, pos: 1206

### Candidate
type: RSZ, layer: 3, pos: 312

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2707644, upper bound: 1.2928554
time: 4.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2747855, upper bound: 1.2881663
time: 4.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.3104386, 2.2949576
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6263108, 2.6106758
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1695313, 2.1649661
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.7652159, 1.7282877
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8703160, 1.8722885
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4411488, 2.4338572
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.1522884, 2.1585865
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.5705829, 2.5518272
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1929

### Candidate
type: RSZ, layer: 3, pos: 1206

### Candidate
type: RSZ, layer: 3, pos: 312

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2818832, upper bound: 1.2815139
time: 3.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2859401, upper bound: 1.2768200
time: 3.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.2828221, 2.3225741
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6186018, 2.6183846
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1661487, 2.1683493
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.7337132, 1.7597907
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8679242, 1.8746803
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4351683, 2.4398372
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.1501060, 2.1607687
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.5638022, 2.5586078
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 14.41 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.2286510467529297
rel_dist={7: [-1.3704629636566878, 1.370462874585793]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6181

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6181

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0582297, upper bound: 1.0634195
time: 3.89 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0634198, upper bound: 1.0582292
time: 3.84 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 7.89 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 7.89
Output dim: 7, lower bound: -1.0582297, upper bound: 1.0634195
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 7.89
Output dim: 7, lower bound: -1.0634198, upper bound: 1.0582292

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9732170, 1.9792848
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.8134956, 2.8090043
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3608928, 2.3569298
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.9643664, 1.9637749
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.6314807, 1.6287680
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.6023397, 1.6045220
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0790238, 2.0783682
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9846301, 1.9888706
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.3306298, 2.3246417
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0789800, 2.0793948

Time for backsubstitution: 14.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0408109, upper bound: 1.0442964
time: 3.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0391182, upper bound: 1.0459915
time: 3.66 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9792848, 1.9732170
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.8090048, 2.8134961
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3569298, 2.3608928
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.9637752, 1.9643662
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.6287675, 1.6314802
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.6045222, 1.6023400
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0783682, 2.0790238
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9888706, 1.9846306
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.3246427, 2.3306293
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0793943, 2.0789800

Time for backsubstitution: 14.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.28 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0459899, upper bound: 1.0391198
time: 3.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0442948, upper bound: 1.0408125
time: 3.60 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 22.08 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 22.08
Output dim: 7, lower bound: -1.0408109, upper bound: 1.0442964
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 22.08
Output dim: 7, lower bound: -1.0391182, upper bound: 1.0459915
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 22.08
Output dim: 7, lower bound: -1.0459899, upper bound: 1.0391198
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 22.08
Output dim: 7, lower bound: -1.0442948, upper bound: 1.0408125

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9339390, 1.9404058
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.6713877, 2.6800599
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3501835, 2.3436208
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.8573365, 1.8556201
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.6282067, 1.6348000
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.6035199, 1.6045461
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0471144, 2.0525157
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.8721948, 1.8815336
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.2550850, 2.2365866
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0196118, 2.0249338

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1929

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0344879, upper bound: 1.0372828
time: 3.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0337968, upper bound: 1.0379733
time: 3.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9343381, 1.9400065
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.6845541, 2.6668949
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3475838, 2.3462205
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.8562117, 1.8567450
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.6375127, 1.6254950
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.6023641, 1.6057019
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0531721, 2.0464587
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.8772931, 1.8764381
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.2425747, 2.2490978
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0245190, 2.0200267

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1929

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0327952, upper bound: 1.0389778
time: 3.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0321040, upper bound: 1.0396683
time: 3.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9400063, 1.9343383
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.6668959, 2.6845531
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3462205, 2.3475838
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.8567452, 1.8562114
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.6254945, 1.6375127
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.6057019, 1.6023641
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0464587, 2.0531721
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.8764381, 1.8772933
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.2490978, 2.2425747
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0200267, 2.0245194

Time for backsubstitution: 14.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1929

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0396667, upper bound: 1.0321056
time: 3.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0389762, upper bound: 1.0327968
time: 3.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9404058, 1.9339387
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.6800604, 2.6713867
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3436208, 2.3501835
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.8556204, 1.8573363
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.6348004, 1.6282072
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.6045461, 1.6035199
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0525155, 2.0471146
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.8815336, 1.8721948
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.2365875, 2.2550850
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0249338, 2.0196118

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 1929

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0379716, upper bound: 1.0337971
time: 3.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0372811, upper bound: 1.0344895
time: 3.91 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 22.37 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.37
Output dim: 7, lower bound: -1.0344879, upper bound: 1.0372828
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.37
Output dim: 7, lower bound: -1.0337968, upper bound: 1.0379733
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.37
Output dim: 7, lower bound: -1.0327952, upper bound: 1.0389778
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.37
Output dim: 7, lower bound: -1.0321040, upper bound: 1.0396683
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.37
Output dim: 7, lower bound: -1.0396667, upper bound: 1.0321056
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.37
Output dim: 7, lower bound: -1.0389762, upper bound: 1.0327968
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.37
Output dim: 7, lower bound: -1.0379716, upper bound: 1.0337971
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.37
Output dim: 7, lower bound: -1.0372811, upper bound: 1.0344895

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9688606, 1.9767494
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.8078508, 2.8035955
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3604612, 2.3568163
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.9586320, 1.9562771
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.6295223, 1.6273136
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.5983958, 1.5958614
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0781665, 2.0774016
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9830565, 1.9867969
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.3278251, 2.3174129
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0782633, 2.0786853

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1206

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0138830, upper bound: 1.0164167
time: 3.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0125571, upper bound: 1.0171652
time: 3.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9706817, 1.9749284
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.8080873, 2.8033590
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3607793, 2.3564982
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.9568682, 1.9580407
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.6300259, 1.6268091
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.5936794, 1.6005781
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0780573, 2.0775108
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9825563, 1.9872961
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.3234000, 2.3218379
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0782704, 2.0786781

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1206

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0131904, upper bound: 1.0171088
time: 3.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0118645, upper bound: 1.0178574
time: 4.27 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9688606, 1.9767494
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.8078508, 2.8035955
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3604612, 2.3568163
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.9586320, 1.9562771
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.6295223, 1.6273136
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.5983958, 1.5958614
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0781665, 2.0774016
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9830565, 1.9867969
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.3278251, 2.3174129
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0782633, 2.0786853

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1206

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0126883, upper bound: 1.0170401
time: 3.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0119407, upper bound: 1.0183664
time: 3.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9706817, 1.9749284
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.8080873, 2.8033590
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3607793, 2.3564982
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.9568682, 1.9580407
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.6300259, 1.6268091
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.5936794, 1.6005781
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0780573, 2.0775108
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9825563, 1.9872961
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.3234000, 2.3218379
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0782704, 2.0786781

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1206

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0119957, upper bound: 1.0177321
time: 4.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0112481, upper bound: 1.0190585
time: 3.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9749284, 1.9706819
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.8033590, 2.8080869
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3564978, 2.3607793
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.9580407, 1.9568682
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.6268091, 1.6300259
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.6005783, 1.5936792
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0775108, 2.0780573
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9872961, 1.9825563
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.3218379, 2.3234000
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0786781, 2.0782704

Time for backsubstitution: 14.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1206

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0190569, upper bound: 1.0112496
time: 3.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0177306, upper bound: 1.0119972
time: 3.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9767494, 1.9688606
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.8035955, 2.8078513
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3568163, 2.3604612
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.9562774, 1.9586318
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.6273136, 1.6295223
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.5958614, 1.5983961
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0774016, 2.0781665
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9867969, 1.9830565
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.3174129, 2.3278251
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0786853, 2.0782633

Time for backsubstitution: 14.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1206

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0183649, upper bound: 1.0119422
time: 3.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0170386, upper bound: 1.0126899
time: 3.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9749284, 1.9706819
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.8033590, 2.8080869
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3564978, 2.3607793
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.9580407, 1.9568682
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.6268091, 1.6300259
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.6005783, 1.5936792
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0775108, 2.0780573
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9872961, 1.9825563
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.3218379, 2.3234000
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0786781, 2.0782704

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1206

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0178558, upper bound: 1.0118659
time: 3.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0171072, upper bound: 1.0131919
time: 4.49 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9767494, 1.9688606
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.8035955, 2.8078513
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3568163, 2.3604612
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.9562774, 1.9586318
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.6273136, 1.6295223
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.5958614, 1.5983961
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0774016, 2.0781665
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9867969, 1.9830565
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.3174129, 2.3278251
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0786853, 2.0782633

Time for backsubstitution: 14.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1206

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0171637, upper bound: 1.0125586
time: 3.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0164151, upper bound: 1.0138845
time: 3.66 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 21.85 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.85
Output dim: 7, lower bound: -1.0138830, upper bound: 1.0164167
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.85
Output dim: 7, lower bound: -1.0125571, upper bound: 1.0171652
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.85
Output dim: 7, lower bound: -1.0131904, upper bound: 1.0171088
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.85
Output dim: 7, lower bound: -1.0118645, upper bound: 1.0178574
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.85
Output dim: 7, lower bound: -1.0126883, upper bound: 1.0170401
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.85
Output dim: 7, lower bound: -1.0119407, upper bound: 1.0183664
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.85
Output dim: 7, lower bound: -1.0119957, upper bound: 1.0177321
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.85
Output dim: 7, lower bound: -1.0112481, upper bound: 1.0190585
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.85
Output dim: 7, lower bound: -1.0190569, upper bound: 1.0112496
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.85
Output dim: 7, lower bound: -1.0177306, upper bound: 1.0119972
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.85
Output dim: 7, lower bound: -1.0183649, upper bound: 1.0119422
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.85
Output dim: 7, lower bound: -1.0170386, upper bound: 1.0126899
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.85
Output dim: 7, lower bound: -1.0178558, upper bound: 1.0118659
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.85
Output dim: 7, lower bound: -1.0171072, upper bound: 1.0131919
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.85
Output dim: 7, lower bound: -1.0171637, upper bound: 1.0125586
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.85
Output dim: 7, lower bound: -1.0164151, upper bound: 1.0138845

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9731603, 1.9792356
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.8110008, 2.8077774
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3591900, 2.3566236
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.9637251, 1.9645352
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.6302028, 1.6305971
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.6022315, 1.6043680
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0779972, 2.0778904
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9820704, 1.9874611
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.3283629, 2.3233504
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0788779, 2.0793705

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1929

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0114769, upper bound: 1.0077690
time: 3.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0052154, upper bound: 1.0140039
time: 4.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9731679, 1.9792848
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.8134956, 2.8065090
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3608928, 2.3552270
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.9643664, 1.9631336
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.6314807, 1.6274900
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.6021857, 1.6045220
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0785456, 2.0783682
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9832211, 1.9888706
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.3293376, 2.3246417
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0789800, 2.0792923

Time for backsubstitution: 14.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1929

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0101510, upper bound: 1.0085172
time: 3.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0038896, upper bound: 1.0147528
time: 4.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9731603, 1.9792356
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.8110008, 2.8077774
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3591900, 2.3566236
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.9637251, 1.9645352
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.6302028, 1.6305971
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.6022315, 1.6043680
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0779972, 2.0778904
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9820704, 1.9874611
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.3283629, 2.3233504
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0788779, 2.0793705

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1929

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0107843, upper bound: 1.0084612
time: 3.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0045229, upper bound: 1.0146961
time: 4.27 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9731679, 1.9792848
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.8134956, 2.8065090
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3608928, 2.3552270
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.9643664, 1.9631336
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.6314807, 1.6274900
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.6021857, 1.6045220
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0785456, 2.0783682
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9832211, 1.9888706
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.3293376, 2.3246417
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0789800, 2.0792923

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1929

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0094584, upper bound: 1.0092094
time: 3.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0031970, upper bound: 1.0154449
time: 4.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9731603, 1.9792356
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.8110008, 2.8077774
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3591900, 2.3566236
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.9637251, 1.9645352
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.6302028, 1.6305971
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.6022315, 1.6043680
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0779972, 2.0778904
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9820704, 1.9874611
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.3283629, 2.3233504
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0788779, 2.0793705

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1929

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0102822, upper bound: 1.0083864
time: 3.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0040208, upper bound: 1.0146274
time: 4.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9731679, 1.9792848
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.8134956, 2.8065090
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3608928, 2.3552270
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.9643664, 1.9631336
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.6314807, 1.6274900
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.6021857, 1.6045220
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0785456, 2.0783682
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9832211, 1.9888706
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.3293376, 2.3246417
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0789800, 2.0792923

Time for backsubstitution: 14.73 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1929

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0095345, upper bound: 1.0097126
time: 4.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0032733, upper bound: 1.0159551
time: 3.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9731603, 1.9792356
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.8110008, 2.8077774
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3591900, 2.3566236
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.9637251, 1.9645352
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.6302028, 1.6305971
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.6022315, 1.6043680
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0779972, 2.0778904
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9820704, 1.9874611
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.3283629, 2.3233504
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0788779, 2.0793705

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1929

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0095896, upper bound: 1.0090784
time: 3.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0033282, upper bound: 1.0153208
time: 4.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9731679, 1.9792848
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.8134956, 2.8065090
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3608928, 2.3552270
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.9643664, 1.9631336
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.6314807, 1.6274900
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.6021857, 1.6045220
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0785456, 2.0783682
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9832211, 1.9888706
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.3293376, 2.3246417
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0789800, 2.0792923

Time for backsubstitution: 14.72 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1929

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0088419, upper bound: 1.0104046
time: 4.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0025806, upper bound: 1.0166472
time: 4.21 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9792280, 1.9731679
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.8065090, 2.8122687
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3552270, 2.3605866
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.9631338, 1.9651265
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.6274896, 1.6333094
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.6044135, 1.6021860
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0773416, 2.0785456
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9863110, 1.9832211
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.3223758, 2.3293381
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0792923, 2.0789557

Time for backsubstitution: 14.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1929

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0166455, upper bound: 1.0025822
time: 3.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0104031, upper bound: 1.0088423
time: 4.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9792356, 1.9732170
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.8090048, 2.8110008
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3569298, 2.3591900
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.9637752, 1.9637246
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.6287675, 1.6302023
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.6043682, 1.6023400
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0778899, 2.0790238
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9874611, 1.9846306
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.3233504, 2.3306293
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0793943, 2.0788779

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1929

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0153192, upper bound: 1.0033296
time: 3.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0090768, upper bound: 1.0095898
time: 4.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9792280, 1.9731679
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.8065090, 2.8122687
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3552270, 2.3605866
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.9631338, 1.9651265
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.6274896, 1.6333094
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.6044135, 1.6021860
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0773416, 2.0785456
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9863110, 1.9832211
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.3223758, 2.3293381
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0792923, 2.0789557

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1929

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0159535, upper bound: 1.0032748
time: 3.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0097111, upper bound: 1.0095367
time: 3.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9792356, 1.9732170
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.8090048, 2.8110008
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3569298, 2.3591900
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.9637752, 1.9637246
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.6287675, 1.6302023
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.6043682, 1.6023400
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0778899, 2.0790238
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9874611, 1.9846306
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.3233504, 2.3306293
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0793943, 2.0788779

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1929

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0146272, upper bound: 1.0040223
time: 3.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0083849, upper bound: 1.0102837
time: 4.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9792280, 1.9731679
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.8065090, 2.8122687
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3552270, 2.3605866
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.9631338, 1.9651265
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.6274896, 1.6333094
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.6044135, 1.6021860
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0773416, 2.0785456
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9863110, 1.9832211
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.3223758, 2.3293381
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0792923, 2.0789557

Time for backsubstitution: 14.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1929

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0154444, upper bound: 1.0031985
time: 4.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0092078, upper bound: 1.0094585
time: 4.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9792356, 1.9732170
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.8090048, 2.8110008
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3569298, 2.3591900
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.9637752, 1.9637246
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.6287675, 1.6302023
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.6043682, 1.6023400
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0778899, 2.0790238
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9874611, 1.9846306
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.3233504, 2.3306293
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0793943, 2.0788779

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1929

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0146958, upper bound: 1.0045234
time: 3.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0084596, upper bound: 1.0107859
time: 3.99 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9792280, 1.9731679
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.8065090, 2.8122687
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3552270, 2.3605866
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.9631338, 1.9651265
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.6274896, 1.6333094
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.6044135, 1.6021860
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0773416, 2.0785456
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9863110, 1.9832211
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.3223758, 2.3293381
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0792923, 2.0789557

Time for backsubstitution: 14.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1929

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0147523, upper bound: 1.0038911
time: 3.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0085157, upper bound: 1.0101511
time: 4.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9792356, 1.9732170
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.8090048, 2.8110008
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3569298, 2.3591900
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.9637752, 1.9637246
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.6287675, 1.6302023
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.6043682, 1.6023400
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0778899, 2.0790238
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9874611, 1.9846306
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.3233504, 2.3306293
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0793943, 2.0788779

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1929

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0140037, upper bound: 1.0052170
time: 3.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0077675, upper bound: 1.0114785
time: 4.01 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 22.36 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.36
Output dim: 7, lower bound: -1.0114769, upper bound: 1.0077690
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.36
Output dim: 7, lower bound: -1.0052154, upper bound: 1.0140039
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.36
Output dim: 7, lower bound: -1.0101510, upper bound: 1.0085172
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.36
Output dim: 7, lower bound: -1.0038896, upper bound: 1.0147528
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.36
Output dim: 7, lower bound: -1.0107843, upper bound: 1.0084612
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.36
Output dim: 7, lower bound: -1.0045229, upper bound: 1.0146961
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.36
Output dim: 7, lower bound: -1.0094584, upper bound: 1.0092094
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.36
Output dim: 7, lower bound: -1.0031970, upper bound: 1.0154449
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.36
Output dim: 7, lower bound: -1.0102822, upper bound: 1.0083864
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.36
Output dim: 7, lower bound: -1.0040208, upper bound: 1.0146274
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.36
Output dim: 7, lower bound: -1.0095345, upper bound: 1.0097126
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.36
Output dim: 7, lower bound: -1.0032733, upper bound: 1.0159551
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.36
Output dim: 7, lower bound: -1.0095896, upper bound: 1.0090784
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.36
Output dim: 7, lower bound: -1.0033282, upper bound: 1.0153208
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.36
Output dim: 7, lower bound: -1.0088419, upper bound: 1.0104046
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.36
Output dim: 7, lower bound: -1.0025806, upper bound: 1.0166472
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.36
Output dim: 7, lower bound: -1.0166455, upper bound: 1.0025822
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.36
Output dim: 7, lower bound: -1.0104031, upper bound: 1.0088423
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.36
Output dim: 7, lower bound: -1.0153192, upper bound: 1.0033296
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.36
Output dim: 7, lower bound: -1.0090768, upper bound: 1.0095898
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.36
Output dim: 7, lower bound: -1.0159535, upper bound: 1.0032748
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.36
Output dim: 7, lower bound: -1.0097111, upper bound: 1.0095367
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.36
Output dim: 7, lower bound: -1.0146272, upper bound: 1.0040223
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.36
Output dim: 7, lower bound: -1.0083849, upper bound: 1.0102837
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.36
Output dim: 7, lower bound: -1.0154444, upper bound: 1.0031985
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.36
Output dim: 7, lower bound: -1.0092078, upper bound: 1.0094585
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.36
Output dim: 7, lower bound: -1.0146958, upper bound: 1.0045234
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.36
Output dim: 7, lower bound: -1.0084596, upper bound: 1.0107859
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.36
Output dim: 7, lower bound: -1.0147523, upper bound: 1.0038911
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.36
Output dim: 7, lower bound: -1.0085157, upper bound: 1.0101511
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.36
Output dim: 7, lower bound: -1.0140037, upper bound: 1.0052170
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.36
Output dim: 7, lower bound: -1.0077675, upper bound: 1.0114785

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9597869, 1.9520464
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.7836685, 2.7745218
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3577080, 2.3498907
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.9394293, 1.9371467
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.5172157, 1.4987521
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.5820813, 1.5830674
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0763860, 2.0727403
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9411440, 1.9442930
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.2879062, 2.2785280
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0486007, 2.0320959

Time for backsubstitution: 14.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1929

### Candidate
type: RSZ, layer: 3, pos: 1206

### Candidate
type: RSZ, layer: 3, pos: 312

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0076581, upper bound: 1.0064228
time: 3.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0101743, upper bound: 1.0039907
time: 3.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9459786, 1.9658546
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.7790127, 2.7791777
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3538537, 2.3537450
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.9377379, 1.9388380
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.5014648, 1.5145035
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.5808849, 1.5842633
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0733962, 2.0757306
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9400530, 1.9453845
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.2845159, 2.2819183
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0316815, 2.0490155

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1929

### Candidate
type: RSZ, layer: 3, pos: 1206

### Candidate
type: RSZ, layer: 3, pos: 312

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0013966, upper bound: 1.0126581
time: 4.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0039128, upper bound: 1.0102258
time: 4.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9597869, 1.9520464
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.7836685, 2.7745218
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3577080, 2.3498907
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.9394293, 1.9371467
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.5172157, 1.4987521
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.5820813, 1.5830674
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0763860, 2.0727403
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9411440, 1.9442930
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.2879062, 2.2785280
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0486007, 2.0320959

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1929

### Candidate
type: RSZ, layer: 3, pos: 1206

### Candidate
type: RSZ, layer: 3, pos: 312

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0063321, upper bound: 1.0071701
time: 3.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0088483, upper bound: 1.0047386
time: 3.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9459786, 1.9658546
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.7790127, 2.7791777
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3538537, 2.3537450
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.9377379, 1.9388380
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.5014648, 1.5145035
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.5808849, 1.5842633
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0733962, 2.0757306
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9400530, 1.9453845
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.2845159, 2.2819183
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0316815, 2.0490155

Time for backsubstitution: 14.40 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=2.0185980796813965
rel_dist={7: [-1.0634246550232, 1.0634268200891506]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6181

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6181

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7908454, upper bound: 0.7927002
time: 8.99 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7927021, upper bound: 0.7908454
time: 4.25 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 13.41 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 13.41
Output dim: 7, lower bound: -0.7908454, upper bound: 0.7927002
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 13.41
Output dim: 7, lower bound: -0.7927021, upper bound: 0.7908454

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.7486548, 1.7506776
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.6175194, 2.6160221
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.1843939, 2.1830730
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.8109646, 1.8107674
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.4766479, 1.4757442
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.4101834, 1.4109106
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.8378420, 1.8376236
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.8445950, 1.8460083
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.1421781, 2.1401820
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -1.9854679, 1.9856062

Time for backsubstitution: 14.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.28 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7846232, upper bound: 0.7856255
time: 7.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7837707, upper bound: 0.7864834
time: 6.49 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.7506771, 1.7486548
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.6160221, 2.6175194
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.1830726, 2.1843939
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.8107677, 1.8109646
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.4757438, 1.4766483
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.4109106, 1.4101834
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.8376236, 1.8378420
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.8460083, 1.8445950
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.1401830, 2.1421781
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -1.9856062, 1.9854679

Time for backsubstitution: 14.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7864852, upper bound: 0.7837659
time: 8.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7837707, upper bound: 0.7846215
time: 5.03 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 27.82 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 27.82
Output dim: 7, lower bound: -0.7846232, upper bound: 0.7856255
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 27.82
Output dim: 7, lower bound: -0.7837707, upper bound: 0.7864834
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 27.82
Output dim: 7, lower bound: -0.7864852, upper bound: 0.7837659
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 27.82
Output dim: 7, lower bound: -0.7837707, upper bound: 0.7846215

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.7093763, 1.7115321
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.4754095, 2.4783010
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.1719518, 2.1697640
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.7031851, 1.7026129
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.4733748, 1.4755726
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.4105926, 1.4109349
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.8059330, 1.8077335
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.7355585, 1.7386713
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.0666332, 2.0604677
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -1.9260998, 1.9278741

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 1929

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7828882, upper bound: 0.7840252
time: 4.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7830239, upper bound: 0.7838681
time: 4.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.7095094, 1.7113991
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.4797983, 2.4739127
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.1710849, 2.1706305
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.7028103, 1.7029879
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.4764771, 1.4724708
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.4102073, 1.4113200
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.8079519, 1.8057146
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.7372580, 1.7369728
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.0624638, 2.0646381
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -1.9277358, 1.9262381

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1929

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7820118, upper bound: 0.7848830
time: 4.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7821637, upper bound: 0.7847507
time: 4.47 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.7113991, 1.7095094
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.4739122, 2.4797988
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.1706305, 2.1710849
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.7029881, 1.7028100
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.4724708, 1.4764767
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.4113202, 1.4102073
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.8057141, 1.8079524
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.7369728, 1.7372580
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.0646381, 2.0624638
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -1.9262381, 1.9277358

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1929

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7847470, upper bound: 0.7821673
time: 4.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7848838, upper bound: 0.7820117
time: 4.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.7115321, 1.7093763
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.4783010, 2.4754100
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.1697640, 2.1719513
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.7026129, 1.7031848
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.4755721, 1.4733753
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.4109349, 1.4105926
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.8077331, 1.8059330
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.7386713, 1.7355585
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.0604677, 2.0666337
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -1.9278741, 1.9261003

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1929

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7820155, upper bound: 0.7830240
time: 4.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7840226, upper bound: 0.7828919
time: 4.61 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 23.89 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.89
Output dim: 7, lower bound: -0.7828882, upper bound: 0.7840252
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.89
Output dim: 7, lower bound: -0.7830239, upper bound: 0.7838681
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.89
Output dim: 7, lower bound: -0.7820118, upper bound: 0.7848830
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.89
Output dim: 7, lower bound: -0.7821637, upper bound: 0.7847507
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.89
Output dim: 7, lower bound: -0.7847470, upper bound: 0.7821673
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.89
Output dim: 7, lower bound: -0.7848838, upper bound: 0.7820117
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.89
Output dim: 7, lower bound: -0.7820155, upper bound: 0.7830240
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.89
Output dim: 7, lower bound: -0.7840226, upper bound: 0.7828919

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.7442985, 1.7469277
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.6120319, 2.6106129
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.1839619, 2.1827474
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.8052306, 1.8044453
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.4750261, 1.4742894
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.4062395, 1.4053946
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.8369122, 1.8366575
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.8426876, 1.8439345
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.1393743, 2.1359034
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -1.9847512, 1.9848919

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1206

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7753473, upper bound: 0.7755904
time: 7.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7742240, upper bound: 0.7761712
time: 4.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.7449055, 1.7463207
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.6121101, 2.6105342
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.1840682, 2.1826410
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.8046427, 1.8050332
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.4751940, 1.4741211
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.4046674, 1.4069669
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.8368759, 1.8366938
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.8425212, 1.8441010
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.1378989, 2.1373782
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -1.9847536, 1.9848895

Time for backsubstitution: 14.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1206

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7753403, upper bound: 0.7755962
time: 6.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7742257, upper bound: 0.7761690
time: 4.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.7442985, 1.7469277
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.6120319, 2.6106129
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.1839619, 2.1827474
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.8052306, 1.8044453
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.4750261, 1.4742894
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.4062395, 1.4053946
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.8369122, 1.8366575
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.8426876, 1.8439345
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.1393743, 2.1359034
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -1.9847512, 1.9848919

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1206

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7742689, upper bound: 0.7761266
time: 5.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7736843, upper bound: 0.7772606
time: 5.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.7449055, 1.7463207
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.6121101, 2.6105342
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.1840682, 2.1826410
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.8046427, 1.8050332
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.4751940, 1.4741211
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.4046674, 1.4069669
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.8368759, 1.8366938
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.8425212, 1.8441010
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.1378989, 2.1373782
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -1.9847536, 1.9848895

Time for backsubstitution: 14.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1206

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7742761, upper bound: 0.7761268
time: 4.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7736787, upper bound: 0.7772621
time: 5.47 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.7463207, 1.7449055
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.6105337, 2.6121101
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.1826410, 2.1840682
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.8050332, 1.8046424
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.4741211, 1.4751940
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.4069667, 1.4046671
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.8366938, 1.8368759
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.8441010, 1.8425212
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.1373782, 2.1378989
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -1.9848895, 1.9847536

Time for backsubstitution: 14.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1206

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7772629, upper bound: 0.7736766
time: 5.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7761258, upper bound: 0.7742767
time: 5.48 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.7469277, 1.7442985
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.6106129, 2.6120319
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.1827474, 2.1839623
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.8044457, 1.8052304
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.4742899, 1.4750257
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.4053946, 1.4062393
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.8366570, 1.8369122
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.8439345, 1.8426876
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.1359029, 2.1393743
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -1.9848919, 1.9847512

Time for backsubstitution: 14.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1206

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7772621, upper bound: 0.7736826
time: 7.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7761286, upper bound: 0.7742647
time: 4.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.7463207, 1.7449055
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.6105337, 2.6121101
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.1826410, 2.1840682
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.8050332, 1.8046424
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.4741211, 1.4751940
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.4069667, 1.4046671
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.8366938, 1.8368759
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.8441010, 1.8425212
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.1373782, 2.1378989
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -1.9848895, 1.9847536

Time for backsubstitution: 14.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1206

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7761683, upper bound: 0.7742228
time: 5.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7755971, upper bound: 0.7753389
time: 9.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.7469277, 1.7442985
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.6106129, 2.6120319
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.1827474, 2.1839623
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.8044457, 1.8052304
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.4742899, 1.4750257
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.4053946, 1.4062393
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.8366570, 1.8369122
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.8439345, 1.8426876
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.1359029, 2.1393743
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -1.9848919, 1.9847512

Time for backsubstitution: 14.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1206

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7761754, upper bound: 0.7742210
time: 6.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7755916, upper bound: 0.7753503
time: 4.72 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 25.79 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 25.79
Output dim: 7, lower bound: -0.7753473, upper bound: 0.7755904
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 25.79
Output dim: 7, lower bound: -0.7742240, upper bound: 0.7761712
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 25.79
Output dim: 7, lower bound: -0.7753403, upper bound: 0.7755962
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 25.79
Output dim: 7, lower bound: -0.7742257, upper bound: 0.7761690
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 25.79
Output dim: 7, lower bound: -0.7742689, upper bound: 0.7761266
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 25.79
Output dim: 7, lower bound: -0.7736843, upper bound: 0.7772606
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 25.79
Output dim: 7, lower bound: -0.7742761, upper bound: 0.7761268
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 25.79
Output dim: 7, lower bound: -0.7736787, upper bound: 0.7772621
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 25.79
Output dim: 7, lower bound: -0.7772629, upper bound: 0.7736766
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 25.79
Output dim: 7, lower bound: -0.7761258, upper bound: 0.7742767
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 25.79
Output dim: 7, lower bound: -0.7772621, upper bound: 0.7736826
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 25.79
Output dim: 7, lower bound: -0.7761286, upper bound: 0.7742647
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 25.79
Output dim: 7, lower bound: -0.7761683, upper bound: 0.7742228
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 25.79
Output dim: 7, lower bound: -0.7755971, upper bound: 0.7753389
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 25.79
Output dim: 7, lower bound: -0.7761754, upper bound: 0.7742210
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 25.79
Output dim: 7, lower bound: -0.7755916, upper bound: 0.7753503
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=1.8785624504089355
rel_dist={7: [-0.7927072900922123, 0.7927031607168487]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6181

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6181

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9392352, upper bound: 0.9430627
time: 6.22 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9430632, upper bound: 0.9392348
time: 3.85 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 10.24 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 10.24
Output dim: 7, lower bound: -0.9392352, upper bound: 0.9430627
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 10.24
Output dim: 7, lower bound: -0.9430632, upper bound: 0.9392348

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.8609357, 1.8649812
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.7155075, 2.7125130
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.2726431, 2.2700014
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.8876653, 1.8872712
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.5540643, 1.5522556
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.5062613, 1.5077164
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.9584327, 1.9579959
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9146128, 1.9174395
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.2364030, 2.2324119
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0322242, 2.0325003

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.28 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9276496, upper bound: 0.9302916
time: 4.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9264749, upper bound: 0.9314686
time: 4.12 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.8649812, 1.8609362
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.7125130, 2.7155075
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.2700014, 2.2726431
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.8872714, 1.8876653
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.5522561, 1.5540643
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.5077162, 1.5062616
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.9579959, 1.9584332
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9174395, 1.9146128
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.2324128, 2.2364035
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0325003, 2.0322242

Time for backsubstitution: 14.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9314688, upper bound: 0.9264745
time: 4.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9302919, upper bound: 0.9276494
time: 4.02 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 22.92 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 22.92
Output dim: 7, lower bound: -0.9276496, upper bound: 0.9302916
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 22.92
Output dim: 7, lower bound: -0.9264749, upper bound: 0.9314686
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 22.92
Output dim: 7, lower bound: -0.9314688, upper bound: 0.9264745
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 22.92
Output dim: 7, lower bound: -0.9302919, upper bound: 0.9276494

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.8216577, 1.8259690
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.5733976, 2.5791802
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.2610674, 2.2566924
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.7802610, 1.7791166
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.5507913, 1.5551863
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.5070562, 1.5077405
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.9265237, 1.9301248
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.8038764, 1.8101025
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.1608601, 2.1485271
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -1.9728560, 1.9764037

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1929

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9226295, upper bound: 0.9245369
time: 4.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9220237, upper bound: 0.9251232
time: 4.27 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.8219237, 1.8257027
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.5821753, 2.5704041
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.2593346, 2.2584252
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.7795110, 1.7798665
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.5569949, 1.5489826
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.5062857, 1.5085111
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.9305620, 1.9260869
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.8072758, 1.8067055
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.1525192, 2.1568680
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -1.9761276, 1.9731326

Time for backsubstitution: 14.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1929

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9213796, upper bound: 0.9257853
time: 4.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9207838, upper bound: 0.9263801
time: 4.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.8257027, 1.8219240
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.5704041, 2.5821757
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.2584257, 2.2593341
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.7798667, 1.7795107
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.5489831, 1.5569944
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.5085111, 1.5062857
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.9260864, 1.9305620
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.8067055, 1.8072758
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.1568680, 2.1525192
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -1.9731321, 1.9761276

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 1929

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9263803, upper bound: 0.9207836
time: 4.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9257855, upper bound: 0.9213798
time: 4.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.8259692, 1.8216577
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.5791807, 2.5733981
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.2566924, 2.2610674
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.7791166, 1.7802606
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.5551858, 1.5507908
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.5077405, 1.5070562
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.9301243, 1.9265237
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.8101025, 1.8038764
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.1485271, 2.1608591
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -1.9764037, 1.9728560

Time for backsubstitution: 14.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 1929

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9251235, upper bound: 0.9220232
time: 4.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9245371, upper bound: 0.9226292
time: 4.56 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 23.88 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.88
Output dim: 7, lower bound: -0.9226295, upper bound: 0.9245369
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.88
Output dim: 7, lower bound: -0.9220237, upper bound: 0.9251232
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.88
Output dim: 7, lower bound: -0.9213796, upper bound: 0.9257853
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.88
Output dim: 7, lower bound: -0.9207838, upper bound: 0.9263801
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.88
Output dim: 7, lower bound: -0.9263803, upper bound: 0.9207836
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.88
Output dim: 7, lower bound: -0.9257855, upper bound: 0.9213798
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.88
Output dim: 7, lower bound: -0.9251235, upper bound: 0.9220232
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.88
Output dim: 7, lower bound: -0.9245371, upper bound: 0.9226292

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.8565793, 1.8618388
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.7099409, 2.7071042
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.2722116, 2.2697816
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.8819313, 1.8803611
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.5522738, 1.5508013
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.5023179, 1.5006280
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.9575391, 1.9570293
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9128718, 1.9153657
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.2335992, 2.2266583
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0315075, 2.0317883

Time for backsubstitution: 14.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1206

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9078707, upper bound: 0.9084016
time: 4.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9065034, upper bound: 0.9092566
time: 4.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.8577938, 1.8606248
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.7100983, 2.7069464
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.2724237, 2.2695694
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.8807554, 1.8815370
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.5526094, 1.5504656
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.4991732, 1.5037725
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.9574666, 1.9571023
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9125385, 1.9156985
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.2306495, 2.2296081
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0315123, 2.0317836

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1206

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9072796, upper bound: 0.9090613
time: 4.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9059136, upper bound: 0.9099045
time: 4.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.8565793, 1.8618388
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.7099409, 2.7071042
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.2722116, 2.2697816
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.8819313, 1.8803611
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.5522738, 1.5508013
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.5023179, 1.5006280
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.9575391, 1.9570293
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9128718, 1.9153657
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.2335992, 2.2266583
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0315075, 2.0317883

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1206

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9063167, upper bound: 0.9095272
time: 4.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9054759, upper bound: 0.9108482
time: 4.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.8577938, 1.8606248
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.7100983, 2.7069464
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.2724237, 2.2695694
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.8807554, 1.8815370
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.5526094, 1.5504656
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.4991732, 1.5037725
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.9574666, 1.9571023
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9125385, 1.9156985
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.2306495, 2.2296081
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0315123, 2.0317836

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1206

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9056786, upper bound: 0.9101542
time: 5.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9048395, upper bound: 0.9114888
time: 4.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.8606248, 1.8577938
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.7069464, 2.7100987
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.2695694, 2.2724237
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.8815370, 1.8807554
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.5504656, 1.5526094
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.5037723, 1.4991732
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.9571023, 1.9574666
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9156985, 1.9125385
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.2296081, 2.2306495
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0317836, 2.0315123

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1206

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9114891, upper bound: 0.9048390
time: 3.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9101551, upper bound: 0.9056789
time: 4.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.8618388, 1.8565793
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.7071037, 2.7099414
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.2697821, 2.2722116
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.8803611, 1.8819311
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.5508013, 1.5522738
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.5006280, 1.5023177
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.9570293, 1.9575396
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9153657, 1.9128723
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.2266583, 2.2335997
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0317883, 2.0315075

Time for backsubstitution: 14.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1206

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9108489, upper bound: 0.9054757
time: 3.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9095281, upper bound: 0.9063166
time: 4.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.8606248, 1.8577938
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.7069464, 2.7100987
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.2695694, 2.2724237
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.8815370, 1.8807554
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.5504656, 1.5526094
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.5037723, 1.4991732
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.9571023, 1.9574666
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9156985, 1.9125385
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.2296081, 2.2306495
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0317836, 2.0315123

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1206

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9099050, upper bound: 0.9059132
time: 4.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9090623, upper bound: 0.9072790
time: 4.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.8618388, 1.8565793
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.7071037, 2.7099414
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.2697821, 2.2722116
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.8803611, 1.8819311
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.5508013, 1.5522738
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.5006280, 1.5023177
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.9570293, 1.9575396
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9153657, 1.9128723
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.2266583, 2.2335997
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0317883, 2.0315075

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1206

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9092546, upper bound: 0.9065029
time: 4.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9084021, upper bound: 0.9078705
time: 4.28 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 22.87 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.87
Output dim: 7, lower bound: -0.9078707, upper bound: 0.9084016
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.87
Output dim: 7, lower bound: -0.9065034, upper bound: 0.9092566
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.87
Output dim: 7, lower bound: -0.9072796, upper bound: 0.9090613
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.87
Output dim: 7, lower bound: -0.9059136, upper bound: 0.9099045
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.87
Output dim: 7, lower bound: -0.9063167, upper bound: 0.9095272
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.87
Output dim: 7, lower bound: -0.9054759, upper bound: 0.9108482
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.87
Output dim: 7, lower bound: -0.9056786, upper bound: 0.9101542
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.87
Output dim: 7, lower bound: -0.9048395, upper bound: 0.9114888
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.87
Output dim: 7, lower bound: -0.9114891, upper bound: 0.9048390
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.87
Output dim: 7, lower bound: -0.9101551, upper bound: 0.9056789
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.87
Output dim: 7, lower bound: -0.9108489, upper bound: 0.9054757
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.87
Output dim: 7, lower bound: -0.9095281, upper bound: 0.9063166
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.87
Output dim: 7, lower bound: -0.9099050, upper bound: 0.9059132
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.87
Output dim: 7, lower bound: -0.9090623, upper bound: 0.9072790
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.87
Output dim: 7, lower bound: -0.9092546, upper bound: 0.9065029
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.87
Output dim: 7, lower bound: -0.9084021, upper bound: 0.9078705

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.8608818, 1.8649321
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.7130127, 2.7108636
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.2709408, 2.2692299
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.8870239, 1.8875642
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.5527864, 1.5530496
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.5061378, 1.5075624
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.9575891, 1.9575176
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9124365, 1.9160299
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.2344623, 2.2311206
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0321217, 2.0324502

Time for backsubstitution: 14.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1929

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9070632, upper bound: 0.9020933
time: 4.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9013627, upper bound: 0.9075481
time: 3.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.8608866, 1.8649268
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.7138577, 2.7100182
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.2718716, 2.2682986
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.8879585, 1.8866298
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.5548577, 1.5509777
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.5061078, 1.5075927
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.9579549, 1.9571524
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9132032, 1.9152632
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.2351127, 2.2304707
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0321736, 2.0323982

Time for backsubstitution: 14.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1929

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9056753, upper bound: 0.9029078
time: 4.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.8999991, upper bound: 0.9084382
time: 4.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.8608818, 1.8649321
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.7130127, 2.7108636
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.2709408, 2.2692299
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.8870239, 1.8875642
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.5527864, 1.5530496
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.5061378, 1.5075624
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.9575891, 1.9575176
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9124365, 1.9160299
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.2344623, 2.2311206
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0321217, 2.0324502

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1929

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9064610, upper bound: 0.9027130
time: 4.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9007847, upper bound: 0.9082125
time: 4.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.8608866, 1.8649268
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.7138577, 2.7100182
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.2718716, 2.2682986
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.8879585, 1.8866298
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.5548577, 1.5509777
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.5061078, 1.5075927
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.9579549, 1.9571524
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9132032, 1.9152632
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.2351127, 2.2304707
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0321736, 2.0323982

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1929

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9050673, upper bound: 0.9035199
time: 4.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.8994082, upper bound: 0.9090785
time: 4.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.8608818, 1.8649321
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.7130127, 2.7108636
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.2709408, 2.2692299
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.8870239, 1.8875642
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.5527864, 1.5530496
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.5061378, 1.5075624
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.9575891, 1.9575176
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9124365, 1.9160299
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.2344623, 2.2311206
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0321217, 2.0324502

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1929

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9055135, upper bound: 0.9031181
time: 4.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.8998292, upper bound: 0.9086849
time: 5.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.8608866, 1.8649268
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.7138577, 2.7100182
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.2718716, 2.2682986
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.8879585, 1.8866298
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.5548577, 1.5509777
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.5061078, 1.5075927
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.9579549, 1.9571524
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9132032, 1.9152632
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.2351127, 2.2304707
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0321736, 2.0323982

Time for backsubstitution: 14.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1929

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9046325, upper bound: 0.9044840
time: 4.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.8990169, upper bound: 0.9100182
time: 4.47 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.8608818, 1.8649321
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.7130127, 2.7108636
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.2709408, 2.2692299
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.8870239, 1.8875642
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.5527864, 1.5530496
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.5061378, 1.5075624
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.9575891, 1.9575176
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9124365, 1.9160299
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.2344623, 2.2311206
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0321217, 2.0324502

Time for backsubstitution: 14.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1929

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9048770, upper bound: 0.9037053
time: 4.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.8992437, upper bound: 0.9093118
time: 4.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.8608866, 1.8649268
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.7138577, 2.7100182
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.2718716, 2.2682986
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.8879585, 1.8866298
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.5548577, 1.5509777
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.5061078, 1.5075927
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.9579549, 1.9571524
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9132032, 1.9152632
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.2351127, 2.2304707
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0321736, 2.0323982

Time for backsubstitution: 14.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1929

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9039930, upper bound: 0.9050711
time: 4.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.8984215, upper bound: 0.9106599
time: 4.07 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.8649268, 1.8608871
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.7100182, 2.7138577
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.2682986, 2.2718716
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.8866301, 1.8879585
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.5509782, 1.5548582
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.5075927, 1.5061076
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.9571524, 1.9579549
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9152632, 1.9132032
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.2304702, 2.2351127
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0323982, 2.0321736

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1929

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9106602, upper bound: 0.8984214
time: 4.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9050713, upper bound: 0.9039926
time: 4.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.8649321, 1.8608818
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.7108631, 2.7130122
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.2692299, 2.2709403
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.8875647, 1.8870239
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.5530496, 1.5527864
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.5075622, 1.5061378
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.9575176, 1.9575891
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9160299, 1.9124365
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.2311206, 2.2344623
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0324502, 2.0321217

Time for backsubstitution: 14.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1929

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9093121, upper bound: 0.8992443
time: 4.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9037058, upper bound: 0.9048772
time: 4.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.8649268, 1.8608871
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.7100182, 2.7138577
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.2682986, 2.2718716
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.8866301, 1.8879585
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.5509782, 1.5548582
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.5075927, 1.5061076
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.9571524, 1.9579549
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9152632, 1.9132032
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.2304702, 2.2351127
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0323982, 2.0321736

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1929

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9100184, upper bound: 0.8990171
time: 4.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9044845, upper bound: 0.9046321
time: 4.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.8649321, 1.8608818
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.7108631, 2.7130122
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.2692299, 2.2709403
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.8875647, 1.8870239
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.5530496, 1.5527864
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.5075622, 1.5061378
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.9575176, 1.9575891
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9160299, 1.9124365
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.2311206, 2.2344623
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0324502, 2.0321217

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1929

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9086858, upper bound: 0.8998287
time: 4.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9031190, upper bound: 0.9055124
time: 4.19 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.8649268, 1.8608871
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.7100182, 2.7138577
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.2682986, 2.2718716
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.8866301, 1.8879585
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.5509782, 1.5548582
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.5075927, 1.5061076
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.9571524, 1.9579549
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9152632, 1.9132032
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.2304702, 2.2351127
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0323982, 2.0321736

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1929

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9090796, upper bound: 0.8994078
time: 4.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9035201, upper bound: 0.9050676
time: 4.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.8649321, 1.8608818
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.7108631, 2.7130122
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.2692299, 2.2709403
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.8875647, 1.8870239
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.5530496, 1.5527864
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.5075622, 1.5061378
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.9575176, 1.9575891
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9160299, 1.9124365
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.2311206, 2.2344623
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0324502, 2.0321217

Time for backsubstitution: 14.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1929

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9082135, upper bound: 0.9007843
time: 4.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9027132, upper bound: 0.9064600
time: 4.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.8649268, 1.8608871
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.7100182, 2.7138577
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.2682986, 2.2718716
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.8866301, 1.8879585
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.5509782, 1.5548582
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.5075927, 1.5061076
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.9571524, 1.9579549
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9152632, 1.9132032
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.2304702, 2.2351127
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0323982, 2.0321736

Time for backsubstitution: 14.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1929

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9084391, upper bound: 0.8999992
time: 4.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9029085, upper bound: 0.9056751
time: 4.44 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.8649321, 1.8608818
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.7108631, 2.7130122
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.2692299, 2.2709403
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.8875647, 1.8870239
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.5530496, 1.5527864
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.5075622, 1.5061378
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.9575176, 1.9575891
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9160299, 1.9124365
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.2311206, 2.2344623
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0324502, 2.0321217

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1929

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9075490, upper bound: 0.9013620
time: 4.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9020943, upper bound: 0.9070648
time: 3.90 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 22.74 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.74
Output dim: 7, lower bound: -0.9070632, upper bound: 0.9020933
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.74
Output dim: 7, lower bound: -0.9013627, upper bound: 0.9075481
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.74
Output dim: 7, lower bound: -0.9056753, upper bound: 0.9029078
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.74
Output dim: 7, lower bound: -0.8999991, upper bound: 0.9084382
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.74
Output dim: 7, lower bound: -0.9064610, upper bound: 0.9027130
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.74
Output dim: 7, lower bound: -0.9007847, upper bound: 0.9082125
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.74
Output dim: 7, lower bound: -0.9050673, upper bound: 0.9035199
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.74
Output dim: 7, lower bound: -0.8994082, upper bound: 0.9090785
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.74
Output dim: 7, lower bound: -0.9055135, upper bound: 0.9031181
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.74
Output dim: 7, lower bound: -0.8998292, upper bound: 0.9086849
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.74
Output dim: 7, lower bound: -0.9046325, upper bound: 0.9044840
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.74
Output dim: 7, lower bound: -0.8990169, upper bound: 0.9100182
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.74
Output dim: 7, lower bound: -0.9048770, upper bound: 0.9037053
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.74
Output dim: 7, lower bound: -0.8992437, upper bound: 0.9093118
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.74
Output dim: 7, lower bound: -0.9039930, upper bound: 0.9050711
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.74
Output dim: 7, lower bound: -0.8984215, upper bound: 0.9106599
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.74
Output dim: 7, lower bound: -0.9106602, upper bound: 0.8984214
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.74
Output dim: 7, lower bound: -0.9050713, upper bound: 0.9039926
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.74
Output dim: 7, lower bound: -0.9093121, upper bound: 0.8992443
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.74
Output dim: 7, lower bound: -0.9037058, upper bound: 0.9048772
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.74
Output dim: 7, lower bound: -0.9100184, upper bound: 0.8990171
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.74
Output dim: 7, lower bound: -0.9044845, upper bound: 0.9046321
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.74
Output dim: 7, lower bound: -0.9086858, upper bound: 0.8998287
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.74
Output dim: 7, lower bound: -0.9031190, upper bound: 0.9055124
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.74
Output dim: 7, lower bound: -0.9090796, upper bound: 0.8994078
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.74
Output dim: 7, lower bound: -0.9035201, upper bound: 0.9050676
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.74
Output dim: 7, lower bound: -0.9082135, upper bound: 0.9007843
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.74
Output dim: 7, lower bound: -0.9027132, upper bound: 0.9064600
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.74
Output dim: 7, lower bound: -0.9084391, upper bound: 0.8999992
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.74
Output dim: 7, lower bound: -0.9029085, upper bound: 0.9056751
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.74
Output dim: 7, lower bound: -0.9075490, upper bound: 0.9013620
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.74
Output dim: 7, lower bound: -0.9020943, upper bound: 0.9070648

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.8429027, 1.8377428
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.6841297, 2.6780310
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.2681737, 2.2629619
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.8627286, 1.8612065
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.4345493, 1.4222403
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.4860029, 1.4866602
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.9547987, 1.9523680
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.8707628, 1.8728619
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.1936803, 2.1874282
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -1.9962049, 1.9852018

Time for backsubstitution: 14.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1929

### Candidate
type: RSZ, layer: 3, pos: 1206

### Candidate
type: RSZ, layer: 3, pos: 312

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9040495, upper bound: 0.9006211
time: 4.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9056350, upper bound: 0.8991713
time: 4.24 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.8336973, 1.8469481
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.6810246, 2.6811347
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.2656040, 2.2655315
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.8616009, 1.8623343
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.4240484, 1.4327412
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.4852057, 1.4874575
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.9528050, 1.9543617
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.8700352, 1.8735895
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.1914201, 2.1896884
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -1.9849253, 1.9964814

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Candidate
type: RSZ, layer: 3, pos: 1929

### Candidate
type: RSZ, layer: 3, pos: 1206

### Candidate
type: RSZ, layer: 3, pos: 312

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.8983597, upper bound: 0.9060895
time: 4.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.8999516, upper bound: 0.9046423
time: 4.10 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 22.80 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 22.80
Output dim: 7, lower bound: -0.9040495, upper bound: 0.9006211
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 22.80
Output dim: 7, lower bound: -0.9056350, upper bound: 0.8991713
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 22.80
Output dim: 7, lower bound: -0.8983597, upper bound: 0.9060895
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 22.80
Output dim: 7, lower bound: -0.8999516, upper bound: 0.9046423
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 7, lower bound: -0.9056753, upper bound: 0.9029078
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 7, lower bound: -0.8999991, upper bound: 0.9084382
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 7, lower bound: -0.9064610, upper bound: 0.9027130
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 7, lower bound: -0.9007847, upper bound: 0.9082125
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 7, lower bound: -0.9050673, upper bound: 0.9035199
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 7, lower bound: -0.8994082, upper bound: 0.9090785
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 7, lower bound: -0.9055135, upper bound: 0.9031181
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 7, lower bound: -0.8998292, upper bound: 0.9086849
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 7, lower bound: -0.9046325, upper bound: 0.9044840
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 7, lower bound: -0.8990169, upper bound: 0.9100182
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 7, lower bound: -0.9048770, upper bound: 0.9037053
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 7, lower bound: -0.8992437, upper bound: 0.9093118
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 7, lower bound: -0.9039930, upper bound: 0.9050711
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 7, lower bound: -0.8984215, upper bound: 0.9106599
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 7, lower bound: -0.9106602, upper bound: 0.8984214
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 7, lower bound: -0.9050713, upper bound: 0.9039926
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 7, lower bound: -0.9093121, upper bound: 0.8992443
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 7, lower bound: -0.9037058, upper bound: 0.9048772
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 7, lower bound: -0.9100184, upper bound: 0.8990171
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 7, lower bound: -0.9044845, upper bound: 0.9046321
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 7, lower bound: -0.9086858, upper bound: 0.8998287
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 7, lower bound: -0.9031190, upper bound: 0.9055124
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 7, lower bound: -0.9090796, upper bound: 0.8994078
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 7, lower bound: -0.9035201, upper bound: 0.9050676
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 7, lower bound: -0.9082135, upper bound: 0.9007843
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 7, lower bound: -0.9027132, upper bound: 0.9064600
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 7, lower bound: -0.9084391, upper bound: 0.8999992
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 7, lower bound: -0.9029085, upper bound: 0.9056751
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 7, lower bound: -0.9075490, upper bound: 0.9013620
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.80
Output dim: 7, lower bound: -0.9020943, upper bound: 0.9070648
Binary search (step 3): status=Status.UNKNOWN, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=1.948580265045166
rel_dist={7: [-0.9430664895693686, 0.9430688356373857]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.00390625
execution time: 2851.26 seconds
